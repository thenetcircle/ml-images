from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')

import arrow
import os
import sys
import json

from loguru import logger
from tensorflow.keras.applications.efficientnet import *
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import mixed_precision
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import get_custom_objects
import tensorflow as tf
import matplotlib.pyplot as plt


def check_dir(directory, should_raise: bool = False):
    if not os.path.exists(directory):
        if should_raise:
            raise RuntimeError(f"specified dir '{directory}' does not exist")
        os.mkdir(directory)


def create_strategy():
    """
    TF_CONFIG example:
    {
        "cluster": {
            "worker": ["host1:port", "host2:port", "host3:port"]
        },
        "task": {
            "type": "worker",
            "index": 0
        }
    }
    """
    if "TF_CONFIG_FILE" in os.environ:
        logger.info("using distributed training...")

        with open(os.environ["TF_CONFIG_FILE"], 'r') as f:
            os.environ["TF_CONFIG"] = json.dumps(json.load(f))

        # distribute training on multiple machines, need to set TF_CONFIG on all hosts
        return tf.distribute.MultiWorkerMirroredStrategy(
            communication_options=tf.distribute.experimental.CommunicationOptions(
                implementation=tf.distribute.experimental.CommunicationImplementation.AUTO
            )
        )

    logger.info("using single node training...")
    return tf.distribute.MirroredStrategy()


def create_dataset(directory):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=True,
        interpolation="bilinear",
        follow_links=False,
        smart_resize=False
    )


class SwishActivation(Activation):
    def __init__(self, activation, **kwargs):
        super().__init__(activation, **kwargs)
        self.__name__ = 'swish_act'


# https://towardsdatascience.com/comparison-of-activation-functions-for-deep-neural-networks-706ac4284c8a
def swish_act(x, beta=1):
    return x * sigmoid(beta * x)


# either distributed or single-node; need to create this at program startup
strategy = create_strategy()

# register our custom activation
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

# TODO: issues with mixed precision on tf 2.5 and efficientnet
# use both float32 and float16; float16 will be faster on the gpu, but some layers needs the
# numerical stability provided by float32
# mixed_precision.set_global_policy('mixed_float16')

# suppress image loading warnings, messes up the output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# resolutions for EfficientNet; larger images requires more gpu memory / smaller batch size
ENET_IMG_SIZES = {
    0: 224,
    1: 240,
    2: 260,
    3: 300,
    4: 380,
    5: 456,
    6: 528,
    7: 600
}
ENET_MODEL_CLASSES = {
    0: EfficientNetB0,
    1: EfficientNetB1,
    2: EfficientNetB2,
    3: EfficientNetB3,
    4: EfficientNetB4,
    5: EfficientNetB5,
    6: EfficientNetB6,
    7: EfficientNetB7
}

# turn of interactive plotting mode, we'll only be saving the images to disk
plt.ioff()

# for saving plots and weights
now = arrow.utcnow().format('YYMMDD_HHmm')

# chooses the model size; larger is better, but requires a lot more memory and compute
ENET_MODEL_VERSION = 3
NUM_CLASSES = 3
N_LAYERS_UNFREEZE = 20
BATCH_SIZE = 1024
LEARNING_RATE = BATCH_SIZE / 16_000
EPOCHS_INITIAL = 8
EPOCHS_TRANSFER = 50

IMG_SIZE = ENET_IMG_SIZES[ENET_MODEL_VERSION]
KERAS_F_STR = "{epoch:02d}_{val_categorical_accuracy:.5f}"

# choose keras class based on model version
EfficientNetBx = ENET_MODEL_CLASSES[ENET_MODEL_VERSION]

input_dir = sys.argv[1]
output_dir = sys.argv[2]
log_dir = f"{output_dir}/logs/{now}/"
checkpoint_path = f"{output_dir}/model_ckpt_efficientnet_b{ENET_MODEL_VERSION}_{now}_{KERAS_F_STR}.h5"
train_dir = f"{input_dir}/train"
valid_dir = f"{input_dir}/valid"

check_dir(input_dir, should_raise=True)
check_dir(output_dir)
check_dir(log_dir)


img_augmentation = Sequential(
    [
        preprocessing.RandomFlip(),
        preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_categorical_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1
)
early = EarlyStopping(
    monitor='val_categorical_accuracy',
    min_delta=0,
    patience=1500,
    verbose=1,
    mode='auto'
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_categorical_accuracy',
    factor=0.5,
    patience=50,
    verbose=1
)
logging = TensorBoard(
    log_dir=log_dir
)


def create_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    input_tensor = img_augmentation(inputs)

    # use the headless model with "noisy student" pre-trained weights instead of imagenet weights
    headless_model = EfficientNetBx(
        include_top=False,
        input_tensor=input_tensor,
        weights=f"data/noisy-student-efficientnet-b{ENET_MODEL_VERSION}-notop.h5"
    )

    # freeze the conv layers first so we can train our new top model first
    headless_model.trainable = False

    # rebuild top
    x = headless_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation(swish_act)(x)
    x = Dropout(0.2)(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation(swish_act)(x)

    # when using mixed precision, we need to separate the output layer and the activation, since
    # softmax on fp16 is not numerically stable, while Dense fp16 IS:
    #
    #   Adding a float16 softmax in the middle of a model is fine, but a softmax at the end of the
    #   model should be in float32. The reason is that if the intermediate tensor flowing from the
    #   softmax to the loss is float16 or bfloat16, numeric issues may occur.
    #
    # more info: https://www.tensorflow.org/guide/mixed_precision
    # TODO: issues with fp16 on efficientnet and tf 2.5
    # x = Dense(NUM_CLASSES, name="dense_logits")(x)
    # outputs = Activation('softmax', dtype='float32', name='predictions')(x)

    outputs = Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # compile the model
    model_final = tf.keras.Model(inputs, outputs, name="EfficientNet")
    model_final.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[metrics.mae, metrics.categorical_accuracy]
    )

    return model_final


def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label


def plot_hist(prefix, hist):
    # clear previous plot
    plt.figure()

    plt.plot(hist.history["categorical_accuracy"])
    plt.plot(hist.history["val_categorical_accuracy"])
    plt.title(f"{prefix} model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")

    plt.savefig(f"{output_dir}/hist-{now}-{prefix}.png")


def unfreeze_model(m, n_layers=20):
    # model needs to be set to trainable as well
    m.trainable = True

    # freeze the majority of layers
    for layer in m.layers[:-n_layers]:
        layer.trainable = False

    # unfreeze the top n_layers layers while leaving BatchNorm layers frozen
    for layer in m.layers[-n_layers:]:
        layer.trainable = not isinstance(layer, BatchNormalization)

    # have to re-compile after changing layers; also, use a smaller learning rate to not mess up the weights
    m.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[metrics.mae, metrics.categorical_accuracy]
    )


if __name__ == "__main__":
    # for potential multi-server training
    with strategy.scope():
        model = create_model()

    ds_train = create_dataset(train_dir)
    ds_test = create_dataset(valid_dir)

    hist_initial = model.fit(
        ds_train,
        epochs=EPOCHS_INITIAL,
        validation_data=ds_test,
        verbose=1,
        callbacks=[reduce_lr, early]
    )
    logger.info(f"possible hist_initial keys: {hist_initial.history.keys()}")

    model.save_weights(f"{output_dir}/model_efficientnet_b{ENET_MODEL_VERSION}_init_{now}.h5")
    plot_hist("initial", hist_initial)

    # train a few more layers
    with strategy.scope():
        unfreeze_model(model, n_layers=N_LAYERS_UNFREEZE)

    hist_transfer = model.fit(
        ds_train,
        epochs=EPOCHS_TRANSFER,
        validation_data=ds_test,
        verbose=1,
        callbacks=[logging, checkpoint, reduce_lr, early]
    )
    logger.info(f"possible hist_transfer keys: {hist_initial.history.keys()}")

    model.save_weights(f"{output_dir}/model_efficientnet_b{ENET_MODEL_VERSION}_tl_{now}.h5")
    plot_hist("transfer", hist_initial)
