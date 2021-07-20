from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')

import arrow
import os
import sys
from loguru import logger
from tensorflow.keras.applications.efficientnet import *
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import matplotlib.pyplot as plt


# TODO: try using half-precision; requires half the memory and runs twice as fast
# probably not possible with pre-trained weights, would need to train from scratch, and efficientnet takes ages to train
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')

# TODO: combine subset dirs with regular dirs to get more data
# subset_train_dir = "/usr/local/data/imgs/subset_train"
# subset_valid_dir = "/usr/local/data/imgs/subset_valid"
# train_dir = "/usr/local/data/imgs/train"
# valid_dir = "/usr/local/data/imgs/valid"


def check_dir(directory, should_raise: bool = False):
    if not os.path.exists(directory):
        if should_raise:
            raise RuntimeError(f"specified dir '{directory}' does not exist")
        os.mkdir(directory)


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
ENET_MODEL_VERSION = 7
NUM_CLASSES = 3
N_LAYERS_UNFREEZE = 20
BATCH_SIZE = 256
EPOCHS_INITIAL = 20
EPOCHS_TRANSFER = 20

IMG_SIZE = ENET_IMG_SIZES[ENET_MODEL_VERSION]
KERAS_F_STR = "{epoch:02d}_{val_categorical_accuracy:.5f}"

# choose keras class based on model version
EfficientNetBx = ENET_MODEL_CLASSES[ENET_MODEL_VERSION]

input_dir = sys.argv[1]
output_dir = sys.argv[2]
log_dir = f"{output_dir}/logs/{now}/"
checkpoint_path = f"{output_dir}/model_ckpt_efficientnet_b{ENET_MODEL_VERSION}_{KERAS_F_STR}_{now}.h5"
train_dir = f"{input_dir}/train"
valid_dir = f"{input_dir}/valid"

check_dir(input_dir, should_raise=True)
check_dir(output_dir)
check_dir(log_dir)

# TODO: setup multi-worker training
"""
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"]
    },
   "task": {"type": "worker", "index": 1}
})

# distribute training on multiple machines, need to set TF_CONFIG on all hosts
strategy = tf.distribute.MultiWorkerMirroredStrategy(
    communication_options=tf.distribute.experimental.CommunicationOptions(
        implementation=tf.distribute.experimental.CommunicationImplementation.AUTO
    )
)
"""

strategy = tf.distribute.MirroredStrategy()

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


def create_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    input_tensor = img_augmentation(inputs)

    # use the headless model with "noisy student" pre-trained weights instead of imagenet weights
    headless_model = EfficientNetBx(
        include_top=False,
        input_tensor=input_tensor,
        weights=f"data/noisy-student-efficientnet-b{ENET_MODEL_VERSION}-notop.h5"
    )

    # freeze the conv layers first so we can train out new top model first
    headless_model.trainable = False

    # rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(headless_model.output)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # compile the model
    model_final = tf.keras.Model(inputs, outputs, name="EfficientNet")
    model_final.compile(
        optimizer=optimizers.Adam(learning_rate=1e-2),
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
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    # have to re-compile after changing layers; also, use a smaller learning rate to not mess up the weights
    m.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
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
