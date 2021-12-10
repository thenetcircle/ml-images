from PIL import ImageFile
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2

ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings('ignore')

import arrow
import os
import sys
import json

from loguru import logger
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
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


# either distributed or single-node; need to create this at program startup
strategy = create_strategy()

# suppress image loading warnings, messes up the output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# turn of interactive plotting mode, we'll only be saving the images to disk
plt.ioff()

# for saving plots and weights
now = arrow.utcnow().format('YYMMDD_HHmm')

NUM_CLASSES = 3
IMG_SIZE = 299
N_LAYERS_UNFREEZE = 0
BATCH_SIZE = 128
LEARNING_RATE_INITIAL = 1e-2
LEARNING_RATE_TRANSFER = 1e-4
EPOCHS_INITIAL = 25
EPOCHS_TRANSFER = 150

KERAS_F_STR = "{val_categorical_accuracy:.5f}_{epoch:02d}"

input_dir = sys.argv[1]
output_dir = sys.argv[2]
log_dir = f"{output_dir}/logs/{now}/"
checkpoint_path = f"{output_dir}/model_ckpt_irnv2_{KERAS_F_STR}_{now}.h5"
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
    patience=50,
    verbose=1,
    mode='auto'
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_categorical_accuracy',
    factor=0.5,
    patience=25,
    verbose=1
)
logging = TensorBoard(
    log_dir=log_dir
)


def create_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    input_tensor = img_augmentation(inputs)
    input_tensor = tf.keras.applications.inception_resnet_v2.preprocess_input(input_tensor)

    headless_model = InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', input_tensor=input_tensor)

    # freeze the conv layers first so we can train our new top model first
    headless_model.trainable = False

    # freeze the conv layers first so we can train out new top model first
    for layer in headless_model.layers:
        layer.trainable = False

    # put our own top model to classify our custom classes
    x = headless_model.output
    x = Dense(256, activation="relu")(x)
    predictions = Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    model_final = tf.keras.Model(inputs=headless_model.input, outputs=predictions)
    model_final.compile(
        loss='categorical_crossentropy',
        # here we use adam since the weights for these new layers are
        # randomly initialized, later we use SGD with a small learning rate
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_INITIAL),
        metrics=[metrics.mae, metrics.categorical_accuracy]
    )

    return model_final


def unfreeze_model(m):
    # model needs to be set to trainable as well
    m.trainable = True
    freeze = True
    n_frozen = 0

    for layer in m.layers:
        should_freeze = freeze or isinstance(layer, BatchNormalization)
        layer.trainable = not should_freeze

        if should_freeze:
            n_frozen += 1

        # block8_1_mixed is layer 633 out of 783
        if layer.name == 'block8_1_mixed':  # mixed4/6 for InceptionV3
            freeze = False

    print(f'froze the first {n_frozen} out of {len(m.layers)} layers')

    # have to re-compile after changing layers; also, use a smaller learning rate to not mess up the weights
    m.compile(
        optimizer=optimizers.SGD(lr=LEARNING_RATE_TRANSFER, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=[metrics.mae, metrics.categorical_accuracy]
    )


if __name__ == "__main__":
    # for potential multi-server training
    with strategy.scope():
        model = create_model()

    ds_train = create_dataset(train_dir)
    ds_test = create_dataset(valid_dir)

    weights = None
    if len(sys.argv) > 3:
        weights = sys.argv[3]

    if weights is None:
        hist_initial = model.fit(
            ds_train,
            epochs=EPOCHS_INITIAL,
            validation_data=ds_test,
            verbose=1,
            callbacks=[reduce_lr, early]
        )
        print(f"possible hist_initial keys: {hist_initial.history.keys()}")
        model.save_weights(f"{output_dir}/model_irnv2_init_{now}.h5")
        plot_hist("initial", hist_initial)

    else:
        print(f"loading saved weights: {weights}")
        model.load_weights(weights)

    # train a few more layers
    with strategy.scope():
        unfreeze_model(model)

    hist_transfer = model.fit(
        ds_train,
        epochs=EPOCHS_TRANSFER,
        validation_data=ds_test,
        verbose=1,
        callbacks=[logging, checkpoint, reduce_lr, early]
    )
    print(f"possible hist_transfer keys: {hist_transfer.history.keys()}")

    model.save_weights(f"{output_dir}/model_irnv2_tl_{now}.h5")
    plot_hist("transfer", hist_transfer)
