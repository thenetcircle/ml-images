from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from fastapi import FastAPI
from pydantic import BaseModel
import sys
import numpy as np
import skimage
import tensorflow as tf
from loguru import logger
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import skimage.transform


class PredictRequest(BaseModel):
    path: str


class PredictResponse(BaseModel):
    yhat0: float
    yhat1: float
    yhat2: float


def load_image(image_file, dims=None):
    try:
        image = skimage.io.imread(image_file)
    except Exception as e0:
        logger.error(f"[skimage] could not open image {image_file}, giving up: {str(e0)}")
        raise e0

    if dims is not None:
        try:
            image = skimage.transform.resize(
                image,
                output_shape=(dims, dims, 3),
                mode="reflect",
                anti_aliasing=True,
            )
            image = np.expand_dims(image, axis=0)
        except Exception as e:
            logger.error(f"[skimage] could not transform image {image_file}: {str(e)}")
            raise e

    return image


def predict(image_file):
    image = load_image(image_file, dims=dimensions)

    # seems for some images skimage (pillow?) returns an array of the image, need to extract it
    if image.shape == (2,):
        image = image[0]

    image = preprocess_input(image, mode="tf")

    # seems to be caused by weird pillow/PIL/skimage versions?
    if image.shape == (1, dimensions, dimensions, 3, 3):
        image = image[:, :, :, 0]

    return model.predict(image)[0]


# TODO: change to configurable
fsk_model_file = sys.argv[1]
dimensions = sys.argv[2]
label_0 = sys.argv[3]
label_1 = sys.argv[4]
label_2 = sys.argv[5]
classes = {0: label_0, 1: label_1, 2: label_2}

app = FastAPI()
model = tf.keras.models.load_model(
    fsk_model_file,
    custom_objects=None,
    compile=True
)


@app.post("/v1/detect", response_model=PredictResponse)
def detect(request: PredictRequest):
    y_hats = predict(request.path)

    print(f"{request.path} \t ", end="")

    for i, y_hat in enumerate(y_hats):
        print(f"{classes[i]} \t {y_hat * 100:.2f} % \t", end="")

    print()

    return PredictResponse(
        path=request.path,
        yhat0=y_hats[0],
        yhat1=y_hats[1],
        yhat2=y_hats[2],
    )
