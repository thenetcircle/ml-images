# Multi-Class Classification

TODO: copy-paste with redactions/generalizations.

TODO: include rationale/reasoning/compromises/etc.

TODO: Potentially include historical model/approach changes, and the reasons for it.

Choose which model version of the Student-Teacher pre-trained weights to use (0-7):

```shell
export ENET_MODEL_VERSION=5
cd data/
```

Download the weights:

```shell
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/noisy_student_efficientnet-b{ENET_MODEL_VERSION}.tar.gz
tar -xf noisy_student_efficientnet-b{ENET_MODEL_VERSION}.tar.gz
```

Convert the weights to `h5` format (script from [official tensorflow repo](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/efficientnet_weight_update_util.py)):

```shell
python efficientnet_weight_update_util.py \
  --model b{ENET_MODEL_VERSION} \
  --notop \
  --ckpt efficientnet-b{ENET_MODEL_VERSION}/model.ckpt \
  --o noisy-student-efficientnet-b{ENET_MODEL_VERSION}-notop.h5
```