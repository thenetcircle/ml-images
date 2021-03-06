#!/bin/bash

if [ $# -lt 2 ]; then
  echo 'usage: ./train.sh train_dir output_dir'
  exit 1
fi

TF_CONFIG_FILE=tf_config.json TF_FORCE_GPU_ALLOW_GROWTH=true time python train.py $1 $2 2>training_errors.log
