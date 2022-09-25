#!/bin/bash

if [ $# != 3 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "Usage: bash run_eval.sh [DEVICE_ID] [CKPT_PATH] [DATA_DIR]"
  echo "bash run_eval.sh 0 /home/heu_MEDAI/DPRNN-100_445.ckpt  /mass_data/dataset/LS-2mix/Libri2Mix/tt"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export DEVICE_ID=$1
export RANK_SIZE=1

rm -rf ./eval
mkdir ./eval
mkdir ./eval/src
cp -r ../*.py ./eval
cp -r ../src/*.py ./eval/src

env > env.log
python ./eval/evaluate.py  --device_id=$1 --model_path=$2 --data_dir=$3 > eval.log 2>&1 &
