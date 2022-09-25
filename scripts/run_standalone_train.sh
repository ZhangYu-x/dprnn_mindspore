#!/bin/bash

if [ $# != 2 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "bash run_standalone_train.sh [DEVICE_ID] [DATA_DIR]"
  echo "bash run_standalone_train.sh 0 /mass_data/dataset/LS-2mix/Libri2Mix/tr"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export DEVICE_ID=$1
export RANK_ID=0
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0


rm -rf ./train_dprnn
mkdir ./train_dprnn
mkdir ./train_dprnn/src
cp -r ../*.py ./train_dprnn
cp -r ../src/*.py ./train_dprnn/src
cd ./train_dprnn || exit
python train.py --device_id=$DEVICE_ID --train_dir=$2 > train.log 2>&1 &
