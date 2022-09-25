#!/bin/bash

if [ $# != 4 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "Usage: bash run_distribute_train.sh [DEVICE_NUM] [DISTRIBUTE] [RANK_TABLE_FILE] [DATA_DIR]"
  echo "bash run_distribute_train.sh 8 1 ./hccl_8p.json /mass_data/dataset/LS-2mix/Libri2Mix/tr"
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export RANK_TABLE_FILE=$3
export RANK_START_ID=0
export RANK_SIZE=$1
echo "lets begin!!!!XD"

for((i=0;i<$1;i++))
do
        export DEVICE_ID=$((i + RANK_START_ID))
        export RANK_ID=$i
        echo "start training for rank $i, device $DEVICE_ID"
        env > env.log

        rm -rf ./train_parallel$i
        mkdir ./train_parallel$i
        mkdir ./train_parallel$i/src
        cp -r ../*.py ./train_parallel$i
        cp -r ../src/*.py ./train_parallel$i/src
        cd ./train_parallel$i || exit
        python train.py --device_num=$1 --run_distribute=$2 --device_id=$DEVICE_ID --train_dir=$4 > paralletrain.log 2>&1 &
        cd ..
done