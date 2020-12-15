#!/bin/bash

name=baseline

PY=/home/sergey/anaconda3/envs/tensorflow_2/bin/python3

exclude_from_deps=( artifacts metrics .dvc .git )

artifacts_dir=$name

deps=""
for d in *
do
  if [[ -d $d ]]
  then
    is_excluded=false
    for e in "${exclude_from_deps[@]}"
    do
    if [[ $d == $e ]]
    then
      is_excluded=true
      break
    fi
    done
  if [[ $is_excluded == false ]]
  then
    deps="-d "$d" "$deps
  fi

  fi
done

echo deps=$deps
outputs_dir=artifacts/$artifacts_dir
metric=$PWD/metrics/metrics.txt

cmd="export LD_LIBRARY_PATH=/usr/local/cuda/lib64 && export PYTHONPATH=$PWD && $PY tools/main.py --work_dir=$outputs_dir --backbone=B0 --dropout_rate=0.01 --lr_max=10 --lr_exp_decay=0.5 --focal_loss_gamma=4 --focal_loss_alpha=0.8 --hair_prob=0.1 --microscope_prob=0.01 --lr_warm_up_epochs=5 --image_height=128"
dvc run -n $name $deps -o $outputs_dir -M $metric $cmd