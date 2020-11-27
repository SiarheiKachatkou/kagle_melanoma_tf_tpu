#!/usr/bin/env bash

for backbone in B0 B1
do
    for dropout_rate in  0 0.05
     do
       for lr_max in 1 10
       do
         for lr_exp_decay in 0.8 0.5
          do
            for oversample_mult in 1 2 3
            do
              python train_and_test.py --backbone=$backbone --oversample_mult=$oversample_mult --dropout-rate=$dropout_rate --lr_max=$lr_max --lr_exp_decay=$lr_exp_decay
            done
          done
        done
     done
done