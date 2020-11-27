#!/usr/bin/env bash

for backbone in B0 B1
do
  for cut_mix_prob in 0 0.1
   do
     for dropout_rate in  0  0.1
     do
       for lr_max in 1 10 100
       do
         for lr_exp_decay in 0.8 0.1
          do
          python train_and_test.py --backbone=$backbone --cut-mix-prob=$cut_mix_prob --dropout-rate=$dropout_rate --lr_max=$lr_max --lr_exp_decay=$lr_exp_decay
          done
        done
     done
   done
done