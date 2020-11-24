#!/usr/bin/env bash

for backbone in B0 B1
do
  for cut_mix_prob in 0.1 0
   do
     for dropout_rate in 0  0.1 0.5
     do
       python train_and_test.py --backbone=$backbone --cut-mix-prob=$cut_mix_prob --dropout-rate=$dropout_rate
     done
   done
done