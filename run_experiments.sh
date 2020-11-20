#!/usr/bin/env bash

for backbone in B0 B1 B2
do
  for cut_mix_prob in 0 0.1
   do
     for dropout_rate in 0 0.1 0.5
     do
       python main.py --backbone=$backbone --cut-mix-prob=$cut_mix_prob --dropout-rate=$dropout_rate
     done
   done
done