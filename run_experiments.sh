#!/usr/bin/env bash

for backbone in B0 B1
do
    for dropout_rate in  0 0.05
     do
       for lr_max in 10 100
       do
         for lr_exp_decay in 0.8 0.5
          do
            for focal_loss_gamma in 0.25 1 2
            do
              for focal_loss_alpha in 0.5 0.75 0.8
              do
                python train_and_test.py --backbone=$backbone --dropout-rate=$dropout_rate --lr_max=$lr_max --lr_exp_decay=$lr_exp_decay --focal_loss_gamma=$focal_loss_gamma --focal_loss_alpha=$focal_loss_alpha
              done
            done
          done
        done
     done
done