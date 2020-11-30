#!/usr/bin/env bash

dropout_rate=0

for backbone in B0 B1
do
    for lr_warm_up_epochs in 5 7 10
     do
       for lr_max in 10
       do
         for lr_exp_decay in 0.5
          do
            for focal_loss_gamma in 3
            do
              for focal_loss_alpha in 0.5
              do
                for hair_prob in 0.03
                do
                  for micro_prob in 0 0.01
                  do
                    python train_and_test.py --backbone=$backbone --dropout-rate=$dropout_rate --lr_max=$lr_max --lr_exp_decay=$lr_exp_decay --focal_loss_gamma=$focal_loss_gamma --focal_loss_alpha=$focal_loss_alpha --hair-prob=$hair_prob --microscope-prob=$micro_prob --lr-warm-up-epochs=$lr_warm_up_epochs
                  done
                done
              done
            done
          done
        done
     done
done