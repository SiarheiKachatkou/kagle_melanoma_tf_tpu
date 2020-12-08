import argparse

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backbone',type=str)
    parser.add_argument('--dropout_rate',type=float)
    parser.add_argument('--lr_max',type=float)
    parser.add_argument('--lr_exp_decay',type=float)
    parser.add_argument('--hair_prob',type=float)
    parser.add_argument('--microscope_prob',type=float)
    parser.add_argument('--lr_warm_up_epochs',type=int)
    parser.add_argument('--gpus',type=str,default=None)
    parser.add_argument('--image_height',type=int)

    parser.add_argument('--focal_loss_gamma',type=float,default=4)
    parser.add_argument('--focal_loss_alpha',type=float,default=0.5)
    parser.add_argument('--oversample_mult',type=int,default=1)

    args=parser.parse_args()
    return args
