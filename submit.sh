CSV=b4_focal_loss_768_final/test_0_single_model_tta_submission.csv

git commit -a -m "auto commit in submit.sh"

HASH=$(git rev-parse HEAD)

kaggle competitions submit -c siim-isic-melanoma-classification -f $CSV -m ${CSV}