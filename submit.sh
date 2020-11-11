
set -x

gs_folders=( "2020-11-10 15:09:41._B6_bce_loss_384_penalty_1e-16")
dst_folder=artifacts

for folder in "${gs_folders[@]}"
do
  gsutil -m -q cp -r  "gs://kochetkov_kaggle_melanoma/$folder" artifacts
  python submit.py --work_dir="$dst_folder/$folder"

done
