
set -x

gs_folders=( "B4_focal_loss_512_old_datasets_penalty_1e-6_2020-11-07 06:32:45.283702" )
dst_folder=artifacts

for folder in "${gs_folders[@]}"
do
  #gsutil -m -q cp -r  "gs://kochetkov_kaggle_melanoma/$folder" artifacts
  python submit.py --work_dir="$dst_folder/$folder"
done
