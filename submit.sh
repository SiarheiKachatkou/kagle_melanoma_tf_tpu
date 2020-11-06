
set -x

gs_folders=( "b4_focal_loss_768_old_datasets_penalty_1e-6_2020-11-06 04:42:40.388071" "b4_focal_loss_768_old_datasets_penalty_1e-9_2020-11-06 04:27:19.296397" )
dst_folder=artifacts

for folder in "${gs_folders[@]}"
do
  gsutil -m -q cp -r  "gs://kochetkov_kaggle_melanoma/$folder" artifacts
  python submit.py --work_dir="$dst_folder/$folder"
done
