
set -x

gs_folders=( "2020-11-08 07:00:02._B4_focal_loss_128_old_datasets_penalty_1e-6")
dst_folder=artifacts

for folder in "${gs_folders[@]}"
do
  gsutil -m -q cp -r  "gs://kochetkov_kaggle_melanoma/$folder" artifacts
  python submit.py --work_dir="$dst_folder/$folder"
done
