
set -x

gs_folders=( "B0_bce_loss_128_penalty_1e-16_cycle_lr_e3")

dst_folder=artifacts

for folder in "${gs_folders[@]}"
do
  #gsutil -m -q cp -r  "gs://kochetkov_kaggle_melanoma/$folder" artifacts
  #python submit.py --work_dir="$dst_folder/$folder"
  python submit.py --work_dir="$folder"

done
