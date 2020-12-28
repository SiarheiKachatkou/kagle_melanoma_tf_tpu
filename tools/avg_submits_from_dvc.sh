
TAGS=( baseline_b6 best_B0_384_baseline B6_with_meta B0_384_with_meta)
for t in ${TAGS[@]}
do
  git checkout $t
  dvc pull
  cp artifacts/baseline/kaggle_test__tta_.csv ../subms/$t.csv
done