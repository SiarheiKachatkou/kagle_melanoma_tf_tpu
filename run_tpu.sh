
TPU=${1:-3}

export BUCKET_NAME=kochetkov_belwest
export PROJECT_ID=turing-audio-146210
gcloud config set project $PROJECT_ID

ctpu up --project=${PROJECT_ID} \
 --zone=us-central1-b \
 --tf-version=2.3.1 \
 --tpu-size=v${TPU}-8 \
 --name=tpu-melanoma-${TPU} \
 --machine-type=n1-standard-8
 #--preemptible

exit 0

git clone -b triple https://github.com/SiarheiKachatkou/kagle_melanoma_tf_tpu.git &&
cd kagle_melanoma_tf_tpu &&
./install.sh &&
sudo python3 main.py --backbone=B6 --dropout_rate=0 --lr_max=3 --lr_exp_decay=0.8 --focal_loss_gamma=2 --focal_loss_alpha=0.5 --hair_prob=0 --microscope_prob=0 --lr_warm_up_epochs=5 --image_height=384

exit 0

export PROJECT_ID=turing-audio-146210
gcloud config set project $PROJECT_ID
ctpu delete --project=${PROJECT_ID}   --zone=us-central1-b   --name=tpu-melanoma-3