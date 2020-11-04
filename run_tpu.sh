
TPU=${1:-3}

export BUCKET_NAME=kochetkov_belwest
export PROJECT_ID=turing-audio-146210
gcloud config set project $PROJECT_ID

ctpu up --project=${PROJECT_ID} \
 --zone=us-central1-b \
 --tf-version=2.3.1 \
 --tpu-size=v${TPU}-8 \
 --name=tpu-melanoma-${TPU} \
 --preemptible

exit 0

git clone -b old_dataset_transfer_learning https://github.com/SiarheiKachatkou/kagle_melanoma_tf_tpu.git
cd kagle_melanoma_tf_tpu
./install.sh
python3 main.py
