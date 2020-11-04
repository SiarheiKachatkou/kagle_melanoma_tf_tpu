
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
