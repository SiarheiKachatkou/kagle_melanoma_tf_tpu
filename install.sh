
pip uninstall tensorflow -y
pip uninstall tensorflow-gpu -y
pip install tensorflow-gpu==2.2.0
pip install --upgrade tensorflow_datasets
pip install -q -U albumentations
# use this if you want to fine-tune EfficientNet
pip install -U efficientnet
pip install tqdm