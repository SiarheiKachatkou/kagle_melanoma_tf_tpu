from dataset.display_utils import display_9_images_from_dataset
from dataset.dataset_utils import get_train_val_filenames, get_training_dataset
from config.config import CONFIG
from config.consts import DATASETS

if __name__=="__main__":
    fold=0
    train_files_folds, val_files_folds, *_ = get_train_val_filenames(DATASETS[CONFIG.image_height]['new'],CONFIG.nfolds)

    dataset = get_training_dataset([train_files_folds[fold][0]], str(DATASETS[CONFIG.image_height]['old']),CONFIG)
    display_9_images_from_dataset(dataset,show_zero_labels=False)