from display_utils import display_9_images_from_dataset
from dataset_utils import get_train_val_filenames, get_training_dataset, return_2_values
from consts import CONFIG, IMAGE_HEIGHT, DATASETS

if __name__=="__main__":
    fold=0
    train_files_folds, val_files_folds = get_train_val_filenames(DATASETS[IMAGE_HEIGHT],CONFIG.nfolds)
    dataset = get_training_dataset(train_files_folds[fold])
    display_9_images_from_dataset(return_2_values(dataset))