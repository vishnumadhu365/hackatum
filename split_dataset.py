import os
import shutil
from random import shuffle

def split_dataset(images_path, train_path, test_path):
    '''!
    read all records in a given path
    '''
    (_, folders, files) = next(os.walk(images_path))
    if "not_categorized" in folders:
        folders.remove("not_categorized")
    if "difficult_not_labeled" in folders:
        folders.remove("difficult_not_labeled")
    if len(folders) != 0:
        for subfolder in folders:
            sub_path = os.path.join(images_path, subfolder)
            sub_train = os.path.join(train_path, subfolder)
            os.mkdir(sub_train)
            sub_test = os.path.join(test_path, subfolder)
            os.mkdir(sub_test)
            split_dataset(sub_path, sub_train, sub_test)
    else:
        shuffle(files)
        data_cnt = len(files)
        idx_train = int(0.8 * data_cnt)

        LABELS_FILE = "metadata.txt"
        if LABELS_FILE in files:
            files.remove(LABELS_FILE)
            full_src = os.path.join(images_path, LABELS_FILE)
            full_dst1 = os.path.join(train_path, LABELS_FILE)
            shutil.copyfile(full_src, full_dst1)
            full_dst2 = os.path.join(test_path, LABELS_FILE)
            shutil.copyfile(full_src, full_dst2)

        file_train = files[0:idx_train]
        file_test = files[idx_train:]

        for file in file_train:
            full_src = os.path.join(images_path, file)
            full_dst = os.path.join(train_path, file)
            shutil.copyfile(full_src, full_dst)

        for file in file_test:
            full_src = os.path.join(images_path, file)
            full_dst = os.path.join(test_path, file)
            shutil.copyfile(full_src, full_dst)




split_dataset("train", "train-split", "test-split")
