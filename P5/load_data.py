import os
from os.path import join
import tensorflow as tf
from keras.preprocessing.image import load_img, save_img, image_dataset_from_directory
from files import get_stanford40_splits, get_tvhi_splits
from video_cap import write_video, load_video

BATCH_SIZE = 32


def get_dataset_paths(files_path):

    train_path = join(files_path, r"Train")
    val_path = join(files_path, r"Validation")
    test_path = join(files_path, r"Test")

    return train_path, val_path, test_path


def load_stanford():
    # Gather train/validation/test splits
    (train_x, train_y), (val_x, val_y), (test_x, test_y), classes = get_stanford40_splits()

    # Preprocess dataset (if it hasn't been already)
    __preprocess_stanford40(train_x, train_y, val_x, val_y, test_x, test_y, classes)

    preprocessed_img_path = r"Stanford40/PreprocessedImages/"
    train_path, val_path, test_path = get_dataset_paths(preprocessed_img_path)

    train_set = image_dataset_from_directory(train_path, labels="inferred", class_names=classes, batch_size=BATCH_SIZE, seed=42, image_size=(224,224))
    val_set = image_dataset_from_directory(val_path, labels="inferred", class_names=classes, batch_size=BATCH_SIZE, seed=42, image_size=(224,224))
    test_set = image_dataset_from_directory(test_path, labels="inferred", class_names=classes, batch_size=BATCH_SIZE, seed=42, image_size=(224,224))

    class_dict = { label: idx for idx, label in enumerate(classes) }

    return train_set, val_set, test_set, class_dict


def load_tvhi():
    # Gather train/validation/test splits
    (train_x, train_y), (val_x, val_y), (test_x, test_y), classes = get_tvhi_splits()

    # Preprocess dataset (if it hasn't been already)
    __preprocess_tvhi(train_x, train_y, val_x, val_y, test_x, test_y, classes)

    preprocessed_img_path = r"TV-HI/PreprocessedFrames/"
    train_path, val_path, test_path = get_dataset_paths(preprocessed_img_path)

    # TODO make actual datasets from this

    return 0


def __preprocess_stanford40(train_x, train_y, val_x, val_y, test_x, test_y, classes):
    img_path = r"Stanford40/JPEGImages/"
    preprocessed_img_path = r"Stanford40/PreprocessedImages/"
    train_path, val_path, test_path = get_dataset_paths(preprocessed_img_path)

    # If the preprocessed directory exists, do nothing. Otherwise, create it and start preprocessing images.
    preprocessed_dir = os.path.isdir(preprocessed_img_path)
    if not preprocessed_dir:
        os.mkdir(preprocessed_img_path)
        __preprocess_stanford40_split(img_path, train_path, train_x, train_y, classes)
        __preprocess_stanford40_split(img_path, val_path, val_x, val_y, classes)
        __preprocess_stanford40_split(img_path, test_path, test_x, test_y, classes)


def __preprocess_stanford40_split(img_path, split_path, split_filenames, labels, classes):

    # Create subdirectory for split.
    os.mkdir(split_path)
    for classname in classes:
        os.mkdir(join(split_path, classname))

    # os.mkdir(join(split_path, r"JPEGImages"))

    # For each image in the split filenames, store a resized image in the subdirectory
    for (filename, label) in zip(split_filenames, labels):
        img = load_img(join(img_path, filename))
        img = tf.image.resize(img, (224, 224))
        save_img(join(join(split_path, label), filename), img)


def __preprocess_tvhi(train_x, train_y, val_x, val_y, test_x, test_y, classes):
    vid_path = r"TV-HI/tv_human_interactions_videos"
    preprocessed_vids_path = r"TV-HI/PreprocessedFrames/"
    train_path, val_path, test_path = get_dataset_paths(preprocessed_vids_path)

    # If the preprocessed directory exists, do nothing. Otherwise, create it and start preprocessing images.
    preprocessed_dir = os.path.isdir(preprocessed_vids_path)
    if not preprocessed_dir:
        os.mkdir(preprocessed_vids_path)
        __preprocess_tvhi_split(vid_path, train_path, train_x, train_y, classes)
        __preprocess_tvhi_split(vid_path, val_path, val_x, val_y, classes)
        __preprocess_tvhi_split(vid_path, test_path, test_x, test_y, classes)


def __preprocess_tvhi_split(vid_path, split_path, split_filenames, labels, classes):

    # Create subdirectory for split.
    os.mkdir(split_path)
    for classname in classes:
        os.mkdir(join(split_path, classname))

    # For each image in the split filenames, store a resized image in the subdirectory
    for filename, label in zip(split_filenames, labels):
        os.mkdir(join(join(split_path, label), filename.split('.')[0]))
        vid = load_video(join(vid_path, filename))
        write_video(join(join(join(split_path, label), filename.split('.')[0]), filename), vid, 16)



