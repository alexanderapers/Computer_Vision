import os
from os.path import join
import cv2
import tensorflow as tf
from keras.preprocessing.image import load_img, save_img, image_dataset_from_directory
import files

BATCH_SIZE = 32

def get_dataset_paths(img_path):
    img_path = r"Stanford40/PreprocessedImages/"
    train_path = join(img_path, r"Train")
    val_path = join(img_path, r"Validation")
    test_path = join(img_path, r"Test")
    return train_path, val_path, test_path


def load_stanford():
    # Gather train/validation/test splits
    (train_x, train_y), (val_x, val_y), (test_x, test_y), classes = files.get_stanford40_splits()

    # Preprocess dataset (if it hasn't been already)
    __preprocess_stanford40(train_x, val_x, test_x)

    preprocessed_img_path = r"Stanford40/PreprocessedImages/"
    train_path, val_path, test_path = get_dataset_paths(preprocessed_img_path)

    # TODO: fix problem that this is loading images from directory in whatever order when y order is not the same, possible
    # TODO: solution is to use subdirectories per class after all and
    train_set = image_dataset_from_directory(train_path, labels=train_y, batch_size=BATCH_SIZE, shuffle=False, image_size=(224,224))
    val_set = image_dataset_from_directory(val_path, labels=val_y, batch_size=BATCH_SIZE, shuffle=False, image_size=(224,224))
    test_set = image_dataset_from_directory(test_path, labels=test_y, batch_size=BATCH_SIZE, shuffle=False, image_size=(224,224))

    return train_set, val_set, test_set


def load_tvhi(train_file_names, validation_file_names, test_file_names):
    folder_path = r"TV-HI/tv_human_interactions_videos"

    for file_name in train_file_names:

        cap, frames = __get_videocap(os.path.join(folder_path, file_name))

        # TODO get frames using get_frame
        # frames gives the number of frames in a video
        cap.release()
        break


def __count_frames_manual(video):
    i = 0
    while True:
        video.set(0, i)
        ret, frame = video.read()

        if not ret:
            break
        i += 1
    return i


def __get_videocap(file_path_name):
    cap = cv2.VideoCapture(file_path_name)

    if not cap.isOpened():
        print("could not open :", file_path_name)
        return

    frames = __count_frames_manual(cap)

    return cap, frames


def __get_frame(video, frame_number):
    video.set(0, frame_number)
    ret, frame = video.read()

    if not ret:
        print("frame could not be found")

    return frame


def __preprocess_stanford40(train_split, val_split, test_split):
    img_path = r"Stanford40/JPEGImages/"
    preprocessed_img_path = r"Stanford40/PreprocessedImages/"
    train_path, val_path, test_path = get_dataset_paths(preprocessed_img_path)

    # If the preprocessed directory exists, do nothing. Otherwise, create it and start preprocessing images.
    preprocessed_dir = os.path.isdir(preprocessed_img_path)
    if not preprocessed_dir:
        os.mkdir(preprocessed_img_path)
        __preprocess_stanford40_split(img_path, train_path, train_split)
        __preprocess_stanford40_split(img_path, val_path, val_split)
        __preprocess_stanford40_split(img_path, test_path, test_split)


def __preprocess_stanford40_split(img_path, split_path, split_filenames):

    # Create subdirectory for split.
    os.mkdir(split_path)
    os.mkdir(join(split_path, r"JPEGImages"))

    # For each image in the split filenames, store a resized image in the subdirectory
    for filename in split_filenames:
        img = load_img(join(img_path, filename))
        img = tf.image.resize(img, (224,224))
        save_img(join(join(split_path, r"JPEGImages"), filename), img)
