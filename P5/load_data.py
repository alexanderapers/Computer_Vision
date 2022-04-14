import os
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.image import load_img, save_img, image_dataset_from_directory

BATCH_SIZE = 32

def get_dataset_paths(img_path):
    img_path = r"Stanford40/PreprocessedImages/"
    train_path = join(img_path, r"Train")
    val_path = join(img_path, r"Validation")
    test_path = join(img_path, r"Test")
    return train_path, val_path, test_path

def load_stanford():

    # Gather train/validation/test splits
    (train_x, train_y), (val_x, val_y), (test_x, test_y), classes = __get_stanford40_splits()
    print(train_x[0], train_y[0])
    # Preprocess dataset (if it hasn't been already)
    __preprocess_stanford40(train_x, val_x, test_x)

    preprocessed_img_path = r"Stanford40/PreprocessedImages/"
    train_path, val_path, test_path = get_dataset_paths(preprocessed_img_path)

    train_set = image_dataset_from_directory(train_path, labels=train_y, batch_size=BATCH_SIZE, shuffle=False, image_size=(224,224))
    val_set = image_dataset_from_directory(val_path, labels=val_y, batch_size=BATCH_SIZE, shuffle=False, image_size=(224,224))
    test_set = image_dataset_from_directory(test_path, labels=test_y, batch_size=BATCH_SIZE, shuffle=False, image_size=(224,224))

    return train_set, val_set, test_set



def load_tvhi(train_file_names, validation_file_names, test_file_names):
    folder_path = r"TV-HI/tv_human_interactions_videos"

    for file_name in train_file_names:

        cap, frames = get_videocap(os.path.join(folder_path, file_name))

        # TODO get frames using get_frame
        # frames gives the number of frames in a video
        cap.release()
        break

def count_frames_manual(video):
    i = 0
    while True:
        video.set(0, i)
        ret, frame = video.read()

        if not ret:
            break
        i += 1
    return i


def get_videocap(file_path_name):
    cap = cv2.VideoCapture(file_path_name)

    if not cap.isOpened():
        print("could not open :", file_path_name)
        return

    frames = count_frames_manual(cap)

    return (cap, frames)

def get_frame(video, frame_number):
    video.set(0, frame_number)
    ret, frame = video.read()

    if not ret:
        print("frame could not be found")

    return frame

def __get_stanford40_splits():
    with open('Stanford40/ImageSplits/actions.txt') as f:
        action_categories = f.readlines()[1:]
        action_categories = [action.split()[0] for action in action_categories]
        action_dict = { action: idx for idx, action in enumerate(action_categories) }

    with open('Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = [action_dict['_'.join(name.split('_')[:-1])] for name in train_files]

        train_files, validation_files, train_labels, validation_labels = train_test_split(
            train_files, train_labels, train_size = 0.9, random_state = 42,
            shuffle = True, stratify = train_labels)

        train = (train_files, train_labels)
        validation = (validation_files, validation_labels)

    with open('Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = [action_dict['_'.join(name.split('_')[:-1])] for name in test_files]

        test = (test_files, test_labels)

    return (train, validation, test, action_categories)


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
        save_img(join(split_path, filename), img)

def __process_TVHI():
    set_1_indices = [[2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
                 [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
                 [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
                 [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42]]
    set_2_indices = [[1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
                 [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
                 [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
                 [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50]]

    classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

    # test set
    set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
    #print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
    #print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')

    test = (set_1, set_1_label)

    # training set
    set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
    #print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
    #print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')

    # Use 10% of training data for validation, use stratification
    train_files, validation_files, train_labels, validation_labels = train_test_split(
        set_2, set_2_label, train_size = 0.9, random_state = 42, shuffle = True,
        stratify = set_2_label)

    train = (train_files, train_labels)
    validation = (validation_files, validation_labels)

    return (train, validation, test, classes)