import os
from os.path import join
import tensorflow as tf
from keras.preprocessing.image import load_img, save_img, image_dataset_from_directory
from files import get_stanford40_splits, get_tvhi_splits
from video_cap import write_video, load_video
from tqdm import tqdm
from tools import natural_keys
import numpy as np
from cv2 import readOpticalFlow

BATCH_SIZE = 8


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

    class_dict = { label: idx for idx, label in enumerate(classes) }

    # Preprocess dataset (if it hasn't been already)
    __preprocess_tvhi(train_x, train_y, val_x, val_y, test_x, test_y)

    (train_frames, train_flows, train_labels), (val_frames, val_flows, val_labels), (test_frames, test_flows, test_labels) =\
        __load_frames_and_flows(train_x, train_y, val_x, val_y, test_x, test_y, class_dict)

    train_frames_set = __make_dataset(train_frames, train_labels)
    val_frames_set = __make_dataset(val_frames, val_labels)
    test_frames_set = __make_dataset(test_frames, test_labels)

    train_flows_set = __make_dataset(train_flows, train_labels)
    val_flows_set = __make_dataset(val_flows, val_labels)
    test_flows_set = __make_dataset(test_flows, test_labels)

    return (train_frames_set, train_flows_set), (val_frames_set, val_flows_set), (test_frames_set, test_flows_set)


def __make_dataset(X, y):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(batch_size=BATCH_SIZE)
    return dataset


def __load_frames_and_flows(train_x, train_y, val_x, val_y, test_x, test_y, class_dict):

    preprocessed_frames_path = r"TV-HI/PreprocessedFrames/"
    train_path, val_path, test_path = get_dataset_paths(preprocessed_frames_path)

    train_frames, train_flows, train_labels = __load_frames_and_flows_set(train_path, train_x, train_y, class_dict)
    val_frames, val_flows, val_labels = __load_frames_and_flows_set(val_path, val_x, val_y, class_dict)
    test_frames, test_flows, test_labels = __load_frames_and_flows_set(test_path, test_x, test_y, class_dict)

    return (train_frames, train_flows, train_labels),\
           (val_frames, val_flows, val_labels), \
           (test_frames, test_flows, test_labels)


def __load_frames_and_flows_set(set_path, set_x, set_y, class_dict):
    set_labels = [class_dict[label] for label in set_y]
    set_frames = []
    set_flows = []

    for file_name in set_x:
        file_dir = join(set_path, file_name.split('.')[0])
        frames_dir = join(file_dir, r"frames")
        flows_dir = join(file_dir, r"flows")
        flows = []

        for _, _, files in os.walk(frames_dir):
            middle_frame = np.array(load_img(join(frames_dir, files[0])))
            set_frames.append(middle_frame)
            break

        for _, _, files in os.walk(flows_dir):
            files.sort(key=natural_keys)
            flows = []
            for file in files:
                flow = readOpticalFlow(join(flows_dir, file))
                flows.append(flow)
            flows = np.array(flows)
        set_flows.append(flows)

    set_frames = np.array(set_frames)
    set_flows = np.array(set_flows)

    return set_frames, set_flows, set_labels


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


def __preprocess_tvhi(train_x, train_y, val_x, val_y, test_x, test_y):
    vid_path = r"TV-HI/tv_human_interactions_videos"
    preprocessed_vids_path = r"TV-HI/PreprocessedFrames/"
    train_path, val_path, test_path = get_dataset_paths(preprocessed_vids_path)

    # If the preprocessed directory exists, do nothing. Otherwise, create it and start preprocessing images.
    preprocessed_dir = os.path.isdir(preprocessed_vids_path)
    if not preprocessed_dir:
        os.mkdir(preprocessed_vids_path)
        __preprocess_tvhi_split(vid_path, train_path, train_x, train_y)
        __preprocess_tvhi_split(vid_path, val_path, val_x, val_y)
        __preprocess_tvhi_split(vid_path, test_path, test_x, test_y)


def __preprocess_tvhi_split(vid_path, split_path, split_filenames, labels):
    # Create subdirectory for split.
    os.mkdir(split_path)

    # For each image in the split filenames, store a resized image in the subdirectory
    for filename, label in tqdm(zip(split_filenames, labels)):
        video_file_dir_name = join(split_path, filename.split('.')[0])
        os.mkdir(video_file_dir_name)
        os.mkdir(join(video_file_dir_name, r"frames"))
        os.mkdir(join(video_file_dir_name, r"flows"))

        frames, flows = load_video(join(vid_path, filename))
        video_file_frames_path = join(video_file_dir_name, r"frames")
        video_file_flows_path = join(video_file_dir_name, r"flows")

        write_video(video_file_frames_path, video_file_flows_path, filename, frames, flows, 16)



