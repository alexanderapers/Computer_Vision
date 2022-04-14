import os
import cv2
import tensorflow as tf
import numpy as np
from tensorflow import py_function
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

BATCH_SIZE = 32
NUM_CLASSES = 40
IMG_PATH = r"Stanford40/JPEGImages/"



def load_stanford():
    (train_x, train_y), (val_x, val_y), (test_x, test_y), classes = __process_stanford40()

    # One-hot encode labels.
    num_classes = len(classes)
    train_y = to_categorical(train_y, num_classes=num_classes)
    val_y = to_categorical(val_y, num_classes=num_classes)
    test_y = to_categorical(test_y, num_classes=num_classes)

    # Load all sets.
    train_dataset = __load_stanford_set(train_x, train_y)
    val_dataset = __load_stanford_set(val_x, val_y)
    test_dataset = __load_stanford_set(test_x, test_y)
    return train_dataset, val_dataset, test_dataset, classes


def __load_stanford_set(filenames, labels):

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    # Preprocessing function that prepares images for use in the dataset.
    def preprocess_image(filename, label):
        # Decode the filename and join it with the file path
        image_string = tf.io.read_file(os.path.join(IMG_PATH, bytes.decode(filename.numpy())))

        # Decode jpg, normalize to a float32 between 0 & 1, and then resize the image.
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        print(image)
        resized_image = tf.image.resize(image, [224, 224])

        return resized_image, label

    def set_preprocessed_shape(x, y):
        x_out, y_out = py_function(preprocess_image, [x,y], [tf.float32, tf.float32],)
        x_out.set_shape((224, 224, 3))
        y_out.set_shape(y.get_shape())
        return x_out, y_out
    
    dataset = dataset.map(lambda x, y: set_preprocessed_shape(x, y), num_parallel_calls=4)
    dataset = dataset.shuffle(len(filenames))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset

def load_tvhi(train_file_names, validation_file_names, test_file_names):
    train_file_names, validation_file_names, test_file_names, TVHI_classes = __process_TVHI()

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

def __process_stanford40():
    with open('Stanford40/ImageSplits/actions.txt') as f:
        action_categories = f.readlines()[1:]
        action_categories = [action.split()[0] for action in action_categories]
        action_dict = { action: idx for idx, action in enumerate(action_categories) }

    with open('Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = [action_dict['_'.join(name.split('_')[:-1])] for name in train_files]
        #print(f'Train files ({len(train_files)}):\n\t{train_files}')
        #print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')

        train_files, validation_files, train_labels, validation_labels = train_test_split(
            train_files, train_labels, train_size = 0.9, random_state = 42,
            shuffle = True, stratify = train_labels)

        train = (train_files, train_labels)
        validation = (validation_files, validation_labels)

    with open('Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = [action_dict['_'.join(name.split('_')[:-1])] for name in test_files]
        #print(f'Test files ({len(test_files)}):\n\t{test_files}')
        #print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')

        test = (test_files, test_labels)

    #print(f'Action categories ({len(action_categories)}):\n{action_categories}')

    return (train, validation, test, action_categories)


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
