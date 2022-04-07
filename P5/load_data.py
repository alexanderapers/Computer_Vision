import os
import cv2
from cv2 import waitKey
import numpy as np
from tqdm import tqdm

def load_stanford(train_file_names, validation_file_names, test_file_names):

    # do preprocessing here?

    # shape of img is width x height x 3 (color channels)
    train_files = __get_stanford_files(train_file_names, "train")
    validation_files = __get_stanford_files(validation_file_names, "validation")
    test_files = __get_stanford_files(test_file_names, "test")

    return (train_files, validation_files, test_files)

def __get_stanford_files(file_names, set_name, overwrite=False):
    # Prepare file paths
    raw_img_path = r"Stanford40/JPEGImages/"
    preprocessed_path = r"Stanford40/PreprocessedImages/"
    preprocessed_file_path = os.path.join(preprocessed_path, f"{set_name}.npy")

    # If preprocessed files exist, load them now and return them.
    # Otherwise, preprocess, save, and then return them.
    preprocessed_exists = os.path.isfile(preprocessed_file_path)
    if (preprocessed_exists and overwrite == False):
        print(f"Loading preprocessed set {set_name}...")
        preprocessed_files = np.load(preprocessed_file_path)
    else:
        preprocessed_files = __preprocess_stanford_files(raw_img_path, file_names, set_name)
        np.save(preprocessed_file_path, preprocessed_files)

    return preprocessed_files

def __preprocess_stanford_files(folder_path, file_names, set_name): 
    # files = [cv2.imread(os.path.join(folder_path, file_name)) for file_name in file_names]
    # return files

    num_files = len(file_names)
    files = np.zeros(shape=(num_files, 224, 224, 3), dtype=float)

    for idx, file_name in tqdm(enumerate(file_names), desc=f"Preprocessing {set_name}", total=len(file_names)):
        
        # Get the image
        file_path = os.path.join(folder_path, file_name)
        image = cv2.imread(file_path)

        # Resize and normalize image, and store it in the numpy array.
        resized_image = cv2.resize(image, dsize=(224, 224)) / 255.0
        files[idx] = resized_image

    return files

def load_tvhi(train_file_names, validation_file_names, test_file_names):
    folder_path = r"TV-HI/tv_human_interactions_videos"

    for file_name in train_file_names:

        cap, frames = get_videocap(os.path.join(folder_path, file_name))
        print(frames)

        #frame = get_frame(cap, 500)
        #cv2.imshow("frame", frame)
        #cv2.waitKey(0)



        cap.release()
        break

def manual_frame_count(video):
    # frames = 0
    # while True:
    #     status, _ = video.read()
    #     if not status:
    #         print("thats weird")
    #     frames += 1
    #
    # return frames

    return video.get(cv2.CV_CAP_PROP_FRAME_COUNT)

    #return video.


def get_videocap(file_path_name):
    cap = cv2.VideoCapture(file_path_name)

    if not cap.isOpened():
        print("could not open :", file_path_name)
        return

    frames = manual_frame_count(cap)

    return (cap, frames)

def get_frame(video, frame_number):
    video.set(0, frame_number)
    ret, frame = video.read()

    if not ret:
        print("frame could not be found")

    return frame
