import os
import cv2
import numpy as np

def load_stanford(train_file_names, validation_file_names, test_file_names):
    folder_path = r"Stanford40/JPEGImages/"

    # shape of img is width x height x 3 (color channels)
    train_files = [cv2.imread(os.path.join(folder_path, file_name)) for file_name in train_file_names]
    validation_files = [cv2.imread(os.path.join(folder_path, file_name)) for file_name in validation_file_names]
    test_files = [cv2.imread(os.path.join(folder_path, file_name)) for file_name in test_file_names]

    # do preprocessing here?
    return (train_files, validation_files, test_files)




def load_tvhi(train_file_names, validation_file_names, test_file_names):
    folder_path = r"TV-HI/tv_human_interactions_videos"

    for file_name in train_file_names:
        cap = cv2.VideoCapture(os.path.join(folder_path, file_name))

        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        waitKey(0)
        break
