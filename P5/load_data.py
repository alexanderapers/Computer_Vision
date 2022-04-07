import os
import cv2
import numpy as np

def load_stanford(train_file_names, validation_file_names, test_file_names):
    folder_path = r"Stanford40/JPEGImages/"

    # do preprocessing here?

    # shape of img is width x height x 3 (color channels)
    train_files = [cv2.imread(os.path.join(folder_path, file_name)) for file_name in train_file_names]
    validation_files = [cv2.imread(os.path.join(folder_path, file_name)) for file_name in validation_file_names]
    test_files = [cv2.imread(os.path.join(folder_path, file_name)) for file_name in test_file_names]

    return (train_files, validation_files, test_files)




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
