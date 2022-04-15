import cv2
import numpy as np

def get_videocap(file_path_name):
    cap = cv2.VideoCapture(file_path_name)

    if not cap.isOpened():
        print("could not open :", file_path_name)
        return

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    return cap, int(frames)


def get_frame(video, frame_number):
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video.read()

    if not ret:
        print("frame could not be found")

    return frame


def write_video(file_path, frames, n_frames):

    n_frames_total = frames.shape[0]
    steps = n_frames_total // n_frames

    for i in range(n_frames):
        frame = frames[i*steps, :, :, :]
        file_name = "{}_frame_{}.jpg".format(file_path.split('.')[0], i)
        cv2.imwrite(file_name, frame)


def load_video(file_path):
    cap, n_frames = get_videocap(file_path)
    frames = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
        # TODO get optical flow

    cap.release()
    return np.array(frames)
