import cv2
import numpy as np
from os.path import join


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


def write_video(frames_dir, flows_dir, file_name, frames, flows, n_frames):

    n_frames_total = frames.shape[0]
    steps = n_frames_total // n_frames

    for i in range(n_frames):
        frame = frames[i*steps, :, :, :]
        name = "{}_frame_{}.jpg".format(file_name.split('.')[0], i*steps)
        cv2.imwrite(join(frames_dir, name), frame)

        name = "{}_flow_{}.flo".format(file_name.split('.')[0], i*steps)
        cv2.writeOpticalFlow(join(flows_dir, name), flows[i*steps, :, :, :])


def load_video(file_path):

    cap, n_frames = get_videocap(file_path)
    frames = []
    flows = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

        flow = None
        if len(frames) > 1:
            grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            prev_grayscale_frame = cv2.cvtColor(frames[-2], cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_grayscale_frame, grayscale_frame, flow, pyr_scale=0.5, levels=3,
                                            winsize=5, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            flows.append(flow)

    cap.release()
    return np.array(frames), np.array(flows)
