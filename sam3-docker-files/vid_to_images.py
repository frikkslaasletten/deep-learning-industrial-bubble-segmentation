import os
import cv2
import numpy as np

def video_to_frames(video_folder:str, resize:list[int]=(256, 256), target_fps:int=10):
    """
    :param video_folder: String containing the folder where the videos are stored
    :param resize: List of two ints specifying the size of the frames to be resized
    :param target_fps: Integer specifying how many frames per second should be retained
    :return: numpy array with shape [height, width, 3] with dtype float32 normalized to [0, 1]
    """
    frames = []
    vid_indexes = []
    j = 0
    for video_path in os.listdir(video_folder):
        cap = cv2.VideoCapture(os.path.join(video_folder, video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        accept_rate = int(round(fps / target_fps))
        print(f"video: {video_path}, fps: {fps}, accept_rate: {accept_rate}")
        i = 0
        i_accepted = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % accept_rate != 0:
                i += 1
                continue
            i += 1
            i_accepted += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
        cap.release()
        j += i_accepted
        vid_indexes.append(j)
    print("Amount of frames gathered: ", len(frames))
    frames = np.array(frames, dtype=np.float32)
    frames = frames/255.0
    return (frames, vid_indexes)
