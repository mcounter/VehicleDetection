import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from moviepy.editor import VideoFileClip

from FrameProcessor import FrameProcessor
from ImageClassifier import ImageClassifier
from ImageEngine import ImageEngine

import multiprocessing

def process_image(img):
    global num_frames_global
    
    num_frames_global += 1
    processor.processFrame(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    #return cv2.cvtColor(processor.visHeatMapImage, cv2.COLOR_BGR2RGB)
    #return cv2.cvtColor(processor.visVehicleBoxesImage, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(processor.visImageAnnotated, cv2.COLOR_BGR2RGB)

def process_video(file_name, sub_clip_from = 0, sub_clip_to = 0, visualization = True):
    global num_frames_global
    num_frames_global = 0

    imgEng = ImageEngine(load_setup = True)
    clfLinearSVC = ImageClassifier('clf_linear')
    clfSVC = ImageClassifier('clf_rbf')

    global processor
    processor = FrameProcessor(imgEng, clfLinearSVC, clfSVC, baseWindowSize, detectionRegions, visualization = visualization, heatMapFrames = 5, heatMapThreshold = 15, heatMapTotalMin = 1000, heatMapTotalMinFrame = 200)

    v_clip = VideoFileClip(input_dir_path + file_name)
    if sub_clip_to > 0:
        v_clip = v_clip.subclip(sub_clip_from, sub_clip_to)

    white_clip = v_clip.fl_image(process_image)
    white_clip.write_videofile(output_dir_path + file_name, audio=False)
    print("Video is processed. Frames: {0}.".format(num_frames_global))
    return

input_dir_path = "./test_videos/"
output_dir_path = "./test_videos_output/"

try:
    os.makedirs(output_dir_path)
except:
    pass

# Number of parallel threads
threads_num = 4
# Sliding window size
baseWindowSize = 64
# Set of detection regions and feature sizes used for vehicle detection
detectionRegions = [
    #[(360, 0), (445, 1280), (32,), (0.5, 0.5)],
    [(360, 0), (445, 1280), (48,), (2.0/3.0, 2.0/3.0)],
    [(360, 0), (490, 1280), (64,), (0.75, 0.75)],
    [(360, 0), (655, 1280), (96,), (5.0/6.0, 5.0/6.0)],
    [(360, 0), (655, 1280), (128,), (0.75, 0.75)],
    [(360, 0), (655, 1280), (192,), (5.0/6.0, 5.0/6.0)],
    ]

if __name__ == '__main__':
    args = ["test_video.mp4", "project_video.mp4", "challenge_video.mp4"]
    if threads_num > 1:
        pool = multiprocessing.Pool(processes = threads_num)
        pool.map(process_video, args)
        pool.close()
    else:
        for file_name in args:
            process_video(file_name)

        #process_video("test_video.mp4")

        #process_video("project_video.mp4", sub_clip_from = 27, sub_clip_to = 31)

        #process_video("challenge_video.mp4")

