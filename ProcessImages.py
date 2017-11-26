import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from FrameProcessor import FrameProcessor
from ImageClassifier import ImageClassifier
from ImageEngine import ImageEngine

import multiprocessing
import time

def processFrame(src_name, visualization = True, show_diagram = False):
    img = cv2.imread(input_dir_path + src_name)
    
    imgEng = ImageEngine(load_setup = True)
    clfLinearSVC = ImageClassifier('clf_linear')
    clfSVC = ImageClassifier('clf_rbf')

    processor = FrameProcessor(imgEng, clfLinearSVC, clfSVC, baseWindowSize, detectionRegions, visualization = visualization, heatMapFrames = 1, heatMapThreshold = 2, heatMapTotalMin = 200, heatMapTotalMinFrame = 200)
    processor.processFrame(img)

    if visualization:
        cv2.imwrite(all_boxes_dir_path+src_name, processor.visAllBoxesImage)
        cv2.imwrite(vehicle_boxes_dir_path+src_name, processor.visVehicleBoxesImage)
        cv2.imwrite(heat_map_dir_path+src_name, processor.visHeatMapImage)

    cv2.imwrite(annotated_dir_path+src_name, processor.visImageAnnotated)

    if show_diagram:
        # Visualize img binary
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize = 30)
        ax2.imshow(cv2.cvtColor(processor.visVehicleBoxesImage, cv2.COLOR_BGR2RGB))
        ax2.set_title('Annotated Image', fontsize = 30)
        plt.show()

def processImage(image_name):
    image_path = input_dir_path + image_name
    if os.path.exists(image_path) and os.path.isfile(image_path):
        t = time.time()
        print("Processing image {0} ...".format(image_name))
        processFrame(image_name)
        print("    Processed in {:.2f} sec.".format(time.time() - t))

    print()

input_dir_path = "./test_images/"
output_dir_path = "./test_images_output/"
all_boxes_dir_path = "./test_images_output/all_boxes_images/"
vehicle_boxes_dir_path = "./test_images_output/vehicle_boxes_images/"
heat_map_dir_path = "./test_images_output/heat_map_images/"
annotated_dir_path = "./test_images_output/annotated_images/"

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

images_dir = None
try: images_dir = os.listdir(input_dir_path)
except: print("Cannot read list of images")

try: os.makedirs(output_dir_path)
except: pass
try: os.makedirs(all_boxes_dir_path)
except: pass
try: os.makedirs(vehicle_boxes_dir_path)
except: pass
try: os.makedirs(heat_map_dir_path)
except: pass
try: os.makedirs(annotated_dir_path)
except: pass

if __name__ == '__main__':
    if images_dir is not None:
        if threads_num > 1:
            pool = multiprocessing.Pool(processes = threads_num)
            pool.map(processImage, images_dir)
            pool.close()
        else:
            for image_name in images_dir:
                processImage(image_name)

