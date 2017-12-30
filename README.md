## Vehicle Detection and Tracking

Identify vehicles in a video from a front-facing camera on a car and highlight and track it while it visible.

Please find [Advanced Vehicle Detection and Tracking version with using Dark Flow (YOLO v2) here](https://github.com/mcounter/AdvancedVehicleDetection).

![Vehicle Detection and Tracking](./images/main.jpg)

---

Practical project to recognize vehicles in video stream, label it and track throw whole video while it visible on frames. The main idea of project use classic mathematic approach based on HOG feature vectors and SVM classifier. Project was done on Python. More details are [here](./writeup.md).

**Project content**
*	[CreateClassifier.py](./CreateClassifier.py) – Python class, used to create and train classifiers.
*	[DeepDataEngine.py](./DeepDataEngine.py) – Python class, read set of images from disk, extract features and prepare dataset for classifier training.
*	[FrameProcessor.py](./FrameProcessor.py) – Python class, implements frame processing pipeline.
*	[HOGDemo.py](./HOGDemo.py) – visualize HOG features, used for demonstration purpose only.
*	[ImageClassifier.py](./ImageClassifier.py) – Python class, wrapper for different classifiers from `sklearn` module.
*	[ImageEngine.py](./ImageEngine.py) – Python class, perform all operations with image - color space transformation, features extraction.
*	[ProcessImages.py](./LaneLine.py) – run image pipeline.
*	[ProcessVideos.py](./ProcessVideos.py) – run video pipeline.
*	[writeup.md](./writeup.md) – project description in details
*	[README.md](./README.md) – this file.
* [images](./images) - folder with different images used in project writeup.
*	[test_videos](./test_videos) - folder with test video files: [project_video.mp4](./test_videos/project_video.mp4) - main project video, [challenge_video.mp4](./test_videos/challenge_video.mp4) - challenge video, [test_video.mp4](./test_videos/test_video.mp4) - short test video
*	[test_videos_output](./test_videos_output) - folder with annotated video files: [project_video.mp4](./test_videos_output/project_video.mp4) - main project video, [challenge_video.mp4](./test_videos_output/challenge_video.mp4) - challenge video, [test_video.mp4](./test_videos_output/test_video.mp4) - short test video
*	[test_videos_output/heat_map](./test_videos_output/heat_map) - folder with video files annotated with extended heat map: [project_video.mp4](./test_videos_output/heat_map/project_video.mp4) - main project video, [challenge_video.mp4](./test_videos_output/heat_map/challenge_video.mp4) - challenge video, [test_video.mp4](./test_videos_output/heat_map/test_video.mp4) - short test video
*	[config](./config) - folder with trained classifiers. **Note: Due to GitHub limitation, content of this folder was put in ZIP archive and split on 3 files. Extract files directly in this folder if you plan run pipeline.**
