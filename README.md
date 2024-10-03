**Nova Security**

Nova Security is a Python-based real-time face detection system that uses OpenCV and a deep neural network (DNN) model to detect faces from a live webcam feed. When a face is detected, it plays an alert sound and saves the detected faces as images to a specified folder.

**Features**
- Real-time face detection using OpenCV DNN module.
- Plays an alert sound when a face is detected.
- Saves full-frame images with detected face boxes.
- Crops and saves individual detected faces.
- Supports running on CPU (with optional support for GPU).
- Displays the webcam feed with face detection annotations.

**Requirements**
- Python 3.x
- OpenCV (opencv-python and opencv-python-headless)
- playsound library for playing the alert sound.

**Install Dependencies**
- Install the required libraries using pip:
pip install opencv-python playsound

**Pre-trained Models**
Ensure you have the following pre-trained models in your project directory:
_opencv_face_detector.pbtxt_ - The text graph file for the DNN face detector.
_opencv_face_detector_uint8.pb_ - The binary file containing the trained model.
You can download these files from OpenCV's GitHub repository [https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector].

**Usage**

Clone the repository and navigate to the project folder:

git clone https://github.com/benax-rw/nova-security.git
cd nova-security
Place an alert.wav file in the root directory. This will be the sound played when a face is detected.
Run the script:
_python watch.py_
The system will begin detecting faces from the webcam, play an alert sound on detection, and save the full-frame and cropped face images in the detected_faces/ folder.

**Controls**
Press q to stop the webcam feed and exit the program.

**How It Works**
Face Detection: The system uses OpenCV's deep neural network (DNN) to detect faces from the webcam feed.
Alert Sound: When a face is detected, the system plays the alert.wav sound.
Image Saving: Both the full frame (with the face detection box) and the cropped face images are saved to the detected_faces/ folder. The saved images include a timestamp to avoid overwriting.

**Example Output**
Full-frame with bounding boxes: detected_faces/frame_<timestamp>.jpg
Cropped face images: detected_faces/face_<timestamp>_<index>.jpg
