import cv2 as cv
import time
import os
from playsound import playsound

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

# Load model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

# Load face detection network
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Set to use CPU or GPU if desired
faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)  # For CPU

# Open the webcam
cap = cv.VideoCapture(0)  # Use webcam, replace 0 with index if needed

# Folder to save images
save_folder = "detected_faces"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize alert sound file path
alert_sound = "alert.wav"  # Make sure alert.wav exists in the same directory

padding = 20

# Loop through webcam feed
while True:
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("No frames grabbed. Exiting...")
        break

    # Detect faces
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face detected, checking next frame")
        continue

    # Play the alert sound when a face is detected
    playsound(alert_sound)

    # Save the full frame with face boxes
    full_frame_filename = os.path.join(save_folder, f"frame_{int(time.time())}.jpg")
    cv.imwrite(full_frame_filename, frameFace)
    print(f"Full frame saved at: {full_frame_filename}")

    # Save each cropped face
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        
        # Ensure the bounding box coordinates are within valid ranges
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Crop the face region
        face = frame[y1:y2, x1:x2]

        # Check if the cropped face is valid and not empty
        if face.size > 0:
            face_filename = os.path.join(save_folder, f"face_{int(time.time())}_{i}.jpg")
            cv.imwrite(face_filename, face)
            print(f"Cropped face {i} saved at: {face_filename}")
        else:
            print(f"Face {i} is empty and will not be saved.")

    # Display the result
    num_faces = len(bboxes)
    annotation = f"Faces detected: {num_faces}"
    cv.putText(frameFace, annotation, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
    
    cv.imshow("Face Recognition", frameFace)
    
    print(f"Processing time: {time.time() - t:.3f} seconds")

    # Exit the loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv.destroyAllWindows()