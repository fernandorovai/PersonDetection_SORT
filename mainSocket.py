"""Person detection with SORT Tracker
Author: Fernando Rodrigues Jr (fernandorovai@hotmail.com)
Date: 24/10/2019
THIS IS A DEMO SCRIPT. IT DOES NOT FIT FOR DEPLOYMENT
"""

from detector import Detector
from sort.sort import *
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet
import cv2
import numpy as np
from threadedSocket import ThreadedSocket
import threading  # Handle Threads
import time

# Start detectors
personDetector = Detector('data/person-detection-retail-0013/FP32/person-detection-retail-0013.xml')
faceDetector = Detector('data/face-detection-retail-0004/face-detection-retail-0004.xml')

# Select tracker type
deepSort = False
sort = True

# Initialize Trackers
personTracker = Sort() 

# Open Webcam
video_capturer = cv2.VideoCapture(0)
video_capturer.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
video_capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

infer_time = []
fps = 0

videoSize = (video_capturer.get(3), video_capturer.get(4))
img_str = None
fpsList = []
avgFps = 0

def run():
    while video_capturer.isOpened():
        # Start time
        start = time.time()

        _,frame = video_capturer.read()
        personBBoxes = personDetector.Detect(frame)     
        faceBboxes = faceDetector.Detect(frame)

        # Draw person detections
        for bbox in personBBoxes:
            frame = personDetector.DrawBox(frame, bbox, videoSize,thickness=5)

        # Draw face detections
        for bbox in faceBboxes:
            frame = faceDetector.DrawBox(frame, bbox, videoSize,thickness=5)
        
        # Update Tracker
        person_track_bbs_ids = personTracker.update(np.array(personBBoxes))        

        # Draw tracker
        for person in person_track_bbs_ids:
            frame = personDetector.DrawBox(frame, person.tolist(), videoSize, color=(255,255,255),thickness=2)
        
        if frame is not None:
            _, img_str = cv2.imencode('.jpg', frame)
            img_str = img_str.tobytes()

        # Measure performance
        seconds = time.time() - start
        fps = 1/seconds
        fpsList.append(fps)

        if len(fpsList) > 20:
            avgFps = np.array(fpsList).mean()
            print("Average FPS: %s" % str(avgFps))
            fpsList.pop(0)

        # Send data via socket
        threadedSocket.updateData(personBBoxes)
        threadedSocket.updateFrame(img_str)
        

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capturer.release()
    cv2.destroyAllWindows()
    del exec_net
    del plugin

threading.Thread(target=run, args=()).start()

# Create and start socket
threadedSocket = ThreadedSocket()
threadedSocket.start()