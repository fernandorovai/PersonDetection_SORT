"""Person detection with SORT Tracker
Author: Fernando Rodrigues Jr (fernandorovai@hotmail.com)
Date: 24/10/2019
THIS IS A DEMO SCRIPT. IT DOES NOT FIT FOR DEPLOYMENT
"""
from detector import Detector
from sort.sort import *
import cv2
import numpy as np
import threading  # Handle Threads
import time
import asyncio
import websockets
import json
import time
import base64
import sys
import concurrent.futures

"""Main thread resposible for receiving camera frames, 
performing deep learning infereces and extracting statistics.
It will serve as base class to provide data for websocket
"""
class MainProcess(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.lock = threading.Lock()

        # Start detectors
        self.personDetector = Detector('data/person-detection-retail-0013/FP32/person-detection-retail-0013.xml', 'bin/libcpu_extension.so')
        self.faceDetector = Detector('data/face-detection-retail-0004/face-detection-retail-0004.xml','bin/libcpu_extension.so')

        # Data
        self.person_track_bbs_ids = []
        self.instantFPS = 0
        self.avgFPS = 0
        self.frame = None

    # return unormalized person bboxes
    def getPersonBboxes(self):
        with self.lock:
            return self.person_track_bbs_ids.tolist()
    
    def getFrame(self):
        img_str = None
        if self.frame is not None:
            _, img_str = cv2.imencode('.png', self.frame)
            img_str = img_str.tobytes()
        return img_str

    def run(self):
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

        while video_capturer.isOpened():
            with self.lock:
                # Start time
                start = time.time()

                _, self.frame = video_capturer.read()
                personBBoxes = self.personDetector.Detect(self.frame)     
                # faceBboxes = self.faceDetector.Detect(self.frame)
    
                # Update Tracker
                self.person_track_bbs_ids = personTracker.update(np.array(personBBoxes))
                
                # Measure performance
                seconds = time.time() - start
                fps = 1/seconds
                fpsList.append(fps)

                if len(fpsList) > 20:
                    avgFps = np.array(fpsList).mean()
                    print("Average FPS: %s" % str(avgFps))
                    fpsList.pop(0)

# start main process thread
mainProcess = MainProcess()
mainProcess.start()

# async function to send socket data
async def sendData(websocket, path):
    while True:
        # await websocket.send(json.dumps("%s,%s" % ("data:image/jpeg;base64", base64.b64encode(mainProcess.getFrame()).decode())))
        frame = mainProcess.getFrame()
        if frame is not None:          
            await websocket.send(frame)
            # await websocket.send(json.dumps({"bboxes": mainProcess.getPersonBboxes()}))
        await asyncio.sleep(0.01)

# start websocket
start_server = websockets.serve(sendData, "0.0.0.0", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
