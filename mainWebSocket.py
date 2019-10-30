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
from painter import Painter
import datetime
import pandas as pd


class TrackerBbox():
    def __init__(self, boxID, boxCoords):
        self.boxID = boxID
        self.startTime = datetime.datetime.now()
        self.boxCoords = boxCoords
        self.hits = 0
        self.aliveTime = 0

    def increaseHit(self):
        self.hits += 1

    def getAliveTime(self):
        return self.aliveTime

    def updateAliveTime(self):
        self.aliveTime = alive = (
            datetime.datetime.now() - self.startTime).total_seconds()


"""Main thread resposible for receiving camera frames,
performing deep learning infereces and extracting statistics.
It will serve as base class to provide data for websocket
"""


class MainProcess(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.lock = threading.Lock()

        # Start detectors
        self.personDetector = Detector(
            'data/person-detection-retail-0013/FP32/person-detection-retail-0013.xml', 'bin/libcpu_extension.so')
        self.faceDetector = Detector(
            'data/face-detection-retail-0004/face-detection-retail-0004.xml', 'bin/libcpu_extension.so')

        # Start painter
        self.painter = Painter()

        # Data
        self.person_track_bbs_ids = []
        self.instantFPS = 0
        self.avgFPS = 0
        self.frame = None
        self.startTime = None
        self.avgFilterStartTime = None
        self.detectedPersonHist = {}
        self.detectedPersonHisFiltered = []
        self.sampleTime = 10  # seconds
        self.framesFactor = 0.3

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

    def getPersonHis(self):
        with self.lock:
            return self.detectedPersonHis

    def getPersonHistFiltered(self):
        with self.lock:
            return self.detectedPersonHisFiltered

    def filterNumBoxes(self):
        pass

    def getTimeDiffinSec(self, start, end):
        timeDiff = end-start
        return timeDiff.total_seconds()

    def run(self):
        # Initiliaze timer
        self.startTime = datetime.datetime.now()
        self.avgFilterStartTime = datetime.datetime.now()

        # Initialize Trackers
        personTracker = Sort()

        # Open Webcam
        video_capturer = cv2.VideoCapture(1)
        # video_capturer.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        # video_capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
                timeNow = datetime.datetime.now()

                _, self.frame = video_capturer.read()
                personBBoxes = self.personDetector.Detect(
                    self.frame, maxThresh=0.5)
                # faceBboxes = self.faceDetector.Detect(self.frame, maxThresh=0.9)

                # Update Tracker
                self.person_track_bbs_ids = personTracker.update(
                    np.array(personBBoxes))

                # Measure performance
                seconds = time.time() - start
                fps = 1/seconds
                fpsList.append(fps)

                if len(fpsList) > 20:
                    self.avgFPS = np.array(fpsList).mean()
                    # print("Average FPS: %s" % str(avgFps))
                    fpsList.pop(0)

                # Draw fps
                self.frame = self.painter.DrawFPS(self.frame, self.avgFPS)

                # Draw numBoxes
                self.frame = self.painter.DrawTotalBoxes(
                    self.frame, len(self.person_track_bbs_ids))

                # Draw detections
                for personBbox in personBBoxes:
                    self.frame = self.painter.DrawBox(
                        self.frame, personBbox, videoSize, color=(255, 0, 0), thickness=5)

                # Draw tracker
                for personBbox in self.person_track_bbs_ids:
                    trackID = str(int(personBbox[4]))
                    if trackID in self.detectedPersonHist:
                        self.detectedPersonHist[trackID].increaseHit()
                        self.detectedPersonHist[trackID].updateAliveTime()
                    else:
                        self.detectedPersonHist[trackID] = TrackerBbox(
                            trackID, (0))
                    self.frame = self.painter.DrawBox(
                        self.frame, personBbox.tolist(), videoSize, color=(255, 255, 255), thickness=2)

                # Save detection hist to be consumed
                if self.getTimeDiffinSec(self.startTime, timeNow) > self.sampleTime:
                    # filter boxes that had present at least in x% of the frames during y seconds
                    # seconds * average FPS * %factor
                    # 15*20*0.4 = 120 hits to be considered a valid box
                    filteredDetection = [self.detectedPersonHist[bboxID]
                                         for bboxID in self.detectedPersonHist if self.detectedPersonHist[bboxID].hits > self.sampleTime*self.avgFPS*self.framesFactor]
                    self.detectedPersonHisFiltered.append(
                        {"datetime": str(timeNow), "numPerson": len(filteredDetection)})
                    # "avgStayTime": round(np.array([val.getAliveTime() for val in filteredDetection]).mean()/60, 2) if len(filteredDetection) > 0 else 0)
                    print("---------------")
                    print([val.startTime for val in filteredDetection])
                    [print("hits: %s, trackID: %s" % (str(trackerBox.hits), str(
                        trackerBox.boxID))) for trackerBox in filteredDetection]
                    print({"datetime": str(timeNow),
                           "numPerson": len(filteredDetection),
                           "avgStayTime": round(np.array([val.getAliveTime() for val in filteredDetection]).mean()/60, 2) if len(filteredDetection) > 0 else 0})

                    self.detectedPersonHist = {}
                    self.startTime = datetime.datetime.now()

                # Clean history
                if len(self.detectedPersonHisFiltered) > 1000:
                    self.detectedPersonHisFiltered = []

            cv2.imshow('frame', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# start main process thread
mainProcess = MainProcess()
mainProcess.start()

# async function to send socket data
async def sendData(websocket, path):
    while True:
        # await websocket.send(json.dumps("%s,%s" % ("data:image/jpeg;base64", base64.b64encode(mainProcess.getFrame()).decode())))
        # await websocket.send(frame)
        await websocket.send(json.dumps({"detectionHistory": mainProcess.getPersonHistFiltered()}))
        await asyncio.sleep(10)

# start websocket
start_server = websockets.serve(sendData, "0.0.0.0", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
