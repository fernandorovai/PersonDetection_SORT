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
from random import *


class TrackerBbox():
    def __init__(self, boxID, boxCoords):
        self.boxID = boxID
        self.startTime = datetime.datetime.now()
        self.boxCoords = boxCoords
        self.hits = 0
        self.aliveTime = 0
        self.lastUpdate = datetime.datetime.now()
        self.x = random()*100
        self.y = random()*100
        self.trackerCoords = ()
        self.trackerShadow = []
        self.trackShadowSize = 30
        self.color = []
        self.genColor()

    def genColor(self):
        self.color = list(np.random.choice(range(256), size=3))

    def increaseHit(self):
        self.hits += 1

    def getAliveTime(self):
        return self.aliveTime

    def updateAliveTime(self):
        self.aliveTime = (datetime.datetime.now() -
                          self.startTime).total_seconds()

    def resetHits(self):
        self.hits = 0

    def refreshLastUpdate(self):
        self.lastUpdate = datetime.datetime.now()

    def getIdleTime(self):
        idle = datetime.datetime.now() - self.lastUpdate
        return idle.total_seconds()

    def addTrackerShadowPt(self):
        xMin, yMin, xMax, yMax = self.trackerCoords
        xCenter = xMin + (xMax - xMin)/2
        yCenter = yMin + (yMax - yMin)/2
        self.trackerShadow.append({"xCenter": xCenter, "yCenter": yCenter})
        if len(self.trackerShadow) > self.trackShadowSize:
            self.trackerShadow.pop(0)

    def getTrackerShadow(self):
        return self.trackerShadow

    def setTrackerCoords(self, trackerCoords):
        self.trackerCoords = trackerCoords
    
    def getColor(self):
        return self.color

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
        self.personTrackBbIDs = []
        self.faceTrackBbIDs = []
        self.instantFPS = 0
        self.avgFPS = 0
        self.frame = None
        self.startTime = None
        self.avgFilterStartTime = None
        self.detectedPersonHist = {}
        self.detectedFaceHist = {}
        self.detectedPersonHisFiltered = []
        self.sampleTime = 1  # seconds
        self.framesFactor = 0.3
        self.maxIdleTime = 10
        self.filteredDetection = []
        self.postFrequency = 1
        self.videoSize = []

    # return unormalized person bboxes
    def getPersonBboxes(self):
        with self.lock:
            return self.personTrackBbIDs.tolist()

    def getFrame(self):
        img_str = None
        if self.frame is not None:
            _, img_str = cv2.imencode('.png', self.frame)
            img_str = img_str.tobytes()
        return img_str

    def getPersonHis(self):
        with self.lock:
            return self.detectedPersonHist

    def getLiveSummary(self):
        with self.lock:
            return self.detectedPersonHisFiltered

    def filterNumBoxes(self):
        pass

    def getTimeDiffinSec(self, start, end):
        timeDiff = end-start
        return timeDiff.total_seconds()


    def processTracker(self, trackerDetections, trackerHist):
        # Create / update tracked bbox
        for trackedBbox in trackerDetections:
            trackID = str(int(trackedBbox[4]))
            if trackID in trackerHist:
                trackerHist[trackID].increaseHit()
                trackerHist[trackID].updateAliveTime()
                trackerHist[trackID].refreshLastUpdate()
                trackerHist[trackID].setTrackerCoords(trackedBbox[:4])
                trackerHist[trackID].addTrackerShadowPt()
            else:
                trackerHist[trackID] = TrackerBbox(trackID, (0))
            
            # Draw tracker
            self.frame = self.painter.DrawBox(self.frame, trackedBbox.tolist(), self.videoSize, trackerHist[trackID].getColor(), thickness=2)
            self.frame = self.painter.DrawTrackerShadow(self.frame, trackerHist[trackID].getTrackerShadow(), self.videoSize, trackerHist[trackID].getColor())


    def cleanUpBuffers(self, trackerHist):
        # Clean-up old bboxes from buffer
        if len(trackerHist) > 1000:
            for bbox in list(trackerHist):
                if trackerHist[bbox].getIdleTime() > self.maxIdleTime:
                    del trackerHist[bbox]

        # Clean-up history buffer
        if len(self.detectedPersonHisFiltered) > 5:
            self.detectedPersonHisFiltered.pop(0)

    def run(self):
        # Initiliaze timer
        self.startTime = datetime.datetime.now()
        self.avgFilterStartTime = datetime.datetime.now()
        self.startPostTime = datetime.datetime.now()

        # Initialize Trackers
        personTracker = Sort()
        faceTracker = Sort()

        # Open Webcam
        video_capturer = cv2.VideoCapture("testVideos/street360p.mp4")
        # video_capturer = cv2.VideoCapture(0)
        # video_capturer.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        # video_capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        fps = 0

        self.videoSize = (video_capturer.get(3), video_capturer.get(4))
        img_str = None
        fpsList = []

        while video_capturer.isOpened():
            with self.lock:
                # Start time
                start = time.time()
                timeNow = datetime.datetime.now()

                # Read frame
                ret, self.frame = video_capturer.read()
                
                # Person detector
                personBBoxes = self.personDetector.Detect(self.frame, maxThresh=0.7)
                
                # Face detector
                faceBboxes = self.faceDetector.Detect(self.frame, maxThresh=0.7)

                # Update Trackers
                self.personTrackBbIDs = personTracker.update(np.array(personBBoxes))
                self.faceTrackBbIDs = faceTracker.update(np.array(faceBboxes))

                 # Draw fps
                self.frame = self.painter.DrawFPS(self.frame, self.avgFPS)

                # Draw numBoxes
                self.frame = self.painter.DrawTotalBoxes(
                    self.frame, len(self.personTrackBbIDs))

                # Draw person detections
                for personBbox in personBBoxes:
                    self.frame = self.painter.DrawBox(
                        self.frame, personBbox, self.videoSize, color=([255, 255, 255]), thickness=5)
           
                # Draw face detections
                for faceBbox in faceBboxes:
                    self.frame = self.painter.DrawBox(
                        self.frame, faceBbox, self.videoSize, color=([255, 0, 0]), thickness=2)
                    
                    self.frame = self.painter.ApplyGaussian(self.frame, faceBbox[:4], self.videoSize)

                # Measure performance
                seconds = time.time() - start
                fps = 1/seconds
                fpsList.append(fps)

                if len(fpsList) > 20:
                    self.avgFPS = np.array(fpsList).mean()
                    # print("Average FPS: %s" % str(avgFps))
                    fpsList.pop(0)

                # Process person tracker
                self.processTracker(self.personTrackBbIDs, self.detectedPersonHist)
                
                # Process face tracker
                self.processTracker(self.faceTrackBbIDs, self.detectedFaceHist)
            
                # Save detection hist to be consumed later
                if self.getTimeDiffinSec(self.startTime, timeNow) > self.sampleTime:
                    # filter boxes that had present at least in x% of the frames during y seconds
                    # seconds * average FPS * %factor
                    # 15*20*0.4 = 120 hits to be considered a valid box
                    self.filteredDetection = [self.detectedPersonHist[bboxID]
                                              for bboxID in self.detectedPersonHist if self.detectedPersonHist[bboxID].hits > self.sampleTime*self.avgFPS*self.framesFactor]

                    # Reset person boxes hits
                    [self.detectedPersonHist[bbox].resetHits() for bbox in self.detectedPersonHist]

                    # Reset boxes hits
                    [self.detectedFaceHist[bbox].resetHits() for bbox in self.detectedFaceHist]

                    # Reset timer
                    self.startTime = datetime.datetime.now()


                if self.getTimeDiffinSec(self.startPostTime, timeNow) > self.postFrequency:
                    # Print debug
                    # [print(trackerBox) for trackerBox in self.detectedPersonHisFiltered]
                    # Append data
                    self.detectedPersonHisFiltered.append(
                        {"datetime": str(timeNow),
                         "numPerson": len(self.filteredDetection),
                         "avgStayTime": round(np.array([val.getAliveTime() for val in self.filteredDetection]).mean()/60, 2) if len(self.filteredDetection) > 0 else 0})
                    # print("---------------")
                    self.startPostTime = datetime.datetime.now()

                # Clean-up buffers
                self.cleanUpBuffers(self.detectedPersonHist)
                self.cleanUpBuffers(self.detectedFaceHist)

                cv2.imshow('frame', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# start main process thread
mainProcess = MainProcess()
mainProcess.start()

# async function to send socket data


async def sendData(websocket, path):
    while True:
        await websocket.send(json.dumps({"detectionLiveSummary": mainProcess.getLiveSummary(),
                                         "filteredDetectionBoxes": [bbox.__dict__ for bbox in mainProcess.filteredDetection]}, default=str))
        await asyncio.sleep(mainProcess.postFrequency)

# start websocket
start_server = websockets.serve(sendData, "0.0.0.0", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
