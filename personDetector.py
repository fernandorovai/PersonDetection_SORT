#!/usr/bin/env python
"""
Author: Fernando Rodrigues Jr (fernando.junior@hotmail.com)
Date: 24/10/2019
THIS IS A DEMO SCRIPT. IT DOES NOT FIT FOR DEPLOYMENT

Person detection with two different trackings
#1 - SORT
#2 - DEEP_SORT
"""
from __future__ import print_function
import os
import cv2
import sys
import time
import math
import numpy as np
import logging as log
from argparse import ArgumentParser
from openvino.inference_engine import IENetwork, IEPlugin
from sort.sort import *
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet


class PersonDetector():
    def __init__(self, modelPath, extensionPath):
        self.device         = 'GPU'
        self.cpu_extension  = extensionPath
        self.model_xml      = modelPath
        self.model_bin      = os.path.splitext(self.model_xml)[0] + ".bin"
        self.input_size     = (320,544)
        self.exec_net       = None
        self.input_blob     = None
        self.infer_time     = []

        self.ArrangeNetwork()

    def ArrangeNetwork(self):
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)   
         # Plugin initialization for specified device and load extensions library if specified
        plugin = IEPlugin(device=self.device)
        if self.cpu_extension and 'CPU' in self.device:
            plugin.add_cpu_extension(self.cpu_extension)
    
        # Read IR
        net = IENetwork.from_ir(model=self.model_xml, weights=self.model_bin)
        assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
        assert len(net.outputs) == 1, "Sample supports only single output topologies"
        self.input_blob = next(iter(net.inputs))

        # Load network to the plugin
        self.exec_net = plugin.load(network=net)
        del net
        # Warmup with last image to avoid caching
        self.exec_net.infer(inputs={self.input_blob: np.zeros((1, 3, self.input_size[0], self.input_size[1]))})

    @staticmethod
    def PreProcessFrame(frame):
        n, c, h, w = [1, 3, self.input_size[0], self.input_size[1]]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        image = image.reshape((n, c, h, w))
        return image

    @staticmethod
    def DrawBox(frame, coords, videoRes, color=(0,0,255), thickness=5):
        trackID = ''
        width, height = videoRes

        if len(coords) == 5:
            xMin,yMin,xMax,yMax,trackID = coords
        else:
            xMin,yMin,xMax,yMax = coords

        if xMin < 0:
            xMin= 0
        if yMin < 0:
            yMin= 0
        if xMax < 0:
            xMax= 0
        if yMax < 0:
            yMax= 0

        xMin = xMin*width
        xMax = xMax*width
        yMin = yMin*height
        yMax = yMax*height
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(xMin + (xMax-xMin)/2), int(yMin))
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        try:
            cv2.rectangle(frame, (int(xMin), int(yMin)), (int(xMax),int(yMax)), color, thickness)
            cv2.putText(frame,str(trackID), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        except Exception as e:
            print(e)
        return frame

    @staticmethod
    def PosProcessing(rawResultArr, maxTresh=0.5):
        faces = []
        for idx, data in enumerate(rawResultArr):
            score = data[2]
            if score > maxTresh:
                # log.info("Score: %s" % str(score))
                imageId = data[0]
                label = data[1]
                xMin, yMin = (float(data[3]), float(data[4]))
                xMax, yMax = (float(data[5]), float(data[6]))
                if xMin < 0:
                    xMin= 0
                if yMin < 0:
                    yMin= 0
                if xMax < 0:
                    xMax= 0
                if yMax < 0:
                    yMax= 0
                
                # Avoid 0,0,0,0
                if(xMin and yMin and xMax and yMax == 0.0):
                    continue

                faces.append((xMin,yMin,xMax,yMax))
        return faces

    def Detect(self, frame):
        t0 = time.time()
        processedFrame = self.PreProcessFrame(frame)
        res            = self.exec_net.infer(inputs={self.input_blob: processedFrame})
        rawResultArr   = np.squeeze(res['detection_out'])
        faces          = self.PosProcessing(rawResultArr, maxTresh=0.7)
        
        # self.infer_time.append((time.time()-t0))
        
        # Release last buffer element
        # if len(self.infer_time) > 100:
        #     self.infer_time.pop()

        #  log.info("Face Detection Average FPS: {} FPS".format(1/np.average(np.asarray(self.infer_time))))
        return faces

    def bboxToWH(self, bboxes):
        whBboxes = []
        for bbox in bboxes:
            xMin = bbox[0]*videoSize[0]
            yMin = bbox[1]*videoSize[1]
            xMax = bbox[2]*videoSize[0]
            yMax = bbox[3]*videoSize[1]
            w = xMax - xMin
            h = yMax - yMin
            whBboxes.append((xMin,yMin,w,h))
        return whBboxes


    def bboxToXY(self, bbox, videoSize):
        xMin = bbox[0]/videoSize[0]
        yMin = bbox[1]/videoSize[1]
        xMax = (bbox[2] + xMin)/videoSize[0]
        yMax  = (bbox[3] + yMin)/videoSize[1]

        return (xMin,yMin,xMax,yMax)

if __name__ == '__main__':
    def build_argparser():
        parser = ArgumentParser()
        parser.add_argument("-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
        parser.add_argument("-l", "--cpu_extension",
                            help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                                "impl.", type=str, default=None)
        parser.add_argument("-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
        parser.add_argument("-d", "--device",
                            help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                                "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                            type=str)
        return parser

    args = build_argparser().parse_args()
    personDetector = PersonDetector(args.model, args.cpu_extension)
    deepSort = False
    sort = True

    ####### SORT 
    mot_tracker = Sort() 

    ######## DEEP SORT PARAMS
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None    
   
    model_filename = 'deep_sort/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
   
    # Open Webcam
    video_capturer = cv2.VideoCapture(0)
    video_capturer.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    video_capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    infer_time = []
    videoSize = (video_capturer.get(3), video_capturer.get(4))
    while video_capturer.isOpened():
        _,frame = video_capturer.read()
        bboxes = personDetector.Detect(frame)     

        # Draw detections
        for bbox in bboxes:
            frame = personDetector.DrawBox(frame, bbox, videoSize,thickness=5)

        if deepSort:
            features = encoder(frame,np.array(personDetector.bboxToWH(bboxes)))     
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(personDetector.bboxToWH(bboxes), features)]

            # Call tracker
            tracker.predict()
            tracker.update(detections)
            
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                frame = personDetector.DrawBox(frame, personDetector.bboxToXY(bbox,videoSize), videoSize, color=(255,255,255),thickness=2)
                cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
        elif sort:
            track_bbs_ids = mot_tracker.update(np.array(bboxes))        

            # Draw tracker
            for person in track_bbs_ids:
                frame = personDetector.DrawBox(frame, person.tolist(), videoSize, color=(255,255,255),thickness=2)
        
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capturer.release()
    cv2.destroyAllWindows()
    del exec_net
    del plugin
