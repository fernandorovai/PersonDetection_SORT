#!/usr/bin/env python
"""
Author: Fernando Rodrigues Jr (fernandorovai@hotmail.com)
Date: 24/10/2019
THIS IS A DEMO SCRIPT. IT DOES NOT FIT FOR DEPLOYMENT

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


class Detector():
    def __init__(self, modelPath, extensionPath=None):
        self.device = 'GPU'
        self.cpu_extension = extensionPath
        self.model_xml = modelPath
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.input_size = None
        self.exec_net = None
        self.input_blob = None
        self.infer_time = []

        self.ArrangeNetwork()

    def ArrangeNetwork(self):
        log.basicConfig(format="[ %(levelname)s ] %(message)s",
                        level=log.INFO, stream=sys.stdout)
        # Plugin initialization for specified device and load extensions library if specified
        plugin = IEPlugin(device=self.device)
        if self.cpu_extension and 'CPU' in self.device:
            plugin.add_cpu_extension(self.cpu_extension)

        # Read IR
        net = IENetwork.from_ir(model=self.model_xml, weights=self.model_bin)
        assert len(net.inputs.keys()
                   ) == 1, "Sample supports only single input topologies"
        assert len(
            net.outputs) == 1, "Sample supports only single output topologies"
        self.input_blob = next(iter(net.inputs))

        # Set input size
        self.input_size = net.inputs[self.input_blob].shape[2:]

        # Load network to the plugin
        self.exec_net = plugin.load(network=net)
        del net
        # Warmup with last image to avoid caching
        self.exec_net.infer(inputs={self.input_blob: np.zeros(
            (1, 3, self.input_size[0], self.input_size[1]))})

    def PreProcessFrame(self, frame):
        n, c, h, w = [1, 3, self.input_size[0], self.input_size[1]]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (w, h))
        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        return image

    @staticmethod
    def DrawBox(frame, coords, videoRes, color=(0, 0, 255), thickness=5):
        if coords is None:
            return frame

        trackID = ''
        width, height = videoRes

        xMin, yMin, xMax, yMax, labelID = coords

        if xMin < 0:
            xMin = 0
        if yMin < 0:
            yMin = 0
        if xMax < 0:
            xMax = 0
        if yMax < 0:
            yMax = 0

        xMin = xMin*width
        xMax = xMax*width
        yMin = yMin*height
        yMax = yMax*height

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (int(xMin + (xMax-xMin)/2), int(yMin))
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        try:
            cv2.rectangle(frame, (int(xMin), int(yMin)),
                          (int(xMax), int(yMax)), color, thickness)
        except Exception as e:
            print(e)
        return frame

    @staticmethod
    def PosProcessing(rawResultArr, maxThresh=0.5):
        boxes = []
        for idx, data in enumerate(rawResultArr):
            score = data[2]
            if score > maxThresh:
                # log.info("Score: %s" % str(score))
                imageId = data[0]
                label = data[1]
                xMin, yMin = (float(data[3]), float(data[4]))
                xMax, yMax = (float(data[5]), float(data[6]))
                if xMin < 0:
                    xMin = 0
                if yMin < 0:
                    yMin = 0
                if xMax < 0:
                    xMax = 0
                if yMax < 0:
                    yMax = 0

                # Avoid 0,0,0,0
                if(xMin and yMin and xMax and yMax == 0.0):
                    continue

                boxes.append((xMin, yMin, xMax, yMax, label))
        return boxes

    def Detect(self, frame, maxThresh):
        t0 = time.time()
        processedFrame = self.PreProcessFrame(frame)
        res = self.exec_net.infer(inputs={self.input_blob: processedFrame})
        rawResultArr = np.squeeze(res['detection_out'])
        boxes = self.PosProcessing(rawResultArr, maxThresh)

        #  log.info("Face Detection Average FPS: {} FPS".format(1/np.average(np.asarray(self.infer_time))))
        return boxes

    def GetUnormalizedBboxes(self, bboxes, videoRes):
        width, height = videoRes
        return [(box['xMin']*width, box['yMin']*height, box['xMax']*width, box['yMax']*height) for box in bboxes]

    def BboxToWH(self, bboxes):
        whBboxes = []
        for bbox in bboxes:
            xMin = bbox[0]*videoSize[0]
            yMin = bbox[1]*videoSize[1]
            xMax = bbox[2]*videoSize[0]
            yMax = bbox[3]*videoSize[1]
            w = xMax - xMin
            h = yMax - yMin
            whBboxes.append((xMin, yMin, w, h))
        return whBboxes

    def BboxToXY(self, bbox, videoSize):
        xMin = bbox[0]/videoSize[0]
        yMin = bbox[1]/videoSize[1]
        xMax = (bbox[2] + xMin)/videoSize[0]
        yMax = (bbox[3] + yMin)/videoSize[1]
        label = bbox[4]
        return (xMin, yMin, xMax, yMax, label)
