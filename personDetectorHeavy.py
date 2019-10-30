#!/usr/bin/env python
"""
Author: Fernando Rodrigues Jr (fernando.junior@hotmail.com)
Date: 24/10/2019
THIS IS A DEMO SCRIPT. IT DOES NOT FIT FOR DEPLOYMENT

Person detection with two different trackings
# 1 - SORT
# 2 - DEEP_SORT
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
import json


class PersonDetector():
    def __init__(self, args):
        log.basicConfig(format="[ %(levelname)s ] %(message)s",
                        level=log.INFO, stream=sys.stdout)
        self.args = args
        self.execNet = None
        self.inputShape = None
        self.inputImageInfo = None
        self.labels = []

      # Load labels
        if args.labels:
            self.labels = self.LoadLabels(args.labels)

      # Load network
        self.ArrangeNetwork()

    def ArrangeNetwork(self):
        # Read IR
        log.info("Reading IR...")
        model_xml = self.args.model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        net = IENetwork(model=model_xml, weights=model_bin)

        # Plugin initialization for specified device and load extensions library if specified
        log.info("Initializing plugin for {} device...".format(self.args.device))
        plugin = IEPlugin(device=self.args.device,
                          plugin_dirs=self.args.plugin_dir)

        if self.args.cpu_extension and 'CPU' in self.args.device:
            plugin.add_cpu_extension(self.args.cpu_extension)

        if plugin.device == "CPU":
            supported_layers = plugin.get_supported_layers(net)
            not_supported_layers = [
                l for l in net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".format(
                    plugin.device, ', '.join(not_supported_layers)))
                log.error(
                    "Please try to specify cpu extensions library path in sample's command line parameters using -l or --cpu_extension command line argument")
                sys.exit(1)

        self.execNet = plugin.load(network=net)

        # Documentation for FasterRCNN
        # net.inputs:
        # {'image_info': <openvino.inference_engine.ie_api.InputInfo object at 0x7528af80>, 'image_tensor': <openvino.inference_engine.ie_api.InputInfo object at 0x72fb9020>}

        # net.inputs['image_tensor'].shape
        # NCHW [1, 3, 600, 600]

        # net.inputs['image_info'].shape
        # creates input to store input image height, width and scales (usually 1.0s)
        # [1, 3] ~ [h, w, 1]

        # net.outputs:
        # {'detection_output': <openvino.inference_engine.ie_api.OutputInfo object at 0x7522b7f0>}

        # net.outputs['detection_output'].shape
        # [1, 1, 500, 7]

        # NCHW [1, 3, 600, 600]
        self.inputShape = net.inputs['image_tensor'].shape
        self.inputImageInfo = np.asarray(
            [[self.inputShape[2], self.inputShape[3], 1]], dtype=np.float32)  # [1, 3] ~ [h, w, 1]

    @staticmethod
    def LoadLabels(labelsPath):
        labels = []
        with open(labelsPath) as json_file:
            labels = json.load(json_file)
        return(labels)

    def PreProcessFrame(self, frame):
        n, c, h, w = [1, 3, self.inputShape[2], self.inputShape[3]]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (w, h))
        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        return image

    def DrawBox(self, frame, coords, videoRes, color=(0, 0, 255), thickness=5):
        trackID = ''
        width, height = videoRes

        xMin, yMin, xMax, yMax, id = coords

        print(coords)
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
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 2
        try:
            cv2.rectangle(frame, (int(xMin), int(yMin)),
                          (int(xMax), int(yMax)), color, thickness)
            cv2.putText(frame, str(trackID), bottomLeftCornerOfText,
                        font, fontScale, fontColor, lineType)
            cv2.putText(frame, str(self.labels[id]['display_name']), bottomLeftCornerOfText,
                        font, fontScale, fontColor, lineType)
        except Exception as e:
            cv2.putText(frame, str(id), (int(xMin), int(yMax)),
                        font, fontScale, fontColor, lineType)
        return frame

    @staticmethod
    def PosProcessing(rawResultArr, maxTresh=0.5):
        results = []
        for idx, data in enumerate(rawResultArr):
            score = data[2]
            if score > maxTresh:
                # log.info("Score: %s" % str(score))
                imageId = data[0]
                label = int(data[1])
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

                results.append((xMin, yMin, xMax, yMax, label))
        return results

    def Detect(self, frame):
        t0 = time.time()
        processedFrame = self.PreProcessFrame(frame)
        res = self.execNet.infer(
            {'image_tensor': processedFrame, 'image_info': self.inputImageInfo})
        rawResultArr = np.squeeze(res['detection_output'])
        infer_time = (time.time()-t0)
        results = self.PosProcessing(rawResultArr, maxTresh=0.8)

        return results

    def bboxToWH(self, bboxes):
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

    def bboxToXY(self, bbox, videoSize):
        xMin = bbox[0]/videoSize[0]
        yMin = bbox[1]/videoSize[1]
        xMax = (bbox[2] + xMin)/videoSize[0]
        yMax = (bbox[3] + yMin)/videoSize[1]

        return (xMin, yMin, xMax, yMax)

    def returnJustPersonBox(self, bboxes):
        personBbox = []
        for box in bboxes:
            if self.labels[int(box[4])]['display_name'] == 'person':
                personBbox.append(box)
        return personBbox


if __name__ == '__main__':
    def build_argparser():
        parser = ArgumentParser()
        parser.add_argument(
            "-m", "--model", help="Path to an .xml file with a trained model.", required=True, type=str)
        parser.add_argument("-l", "--cpu_extension",
                            help="MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels "
                            "impl.", type=str, default=None)
        parser.add_argument(
            "-pp", "--plugin_dir", help="Path to a plugin folder", type=str, default=None)
        parser.add_argument("-d", "--device",
                            help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                            "will look for a suitable plugin for device specified (CPU by default)", default="CPU",
                            type=str)
        parser.add_argument("-lbl", "--labels",
                            help="Labels mapping file", default=None, type=str)
        return parser

    args = build_argparser().parse_args()
    personDetector = PersonDetector(args)
    deepSort = False
    sort = True

    # SORT
    mot_tracker = Sort()

    # DEEP SORT PARAMS
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None

    model_filename = 'deep_sort/models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    # Open Webcam
    video_capturer = cv2.VideoCapture(0)
    video_capturer.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    video_capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    infer_time = []
    videoSize = (video_capturer.get(3), video_capturer.get(4))
    while video_capturer.isOpened():

        _, frame = video_capturer.read()
        bboxes = personDetector.Detect(frame)
        # Draw detections
        for bbox in bboxes:
            frame = personDetector.DrawBox(
                frame, bbox, videoSize, thickness=5)

        if deepSort:
            bboxes = personDetector.returnJustPersonBox(bboxes)
            features = encoder(frame, np.array(
                personDetector.bboxToWH(bboxes)))
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(
                personDetector.bboxToWH(bboxes), features)]

            # Call tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                frame = personDetector.DrawBox(frame, personDetector.bboxToXY(
                    bbox, videoSize), videoSize, color=(255, 255, 255), thickness=2)
                cv2.putText(frame, str(track.track_id), (int(
                    bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
        elif sort:
            bboxes = personDetector.returnJustPersonBox(bboxes)
            track_bbs_ids = mot_tracker.update(np.array(bboxes))

            # Draw tracker
            for person in track_bbs_ids:
                frame = personDetector.DrawBox(
                    frame, person.tolist(), videoSize, color=(255, 255, 255), thickness=2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capturer.release()
    cv2.destroyAllWindows()
    del exec_net
    del plugin
