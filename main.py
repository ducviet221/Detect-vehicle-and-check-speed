import math
import cv2
import numpy as np
import torch
torch.cuda.is_available()
import os
import time
import dlib
import argparse

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow.compat.v1 import ConfigProto

# add link path
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# from tensorflow.compat.v1 import InteractiveSession
cap = cv2.VideoCapture(args['input'])
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
count = 0

center_point_pre = []
tracking_objects = {}
speed_checks = {}
time_first = {}
time_ends = {}

track_id = 0

# calculate 2 image update of distance
area = [(20,270),(20,272),(620,272),(620,270)]
zone = [(20,290),(20,292),(620,292),(620,290)]

frameCounter = 0
currentCarID = 0
carTracker = {}
carLocation1 = {}
carLocation2 = {}
speed = [None]*1000

fps_start_time = 0
fps = 0

# distance between car and camera = 30m
def estimateSpeed(location1, location2, fps):
	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ratio pixel/meter
	ppm = 0.2
	d_meters = d_pixels / ppm
	# print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    # ratio frame/second
	speed = d_meters * fps * 3.6
	return speed

try:
    while(True):

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        
        _, frame = cap.read()
        # print("Round {}".format(frameCounter))
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1/time_diff
        fps_start_time = fps_end_time
        
        frame = cv2.resize(frame, (640, 620))
        # frame = frame1[150:640, 0:620]        
        # h, w = frame.shape[:2]

        frameCounter += 1
        carIDtoDelete = []
        writer = None

        detected = model(frame)
        results = detected.pandas().xyxy[0].to_dict(orient="records")
        center_point_cur = []
        count += 1

        cv2.polylines(frame,[np.array(area,np.int32)],True,(0,0,255),2)
        cv2.polylines(frame, [np.array(zone, np.int32)], True, (0, 255, 0), 2)
        
        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(frame)
            if trackingQuality < 7:
                carIDtoDelete.append(carID)
        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        detected = model(frame)
        results = detected.pandas().xyxy[0].to_dict(orient="records")
        for result in results:
            clas = result['class']
            confid = result['confidence']
            if confid > args["confidence"]:
                if clas == 2:

                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])
                    x_bar = int(x1+(x2-x1)/2)
                    y_bar = int(y1+(y2-y1)/2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        #result = cv2.pointPolygonTest(np.array(area, np.int32), pt, False)
                        #cv2.circle(frame, pt, 2, (255, 0, 255), -1)
                    matchCarID  = None
                    for carID in carTracker.keys():
                        trackerPos = carTracker[carID].get_position()
        
                        t_x = int(trackerPos.left())
                        t_y = int(trackerPos.top())
                        t_w = int(trackerPos.width())
                        t_h = int(trackerPos.height())

                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h

                        if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x1 <= t_x_bar <= x2) and (y1 <= t_y_bar <= y2)):
                            matchCarID = carID
                    if matchCarID is None:
                        # print('Create new tracker ' + str(currentCarID))

                        tracker = dlib.correlation_tracker()
                        tracker.start_track(frame, dlib.rectangle(x1, y1, x2, y2))

                        carTracker[currentCarID] = tracker
                        carLocation1[currentCarID] = [x1, y1, x2 - x1, y2 - y1]
                        currentCarID += 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())
                # cv.rectangle(resultImage, (t_x, t_y, t_w, t_h), rec, 2)
            carLocation2[carID] = [t_x, t_y, t_w, t_h] 

            # print(fps)   
        for i in carLocation1.keys():
            if frameCounter%1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]        

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2], fps)
                        
                    if speed[i] != None and y1 >= 180:
                        cv2.putText(frame, str(int(speed[i])) + "km/hr", (int(x1 + w1/2), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[0], frame.shape[1]), True)

        cv2.imshow('window', frame)
        writer.write(frame)

        if cv2.waitKey(33) == 27:
            break
    cv2.destroyAllWindows()
except:
    print('End')
cap.release()
cv2.destroyAllWindows()