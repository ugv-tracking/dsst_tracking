import sys
import os
import requests
import argparse
import base64
import time
import json
import math
import cv2
import numpy as np
from numpy import empty, nan
from numpy import *
from __future__ import print_function
import pylab
import scipy.misc
from optparse import OptionParser

def area_2(p1, p2, p3):
    return abs(p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p1[1] - p1[0]*p3[1] - p2[0]*p1[1] - p3[0]*p2[1])

def contains(triangle, h, px, py):
    p1 = (triangle[0], triangle[1])
    p2 = (triangle[2], h)
    p3 = (triangle[3], h)
    p = (px, py)
    if area_2(p1, p2, p3) == area_2(p1, p2, p) + area_2(p1, p3, p) + area_2(p2, p3, p):
        return 1
    return 0

def isintersect(triangle, rectangle, h):
    if len(triangle) < 4:
        return 1
    if contains(triangle, h, rectangle[0], rectangle[1]) or contains(triangle, h, rectangle[0], rectangle[3]) or\
     contains(triangle, h, rectangle[2], rectangle[3]) or contains(triangle, h, rectangle[2], rectangle[1]):
            return 1
    if triangle[0] >= rectangle[0] and triangle[0] <= rectangle[2] and triangle[1] >= rectangle[1] and\
     triangle[1] <= rectangle[2]:
            return 1
    return 0

def drawing_objdet(res, drawing_board, _h, _w, lane_triangle):
    r_jo = res
    color = [(0, 255, 0), (255, 255, 0), (0, 0, 255)]
    maxy = 0
    for item in r_jo['objs']:
        _x1 = int(_w * item['left'])
        _y1 = int(_h * item['top'])
        _x2 = int(_w * item['right'])
        _y2 = int(_h * item['bottom'])
        _confidence = item['confidence']
        _type = item['type']
        if _confidence > 0.9:
            if _type == 'CAR':
                cl = color[1]
            elif _type == 'PEDESTRIAN':
                cl = color[0]
            elif _type == 'BICYCLE':
                cl = color[2]
            
            if isintersect(lane_triangle, (_x1, _y1, _x2, _y2), _h):
                if _y2 > maxy:
                    maxy = _y2
                    forward_car = (_x1, _y1, _x2, _y2)
                    fc = cl
    if 'forward_car' in dir():
        cv2.rectangle(drawing_board, (forward_car[0], forward_car[1]), (forward_car[2], forward_car[3]), fc, 1)
        valid = True
        bbox = np.array([forward_car[0], forward_car[1], forward_car[2]-forward_car[0], forward_car[3]-forward_car[1]])
    else:
        bbox = np.array([0,0,0,0])
        valid = False
    
    tl = bbox[:2]
    hw = bbox[2:4]
    return valid, tl, hw

def response_visualization(response):
    t = np.asarray(bytearray(base64.b64decode(response)))
    response = cv2.imdecode(t, 0)
    cls = (response != 255).astype(np.uint8)
    response[cls == 0] = 0

    hsv = np.zeros(response.shape + (3,), dtype=np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 0] = response / 2
    hsv[:, :, 2] = cv2.normalize(cls, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def render(frame, result, w):
    if 'response' in result and result['response'] is not None:
        response = response_visualization(result['response'])
        frame = cv2.addWeighted(frame, .7, response, .3, 0)
        
    left_bound = 0
    right_bound = w
    tup1 = ()
    tup2 = ()
    for item in result['lanes']:
        p1 = tuple(item['control_points'][0])
        p2 = tuple(item['control_points'][1])
        if p2[0] <= w/2 and p2[0] >= left_bound:
            left_bound = p2[0]
            pleft = p2
        if p2[0] >= w/2 and p2[0] <= right_bound:
            right_bound = p2[0]
            pright = p2

    if 'pleft' in dir():
        cv2.line(frame, p1, pleft, color = (0, 255, 0), thickness = 2)
        tup1 = p1 + (pleft[0], )
        
    if 'pright' in dir():
        cv2.line(frame, p1, pright, color = (0, 255, 0), thickness = 2)
        tup2 = (pright[0], )

    return frame, tup1 + tup2

vid = cv2.VideoCapture("../../output_5min.mp4")

tic = time.time()
# tracking pram
padding = 1.0 
output_sigma_factor = 1 / float(16)
sigma = 0.2 
lambda_value = 1e-2 
interpolation_factor = 0.075
Dsst_valid = False

# tracking varias
sz = None
cos_window = None
pos = None
global z, response
z = None
alphaf = None
response = None

# input DSST
sys.path.append("../build")
import DSST
dsst = DSST.Tracker()
dsst.setParam()

# initialization for dsst

# ret, im = vid.read()
# dsst.setBbox(300,240,600,800)
# dsst.reinit(im)

# update tracking

#     ret, im = vid.read()
#     dsst.update(im)

while vid.isOpened():
    ret, im = vid.read()
    if not ret or im is None:
        break
  
    im_compress = cv2.imencode('.jpg', im)[1]
    encoded_string = base64.b64encode(im_compress)

    payload = {'image_base64': encoded_string, 'image_name': ""}
    h = im.shape[0]
    w = im.shape[1]
    
    loop = time.time() - tic
    
    if  loop > 2:
        tic = time.time()
        
        # drawing lines
        rline = requests.post('http://10.128.8.10:8005/v1/analyzer/lane', data=json.dumps(payload))
        result_line = json.loads(rline.text)
        (frame, lane_triangle) = render(im, result_line, w)
        
        # drawing cars
        rcar = requests.post('http://10.128.2.5:17001/v1/analyzer/objdetect', data=json.dumps(payload))
        result_car = json.loads(rcar.text)
        valid_, tl_, hw_ = drawing_objdet(result_car, frame, h, w, lane_triangle)
        
        # TODO initial DSST
        if valid_ == True:
            print('Find new target, reinit DSST')
            im = frame
            dsst.setBbox(tl_[0],tl_[1],hw_[0],hw_[1])
            dsst.reinit(im)
            Dsst_valid = True
        else:
            Dsst_valid = False
    else:          
        if Dsst_valid == True:
            dsst.update(im)
            
#             Display Results
#             if dsst.tFound == 1:
    
            _x = int(dsst.tFound.x)
            _y = int(dsst.tFound.y)
            _height = int(dsst.tFound.width)
            _width  = int(dsst.tFound.height)
            cv2.line(im, (_x, _y), (_x + _width, _y), (255, 0, 0), 4)
            cv2.line(im, (_x, _y), (_x , _y + _height), (255, 0, 0), 4)
            cv2.line(im, (_x, _y + _height), (_x + _width, _y + _height), (255, 0, 0), 4)
            cv2.line(im, (_x + _width, _y + _height), (_x + _width, _y), (255, 0, 0), 4)

    cv2.imshow('image', im)
    cv2.waitKey(1)

