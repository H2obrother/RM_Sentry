#!/usr/bin/python
import numpy as np
import cv2
import time
import math
import time
import serial
ser = serial.Serial('/dev/ttyUSB0', 256000, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
import numpy as np
import pyrealsense2 as rs

# red
#BLUE = True 
#RED  = False
#MyColor  = BLUE
#blue
BLUE = False
RED  = True
MyColor  = BLUE
record = False

BLUE_exposure = 15
BLUE_angle    = 15
RED_angle     = 15
    
def slope(line):
    #prameter:[[x1,y1],[x2,y2]],(y1-y2)/(x1-x2)
    if(line[0][0] == line[1][0]):
        return 2000
    tan = abs((line[0][1]*1.0 - line[1][1])/(line[0][0] - line[1][0]))
    return tan

def isRectParallel(rect1,rect2):
    r1_angle ,r2_angle = rect1[2],rect2[2]
    r1_width ,r2_width = rect1[1][0] , rect2[1][0]
    r1_height,r2_height = rect1[1][1] , rect2[1][1]
    if r1_width > r1_height:
        temp = r1_width
        r1_width = r1_height
        r1_height = temp
        r1_angle += 90
        if r1_angle > 180:
            r1_angle -= 180
    if r1_angle > 90:
        r1_angle -= 90
    if r1_angle < -90:
        r1_angle += 90

        
    if r2_width > r2_height:
        temp = r2_width
        r2_width = r2_height
        r2_height = temp
        r2_angle += 90
        if r2_angle > 180:
            r2_angle -= 180
    if r2_angle > 90:
        r2_angle -= 90
    if r2_angle < -90:
        r2_angle += 90
    return abs(r2_angle - r1_angle) < allowedAngle
##def isRectParallel(rect1,rect2)
##    return abs(rect1[2] - rect2[2]) < 12
def isSameArea(rect1,rect2):
    perimeter = (rect1[1][0] + rect1[1][1] + rect2[1][0] + rect2[1][1])
    sides1 = np.array([rect1[1][0] , rect1[1][1]])
    sides2 = np.array([rect2[1][0] , rect2[1][1]])
    return np.abs(sides1[0]*sides1[1] - sides2[0]*sides2[1]) < 600
def isCenterNearby(rect1,rect2):
    perimeter = (rect1[1][0] + rect1[1][1] + rect2[1][0] + rect2[1][1])
    distance = math.sqrt((rect1[0][0]-rect2[0][0])**2 + (rect1[0][1]-rect2[0][1])**2)
    return  distance < perimeter*2.5  and distance > perimeter*0.5
    #return  distance < 200  and distance > 15 
def balckPointRate(frame,p1,p2):
    Vp1p2 = [p2[0]-p1[0]]


def armor_width(rect1,rect2):
    aw = np.array([rect1[1][0] , rect1[1][1] , rect2[1][0] , rect2[1][1]])
    return aw.max()
def DTcolor(frame,box):
    numFitPoint = 0
    maxCol ,maxRow ,_= frame.shape
    for point in box:
        x = point[0]
        y = point[1]
        if y >= maxCol:
            y = maxCol - 1
        if x >= maxRow:
            x = maxRow - 1
        if frame[y][x][0] > frame[y][x][2]:#if dengTiao is BLUE
            numFitPoint +=1
    return (BLUE if numFitPoint >2 else RED)
def adjustRect(rect):
    
    rect[0],rect[1] = list(rect[0]),list(rect[1])
    if rect[1][0] > rect[1][1]:
        temp = rect[1][0]
        rect[1][0] = rect[1][1]
        rect[1][1] = temp
        rect[2] += 90
        if rect[2] > 180:
            rect[2] -= 180
    if rect[2] > 90:
        rect[2] -= 90
    if rect[2] < -90:
        rect[2] += 90
def recvMsg():
    ch = ser.read()
    while(ch != 'r' and ch != 'b' and ch != 'q'):
        ser.flushInput()
        ch = ser.read()
    msg = [0,0,0,0,0]
    msg[0] = ch
    print(ch)
    msg[1:] = ser.read(4)
    return msg
def findMidPoint(frame):
    filteredCont = []
    approxCont = []
    boxes = []
    rects = []
    centerSlope = 1000
    finalPair = []
    midPoint = None
    t1 = cv2.getTickCount()
    width_pixel = 0
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,thresold_val,255,cv2.THRESH_BINARY)
    
   #cv2.imshow('thresold',thresh)
    temp,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 

    # Display the resulting frame
    #cv2.imshow('thresold',thresh)
    numCont = len(contours)
    if numCont == 0:
        return None,0
    for cnt in contours:
        contArea = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        #print('contArea > 60',contArea > 20  , '  contArea < 25000:',contArea < 25000 )
        boxTemp = cv2.boxPoints(rect)
        boxTemp = np.int0(boxTemp)
        if  contArea > 6  and contArea < 10000 and MyColor!=DTcolor(frame,boxTemp) :
            filteredCont.append(cnt)
            rects.append(rect)
    if len(rects) == 0:
        return None,0

    for i in range(len(rects)):
        direction1 = rects[i][1][1] > rects[i][1][0]
        for rect2 in rects[i:]:
            direction2 = rect2[1][1] > rect2[1][0]
            print('isRectParallel:',isRectParallel(rects[i],rect2) ,'  isSameArea:' ,isSameArea(rects[i],rect2) , '  isCenterNearby:',isCenterNearby(rects[i],rect2))
            if isRectParallel(rects[i],rect2) and isSameArea(rects[i],rect2) and isCenterNearby(rects[i],rect2) :
                line = [rects[i][0],rect2[0]]
                cv2.line(frame,(int(math.floor(rects[i][0][0])),int(math.floor(rects[i][0][1]))),(int(math.floor(rect2[0][0])),int(math.floor(rect2[0][1]))),(0,255,0),2)

                box = cv2.boxPoints(rects[i])
                box = np.int0(box)
                boxes.append(box)
                box = cv2.boxPoints(rect2)
                box = np.int0(box)
                boxes.append(box)
                tempSlope = slope([[int(math.floor(rects[i][0][0])),int(math.floor(rects[i][0][1]))],[int(math.floor(rect2[0][0])),int(math.floor(rect2[0][1]))]])
                if tempSlope > 0.6:
                    continue
                if centerSlope > tempSlope: 
                    centerSlope = tempSlope
                    box1 = cv2.boxPoints(rects[i])
                    box1 = np.int0(box1)
                    midPoint= (int((math.floor(rects[i][0][0])+math.floor(rect2[0][0]))/2.0),int((math.floor(rects[i][0][1])+math.floor(rect2[0][1]))/2.0))
                    box2 = cv2.boxPoints(rect2)
                    box2 = np.int0(box2)
                    finalPair=[box1,box2]
                    finalReact = [rects[i],rect2]
        if midPoint is not None:
            width_pixel = armor_width(finalReact[0],finalReact[1])
        
        cv2.drawContours(frame,filteredCont,-1,(0,255,0),3)
        cv2.circle(frame,midPoint,10,(255,255,0),5)
        
        return midPoint,width_pixel
if __name__ == '__main__':
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 60)
    pipeline_profile = pipeline.start(config)
    device = pipeline_profile.get_device()
    color_sensor = device.query_sensors()[1]
    exposure = color_sensor.get_option(rs.option.exposure)
    gain = color_sensor.get_option(rs.option.gain)
    color_sensor.set_option(rs.option.exposure, 10)
    color_sensor.set_option(rs.option.gain, 0)
    prev = time.time()
    buff_len  = 6
    prev_t1 = np.ones(buff_len)
    prev_t2 = np.ones(buff_len)
    prev_x = np.ones(buff_len)
    prev_y = np.ones(buff_len)
    if record == True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(time.ctime(time.time())+'.avi',fourcc, 20.0, (640,360))
    prev = time.time()
    i = 0
    j = 0
    q_size = 10
    fps_l = np.zeros(q_size)
    focal_length = 7
    armor_atrual_length = 65
    while True:
        #msg = recvMsg()
        command = 'r'
        
        
        if MyColor == BLUE:
            thresold_val = 55
        if MyColor == RED:
            thresold_val = 50
        if MyColor == BLUE:
            allowedAngle = BLUE_angle
        else:
            allowedAngle = RED_angle
        if MyColor == BLUE:
            exposure_val = 10
        if MyColor == RED:
            exposure_val = 15
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())
        if record == True:
            out.write(frame)
        target,width_pixel = findMidPoint(frame)
        print(target)
        cv2.imshow('frame',frame)
        fps  = 1/(time.time() - prev)

        print(fps)
        fps_l[i % q_size]  = fps
        values = bytearray([0xA5,   0,   0,   0   ,0,0,0,0,0,0,0,0])
        values[8],values[9] =(int(fps) & 0xff00)>>8, (int(fps) &0xff)
        if (target != None)  :
            prev_t1[j %buff_len] = time.time()
            prev_t2[j %buff_len] = time.time()
            prev_x[j %buff_len] = target[0]
            prev_y[j %buff_len] = target[1]
            j += 1
            distance = (armor_atrual_length * focal_length)/width_pixel
            distance = distance *10
            
            eastimate_distance = 157.4 -26.04 * np.cos(distance * 0.0146) -73.64 * np.sin(distance * 0.0146) -8.831 * np.cos(2 * distance * 0.0146) -15.48 * np.sin(2 * distance * 0.0146)
            print(eastimate_distance)
            distance_int16 = eastimate_distance.astype(int)
            values[10],values[11] = ((distance_int16& 0xff00)>>8, (distance_int16& 0xff))
            prev = time.time()
            values[5] = 1
            values[0],values[1],values[2],values[3],values[4],values[6] =0x55,    (target[0] & 0xff00 )>>8,    target[0]&0xff, (target[1]& 0xff00)>>8,target[1]& 0xff,  0
            if fps_l.mean() > 10 :
                values[6] = 1
        if (target == None):
            t2x = np.poly1d(np.polyfit(prev_t1, prev_x, 1))
            t2y = np.poly1d(np.polyfit(prev_t2, prev_y, 1))
            t_now = time.time()
            pred_x = t2x(t_now)
            pred_y = t2y(t_now)
            values[0],values[1],values[2],values[3],values[4],values[6] =0x55,    (pred_x.astype(int) & 0xff00 )>>8,    pred_x.astype(int)&0xff, (pred_y.astype(int) & 0xff00)>>8,pred_y.astype(int)& 0xff,  0
        ser.write(values)
    
        #command = 'r'
        print(command)
        
        if command == 'q':
            break
        if command == 'r':
            MyColor = BLUE
        if command == 'b':
            MyColor = RED
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
        i += 1

if record == True:
    out.release()
cv2.destroyAllWindows()
pipeline.stop()
