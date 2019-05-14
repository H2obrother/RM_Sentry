#!/usr/bin/python
import numpy as np
import cv2
import time
import math
import time
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
import numpy as np
# red
#BLUE = True 
#RED  = False
#MyColor  = BLUE
#blue
BLUE = False
RED  = True
MyColor  = RED

record = False
if MyColor == BLUE:
    thresold_val = 30
if MyColor == RED:
    thresold_val = 20

def slope(line):
    #prameter:[[x1,y1],[x2,y2]],(y1-y2)/(x1-x2)
    if(line[0][0] == line[1][0]):
        return 2000
    tan = abs((line[0][1]*1.0 - line[1][1])/(line[0][0] - line[1][0]))
    return tan

def isRectParallel(rect1,rect2):
    return abs(rect1[2] - rect2[2]) < 12
def isSameArea(rect1,rect2):
    perimeter = (rect1[1][0] + rect1[1][1] + rect2[1][0] + rect2[1][1]) 
    return abs(rect1[1][0] * rect1[1][1] -  rect2[1][0] * rect2[1][1]) < 500
def isCenterNearby(rect1,rect2):
    perimeter = (rect1[1][0] + rect1[1][1] + rect2[1][0] + rect2[1][1])
    distance = math.sqrt((rect1[0][0]-rect2[0][0])**2 + (rect1[0][1]-rect2[0][1])**2)
    return  distance < perimeter*1.5  and distance > perimeter*0.5
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
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([75,195,185])
    upper_blue = np.array([130,255,255])
    lower_red_1 = np.array([145,135,148])
    upper_red_1 = np.array([180,255,255])
    lower_red_2 = np.array([0,135,148])
    upper_red_2 = np.array([13,255,255])
    # Threshold the HSV image to get only blue colors
    thresh = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image

    cv2.imshow('thresold',thresh)
    temp,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

    # Display the resulting frame
    cv2.imshow('thresold',thresh)
    numCont = len(contours)
    if numCont == 0:
        return None,0
    for cnt in contours:
        contArea = cv2.contourArea(cnt)
        rect = cv2.minAreaRect(cnt)
        print('contArea > 60',contArea > 50  , '  contArea < 25000:',contArea < 25000 , '  rect[1][1] > rect [1][0]:',rect[1][1] > rect [1][0])
        boxTemp = cv2.boxPoints(rect)
        boxTemp = np.int0(boxTemp)
        if  contArea > 50  and contArea < 10000 and MyColor!=DTcolor(frame,boxTemp):
            filteredCont.append(cnt)
            rects.append(rect)
    if len(rects) == 0:
        return None,0
    for i in range(len(rects)):
        direction1 = rects[i][1][1] > rects[i][1][0]
        for rect2 in rects[i+1:]:
            direction2 = rect2[1][1] > rect2[1][0]
            #print('isRectParallel:',isRectParallel(rects[i],rect2) ,'  isSameArea:' ,isSameArea(rects[i],rect2) , '  isCenterNearby:',isCenterNearby(rects[i],rect2))
            if isRectParallel(rects[i],rect2) and isSameArea(rects[i],rect2) and isCenterNearby(rects[i],rect2) and direction1==direction2:
                line = [rects[i][0],rect2[0]]
                #cv2.line(frame,(int(math.floor(rects[i][0][0])),int(math.floor(rects[i][0][1]))),(int(math.floor(rect2[0][0])),int(math.floor(rect2[0][1]))),(0,255,0),2)

                box = cv2.boxPoints(rects[i])
                box = np.int0(box)
                boxes.append(box)
                box = cv2.boxPoints(rect2)
                box = np.int0(box)
                boxes.append(box)
                tempSlope = slope([[int(math.floor(rects[i][0][0])),int(math.floor(rects[i][0][1]))],[int(math.floor(rect2[0][0])),int(math.floor(rect2[0][1]))]])
                if tempSlope > 2:
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
    
    cap = cv2.VideoCapture(0)
    prev = time.time()
    if record == True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(time.ctime(time.time())+'.avi',fourcc, 20.0, (640,360))
    prev = time.time()
    i = 0
    q_size = 20
    fps_l = np.zeros(q_size)
    focal_length = 7
    armor_atrual_length = 65
    while(cap.isOpened()):
        
        ret, frame = cap.read()
        if record == True:
            out.write(frame)
        target,width_pixel = findMidPoint(frame)
        cv2.imshow('frame',frame)
        fps  = 1/(time.time() - prev)

        #print(fps)
        fps_l[i % q_size]  = fps
        values = bytearray([0xA5,   0,   0,   0   ,0,0,0,0,0,0,0,0])
        values[8],values[9] =(int(fps) & 0xff00)>>8, (int(fps) &0xff)
        if (target != None)  :
            distance = (armor_atrual_length * focal_length)/width_pixel
            #if distance > 26 :
                #distance -= 4
            #if distance < 26 :
                #distance -= 2
            distance = distance *10
            
            eastimate_distance = 157.4 -26.04 * np.cos(distance * 0.0146) -73.64 * np.sin(distance * 0.0146) -8.831 * np.cos(2 * distance * 0.0146) -15.48 * np.sin(2 * distance * 0.0146)
            print(eastimate_distance)
            distance_int16 = eastimate_distance.astype(int)
            values[10],values[11] = ((distance_int16& 0xff00)>>8, (distance_int16& 0xff))
            prev = time.time()
            values[5] = 1
            values[0],values[1],values[2],values[3],values[4],values[6] =0x55,    (target[0] & 0xff00 )>>8,    target[0]&0xff, (target[1]& 0xff00)>>8,target[1]& 0xff,  0
            if fps_l.mean() > 30 :
                values[6] = 1
            
        
        ser.write(values)
        esc = ser.read()
        if esc == 'q':
            break
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
        i += 1
cap.release()
if record == True:
    out.release()
cv2.destroyAllWindows()
