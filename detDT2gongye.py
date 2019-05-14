#!/usr/bin/python
import numpy as np
import cv2
import time
import math
import time
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
import numpy as np
import mvsdk
# red
#BLUE = True 
#RED  = False
#MyColor  = BLUE
#blue
BLUE = False
RED  = True
MyColor  = BLUE
record = True
if MyColor == BLUE:
    thresold_val = 50
    gain_b= 0
    gain_r= 100
if MyColor == RED:
    thresold_val = 50
    gain_b= 100
    gain_r= 0
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
    return abs(r2_angle - r1_angle) < 15
def isSameArea(rect1,rect2):
    perimeter = (rect1[1][0] + rect1[1][1] + rect2[1][0] + rect2[1][1]) 
    return abs(rect1[1][0] * rect1[1][1] -  rect2[1][0] * rect2[1][1]) < 900
def isCenterNearby(rect1,rect2):
    perimeter = np.max([rect1[1][0] , rect1[1][1] , rect2[1][0] , rect2[1][1]])
    distance = math.sqrt((rect1[0][0]-rect2[0][0])**2 + (rect1[0][1]-rect2[0][1])**2)
    return  distance < perimeter*5  and distance > perimeter*0.3
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
        #print('contArea > 30',contArea > 30  , '  contArea < 25000:',contArea < 10000 , '  rect[1][1] > rect [1][0]:',rect[1][1] > rect [1][0])
        boxTemp = cv2.boxPoints(rect)
        boxTemp = np.int0(boxTemp)
        if  contArea > 30  and contArea < 10000 and MyColor!=DTcolor(frame,boxTemp):
            filteredCont.append(cnt)
            rects.append(rect)
    if len(rects) == 0:
        return None,0

    for i in range(len(rects)):
        for rect2 in rects[i+1:]:
            #print('isRectParallel:',isRectParallel(rects[i],rect2) ,'  isSameArea:' ,isSameArea(rects[i],rect2) , '  isCenterNearby:',isCenterNearby(rects[i],rect2))
            if isRectParallel(rects[i],rect2) and isSameArea(rects[i],rect2) and isCenterNearby(rects[i],rect2):
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
        
        #cv2.drawContours(frame,filteredCont,-1,(0,255,0),3)
        cv2.circle(frame,midPoint,10,(255,255,0),5)
        
        return midPoint,width_pixel
if __name__ == '__main__':
    
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
    i = 0
    DevInfo = DevList[i]
    print(DevInfo)

    hCamera = 0
    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        print("CameraInit Failed({}): {}".format(e.error_code, e.message) )

    cap = mvsdk.CameraGetCapability(hCamera)

    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)


    if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)

    mvsdk.CameraSetTriggerMode(hCamera, 0)
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 5 * 1000)
    mvsdk.CameraSetGain(hCamera, gain_r, 0, gain_b)
    mvsdk.CameraPlay(hCamera)

    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    prev = time.time()
    if record == True:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(time.ctime(time.time())+'.avi',fourcc, 20.0, (640,360))
    prev = time.time()
    i = 0
    q_size = 10
    fps_l = np.zeros(q_size)
    focal_length = 8
    armor_atrual_length = 65
    while  ((cv2.waitKey(1) & 0xFF) != ord('q')):
        try:
                    pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
                    mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
                    mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
                    frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )

                    frame_r = frame[:,:,0].copy()
                    frame_b = frame[:,:,2].copy()
                    frame[:,:,0] = frame_b
                    frame[:,:,2] = frame_r     
                    
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )
        if record == True:
            out.write(frame)
        target,width_pixel = findMidPoint(frame)
        cv2.imshow('frame',frame)
        #print(target)
        fps  = 1/(time.time() - prev)

        #print(fps)
        fps_l[i % q_size]  = fps
        values = bytearray([0xA5,   0,   0,   0   ,0,0,0,0,0,0,0,0])
        values[8],values[9] =(int(fps) & 0xff00)>>8, (int(fps) &0xff)
        if (target != None)  :
            distance = (armor_atrual_length * focal_length)/width_pixel
            distance = distance *10
            print(distance)
            eastimate_distance = -691.5 +286.5 * np.cos(distance * 0.009976) + 1194 * np.sin(distance * 0.009976) +232.6 * np.cos(2 * distance * 0.009976) - 217.5 * np.sin(2 * distance * 0.009976)
            #print(eastimate_distance)
            distance_int16 = eastimate_distance.astype(int)
            values[10],values[11] = ((distance_int16& 0xff00)>>8, (distance_int16& 0xff))
            prev = time.time()
            values[5] = 1
            values[0],values[1],values[2],values[3],values[4],values[6] =0x55,    (target[0] & 0xff00 )>>8,    target[0]&0xff, (target[1]& 0xff00)>>8,target[1]& 0xff,  0
            if fps_l.mean() > 10 :
                values[6] = 1
            
        
        ser.write(values)
        #command = ser.read()

        #command = 'r'
      #  if command == 'q':
        #    break
        #if command == 'r':
        #    MyColor = BLUE
       # if command == 'b':
        #    MyColor = RED
        i += 1

if record == True:
    out.release()

mvsdk.CameraUnInit(hCamera)
mvsdk.CameraAlignFree(pFrameBuffer)
cv2.destroyAllWindows()
