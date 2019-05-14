import numpy as np
import cv2
import time
import math
import time
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
import numpy as np
while True:
    command = 'q'
    ser.write(command)
    print(ser.read(1))
