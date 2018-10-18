import numpy as np
import cv2
import math
import serial
import time
from picamera.array import PiRGBArray
from picamera import PiCamera

# GLOBAL VARIABLES
COLOR_BLUE = 0
COLOR_GREEN = 1

# ARDUINO VARIABLES
BAUD_RATE = 9600
#SERIAL_PORT = "/dev/cu.usbmodem1411"
SERIAL_PORT = "/dev/ttyACM0"
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

# CAMERA VARIABLES
camera = PiCamera()
camera.resolution = (160, 120)
camera.framerate = 60
rawCapture = PiRGBArray(camera, size=(160,120))
time.sleep(0.1)

# OPENCV VARIABLES
INIT_PERIOD = 100

def shouldFlip(curr, frameNum):
    player = [] # stores centroids of player sides (and colour)
    obstacles = [] # stores centroids of obstacles (and colour)

    # REMEMBER: H (0,180), S (0,255), V (0,255)
    hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((19,19), np.uint8)
    colors = [(255,0,0), (0,255,0)]

    # Create HSV ranges for blue
    lower_blue = np.array([85,100,135])
    upper_blue = np.array([120,255,255])

    # Create HSV ranges for green
    lower_green = np.array([21,100,145])
    upper_green = np.array([85,255,255])

    # Eliminate all non-blue 
    blue = cv2.inRange(hsv, lower_blue, upper_blue)
    #blue = cv2.bitwise_and(hsv,hsv,mask=mask)
    #blue = blue[:,:,2]
    #blue[blue > 0] = 255

    if frameNum < INIT_PERIOD:
        withBox = curr.copy()
    #withBox[withBox != 0] = 0
    blue = cv2.resize(blue, (0,0), fx=0.15, fy=0.15)
    c = cv2.connectedComponentsWithStats(blue, 4, cv2.CV_32S)
    for ob in range(1,len(c[2])):
        cent = c[3][ob]
        stat = c[2][ob]
        if stat[cv2.CC_STAT_AREA] > 0:
            # Determine boundaries of component
            (x1,y1) = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
            (x2,y2) = x1+stat[cv2.CC_STAT_WIDTH], y1+stat[cv2.CC_STAT_HEIGHT]
            #withBox = cv2.rectangle(withBox, (x1,y1), (x2,y2), colors[0], 2)
            if stat[cv2.CC_STAT_AREA] > 4:
                # Assume obstacle
                #withBox = cv2.putText(withBox, "Blue Obstacle", (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1);
                #withBox = cv2.rectangle(withBox, (int(x1/0.15),int(y1/0.15)), (int(x2/0.15),int(y2/0.15)), colors[0], 2)
                obstacles.append([(int(cent[0]), int(cent[1])),COLOR_BLUE])
            else:
                # Assmume player
                #withBox = cv2.putText(withBox, "Blue Player", (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1);
                #withBox = cv2.circle(withBox, (int(cent[0]),int(cent[1])), 3, colors[0], 1)
                #withBox = cv2.rectangle(withBox, (x1,y1), (x2,y2), colors[0], 2)
                if frameNum < INIT_PERIOD:
                    withBox = cv2.rectangle(withBox, (int(x1/0.15),int(y1/0.15)), (int(x2/0.15),int(y2/0.15)), colors[0], 2)
                if (len(player) == 0):
                    player.append([(int(cent[0]),int(cent[1])), COLOR_BLUE])
                else:
                    player[0] = [(int(cent[0]),int(cent[1])), COLOR_BLUE]

    green = cv2.inRange(hsv, lower_green, upper_green)
    #cv2.imshow('mask',mask)
    #cv2.waitKey(1)
    #green = cv2.bitwise_and(hsv,hsv,mask=mask)
    #green = green[:,:,2]
    #green[green > 0] = 255

    green = cv2.resize(green, (0,0), fx=0.15, fy=0.15)
    c = cv2.connectedComponentsWithStats(green, 4, cv2.CV_32S)
    for ob in range(1,len(c[2])):
        cent = c[3][ob]
        stat = c[2][ob]
        if stat[cv2.CC_STAT_AREA] > 0:
            (x1,y1) = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
            (x2,y2) = x1+stat[cv2.CC_STAT_WIDTH], y1+stat[cv2.CC_STAT_HEIGHT]
            if stat[cv2.CC_STAT_AREA] > 4:
                #withBox = cv2.rectangle(withBox, (int(x1/0.15),int(y1/0.15)), (int(x2/0.15),int(y2/0.15)), colors[1], 2)
                #withBox = cv2.putText(withBox, "Green Obstacle", (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1);
                obstacles.append([(int(cent[0]), int(cent[1])),COLOR_GREEN])
            else:
                if frameNum < INIT_PERIOD:
                    withBox = cv2.rectangle(withBox, (int(x1/0.15),int(y1/0.15)), (int(x2/0.15),int(y2/0.15)), colors[1], 2)
                #withBox = cv2.putText(withBox, "Green Player", (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1);
                #withBox = cv2.circle(withBox, (int(cent[0]),int(cent[1])), 3, colors[1], 1)
                #withBox = cv2.rectangle(withBox, (x1,y1), (x2,y2), colors[1], 2)
                if (len(player) < 2):
                    player.append([(int(cent[0]),int(cent[1])),COLOR_GREEN])
                else:
                    player[1] = [(int(cent[0]),int(cent[1])),COLOR_GREEN]

    flip = False 
    if len(player) == 0:
        #print("Player not found")
        dummy = True
    elif len(player) == 1:
        #print("1 player found")
        dummy = True
    elif len(player) == 2:
        # check colours are different
        result = [-1,-1,-1]
        if (player[0][-1] != player[1][-1]):
            #print("2 players found")
            for side in player:
                for o in obstacles:
                    dist = math.sqrt((side[0][0]-o[0][0])**2 + (side[0][1]-o[0][1])**2)
                    if (dist < result[0] or result[0] < 0):
                        result = [dist, side[-1], o[-1]]
        else:
            #print("Wrong player colours...")
            dummy = True
        if (result[1] != result[2]):
            flip = True
    else:
        #print("Too many player sides detected...")
        dummy = True

    if frameNum < INIT_PERIOD:
        cv2.imshow('With Box', withBox)
        cv2.waitKey(1)

    #cv2.imshow('Blue', blue)
    #cv2.waitKey(1)

    #cv2.imshow('Green', green)
    #cv2.waitKey(1)
    return flip

def findScreen(im):
    # Apply edge detection, dilate result
    kernel = np.ones((2,2), np.uint16)
    im = cv2.filter2D(im,-1,kernel)
    im = cv2.Canny(im,100,200)
    kernel = np.ones((20,20), np.uint16)
    im = cv2.dilate(im, kernel, iterations=1)

    # Find connected components
    _, _, stats, centroids = cv2.connectedComponentsWithStats(im, 4, cv2.CV_32S)
    sortedBySizeIndex = np.argsort(stats[:,-1])

    # Get coordinate of largest component
    pos = ()
    obj = stats[sortedBySizeIndex[-2]]
    (x1,y1)=(obj[cv2.CC_STAT_LEFT], obj[cv2.CC_STAT_TOP])
    (x2,y2)=(x1+obj[cv2.CC_STAT_WIDTH], y1+obj[cv2.CC_STAT_HEIGHT])
    width = abs(x1-x2)
    height = abs(y1-y2)
    pos = [(x1+int(width*0.3),y1+int(height*0.15)), (x2-int(width*0.5),y2-int(height*0.15))]

    #return pos
    return [(0+int(160*0.3),0+int(120*0.15)), (160-int(width*0.5),120-int(120*0.15))]

def flip():
    #ser.write(b'f')
    ser.write(b'f')
    print('flip')
    return 0

def findPlayer(curr):
    gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    gray[gray > 170] = 0
    gray[gray < 100] = 0
    gray[gray > 0] = 255

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    #result = cv2.bitwise_and(curr,curr,mask=mask)

    mask = cv2.resize(mask, (0,0), fx=0.10, fy=0.10)
    c = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    withBox = curr.copy()
    for ob in range(1,len(c[2])):
        cent = c[3][ob]
        stat = c[2][ob]
        if stat[cv2.CC_STAT_AREA] > 0:
            # Determine boundaries of component
            (x1,y1) = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
            (x2,y2) = x1+stat[cv2.CC_STAT_WIDTH], y1+stat[cv2.CC_STAT_HEIGHT]
            if stat[cv2.CC_STAT_AREA] > 50:
                withBox = cv2.rectangle(withBox, (x1,y1), (x2,y2), (255,0,0), 2)
                # Assume obstacle
                #withBox = cv2.putText(withBox, "Blue Obstacle", (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1);
                #withBox = cv2.rectangle(withBox, (x1,y1), (x2,y2), colors[0], 2)
                #obstacles.append([(int(cent[0]), int(cent[1])),COLOR_BLUE])
            #else:
                # Assmume player
                #withBox = cv2.putText(withBox, "Blue Player", (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1);
                #withBox = cv2.circle(withBox, (int(cent[0]),int(cent[1])), 3, colors[0], 1)
                #withBox = cv2.rectangle(withBox, (x1,y1), (x2,y2), colors[0], 2)
                #if (len(player) == 0):
                #    player.append([(int(cent[0]),int(cent[1])), COLOR_BLUE])
                #else:
                #    player[0] = [(int(cent[0]),int(cent[1])), COLOR_BLUE]

    #cv2.imshow('prev', withBox)
    #cv2.waitKey(1)
    

def main():
    # Create window
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)

    # FPS counting parameters
    interval = 1
    intervalCounter = 0
    startTime = time.time()

    # Flipping parameters
    currEst = False
    flipHist = []

    # Video parameters
    frameNum = 0
    screen = []
    curr = []

    # Main loop
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # Find screen on first frame
        if (frameNum == 0):
            curr = frame.array
            screen = findScreen(curr)

        prevEst = currEst

        # Init frame
        rawCapture.truncate()
        rawCapture.seek(0)

        # Update curr and prev
        prev = curr
        curr = frame.array
        curr = curr[screen[0][1]:screen[1][1],screen[0][0]:screen[1][0]]
        #curr = cv2.resize(curr, (0,0), fx=0.10, fy=0.10)

        flipHist.append(shouldFlip(curr, frameNum))
        if (len(flipHist) > 3):
            flipHist.pop(0)
            currEst = max(set(flipHist), key = flipHist.count)
            if (currEst != prevEst):
                if (currEst == True):
                    flip()

        if (frameNum < INIT_PERIOD):
            cv2.imshow('window', curr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        intervalCounter += 1
        if (time.time() - startTime) > interval:
            print(int(intervalCounter / (time.time() - startTime)))
            intervalCounter = 0
            startTime = time.time()

        frameNum += 1


    cv2.destroyAllWindows()

main()
