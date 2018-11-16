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
INIT_PERIOD = 300


"""
    connectedComps: output after running image through connected comps.
    color:          color to label found objects
    thresh:         threshold size (larger = obstacle, smaller = player)
"""
def findObjects(connectedComps, color, thresh):
    obstacles = []
    player = []

    numObjects = len(connectedComps[2])
    for ob in range(1,numObjects):
        cent = connectedComps[3][ob] # centroid of object
        stat = connectedComps[2][ob] # stats for object
        if stat[cv2.CC_STAT_AREA] > 0:
            if stat[cv2.CC_STAT_AREA] > thresh: # Assume obstacle
                obstacles.append([(int(cent[0]), int(cent[1])),color])
            else: # Assume player
                # Final found player object is assumed to be the actual player
                # All others are assumed to be noise
                centre = (int(cent[0]), int(cent[1]))
                if (len(player) == 0):
                    player.append([centre, color])
                else:
                    player[0] = [centre, color]

    return player, obstacles

def shouldFlip(curr, frameNum, vis=False):
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

    # Eliminate all non-specified color
    blue = cv2.inRange(hsv, lower_blue, upper_blue)
    green = cv2.inRange(hsv, lower_green, upper_green)

    # Reduce size and find blue players and obstacles
    blue = cv2.resize(blue, (0,0), fx=0.15, fy=0.15)
    c = cv2.connectedComponentsWithStats(blue, 4, cv2.CV_32S)
    bluePlayer, blueObstacles = findObjects(c,COLOR_BLUE,4)

    # Reduce size and find green players and obstacles
    green = cv2.resize(green, (0,0), fx=0.15, fy=0.15)
    c = cv2.connectedComponentsWithStats(green, 4, cv2.CV_32S)
    greenPlayer, greenObstacles = findObjects(c,COLOR_GREEN,4)

    # Merge players and obstacles
    """
    Check that player sides are at a relatively
    similar y-level (assuming phone is landscape
    from camera perspective)
    """
    disposePlayers = False
    if (len(bluePlayer) != 0 and len(greenPlayer) != 0):
        blueY = bluePlayer[0][0][1]
        greenY = greenPlayer[0][0][1]
        if abs(blueY - greenY) > 1:
            disposePlayers = True

    if (not disposePlayers):
        player = bluePlayer
        player = player + [p for p in greenPlayer]

        obstacles = blueObstacles
        obstacles = obstacles + [o for o in greenObstacles]
    else:
        player = []
        obstacles = []

    if (vis):
        im = curr.copy()
        for item in player:
            col = (0,0,0)
            if item[-1] == COLOR_GREEN:
                col = (0,255,0)
            else:
                col = (255,0,0)
            im = cv2.circle(im, (int(item[0][0]/0.15), int(item[0][1]/0.15)), 3, col)
        for item in obstacles:
            col = (0,0,0)
            if item[-1] == COLOR_GREEN:
                col = (0,255,0)
            else:
                col = (255,0,0)
            #im = cv2.circle(im, (int(item[0][0]/0.15), int(item[0][1]/0.15)), 6, col)
        cv2.imshow('Disp', im)
        cv2.waitKey(1)

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

    return flip

# Find the phone's screen in the large area seen by the camera
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

# Send a flip action request to the arduino
def flip():
    ser.write(b'f')
    print('flip')
    return 0

def main():
    # Create window
    #cv2.namedWindow('window', cv2.WINDOW_NORMAL)

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

        # Update the current estimated best action
        prevEst = currEst

        # Init frame
        rawCapture.truncate()
        rawCapture.seek(0)

        # Update curr and prev frames
        prev = curr
        curr = frame.array
        curr = curr[screen[0][1]:screen[1][1],screen[0][0]:screen[1][0]]

        # Get the next action
        flipHist.append(shouldFlip(curr, frameNum))
        
        # Get the majority vote over the past 3 suggested actions
        # Only execute the action if the action has changed from the
        #   previously suggested action
        if (len(flipHist) > 3):
            flipHist.pop(0)
            currEst = max(set(flipHist), key = flipHist.count)
            if (currEst != prevEst):
                if (currEst == True):
                    flip()

        # Count the frames being seen per second
        # Useful for debugging/optimisation benchmarking
        intervalCounter += 1
        if (time.time() - startTime) > interval:
            print(int(intervalCounter / (time.time() - startTime)))
            intervalCounter = 0
            startTime = time.time()

        frameNum += 1


    cv2.destroyAllWindows()

main()
