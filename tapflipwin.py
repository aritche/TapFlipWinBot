import numpy as np
import cv2
import math
import serial
import time

# GLOBAL VARIABLES
COLOR_BLUE = 0
COLOR_GREEN = 1

# ARDUINO VARIABLES
BAUD_RATE = 9600
SERIAL_PORT = "/dev/cu.usbmodem1411"
#ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
time.sleep(2)

def shouldFlip(curr):
    player = [] # stores centroids of player sides (and colour)
    obstacles = [] # stores centroids of obstacles (and colour)
    original = curr.copy()

    # REMEMBER: H (0,180), S (0,255), V (0,255)
    hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
    kernel = np.ones((19,19), np.uint8)
    colors = [(255,0,0), (0,255,0)]

    # Create HSV ranges for blue
    lower_blue = np.array([90,160,160])
    upper_blue = np.array([120,255,255])

    # Create HSV ranges for green
    lower_green = np.array([41,140,150])
    upper_green = np.array([85,255,255])

    # Eliminate all non-blue 
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue = cv2.bitwise_and(hsv,hsv,mask=mask)
    blue = cv2.cvtColor(blue, cv2.COLOR_HSV2BGR)
    blue = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)
    blue[blue > 0] = 255
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)

    withBox = original.copy()
    c = cv2.connectedComponentsWithStats(blue, 4, cv2.CV_32S)
    for ob in range(1,len(c[2])):
        cent = c[3][ob]
        stat = c[2][ob]
        if stat[cv2.CC_STAT_AREA] > 0:
            withBox = cv2.circle(withBox, (int(cent[0]),int(cent[1])), 3, colors[0], 2)
            # Determine boundaries of component
            (x1,y1) = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
            (x2,y2) = x1+stat[cv2.CC_STAT_WIDTH], y1+stat[cv2.CC_STAT_HEIGHT]
            withBox = cv2.rectangle(withBox, (x1,y1), (x2,y2), colors[0], 2)
            if stat[cv2.CC_STAT_AREA] > 200:
                # Assume obstacle
                withBox = cv2.putText(withBox, "Blue Obstacle", (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1);
                obstacles.append([(int(cent[0]), int(cent[1])),COLOR_BLUE])
            else:
                # Assmume player
                withBox = cv2.putText(withBox, "Blue Player", (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1);
                if (len(player) < 1):
                    player.append([(int(cent[0]),int(cent[1])), COLOR_BLUE])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green = cv2.bitwise_and(hsv,hsv,mask=mask)
    green = cv2.cvtColor(green, cv2.COLOR_HSV2BGR)
    green = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    green[green > 0] = 255
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)

    c = cv2.connectedComponentsWithStats(green, 4, cv2.CV_32S)
    for ob in range(1,len(c[2])):
        cent = c[3][ob]
        stat = c[2][ob]
        if stat[cv2.CC_STAT_AREA] > 0:
            withBox = cv2.circle(withBox, (int(cent[0]),int(cent[1])), 3, colors[1], 2)
            (x1,y1) = stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP]
            (x2,y2) = x1+stat[cv2.CC_STAT_WIDTH], y1+stat[cv2.CC_STAT_HEIGHT]
            withBox = cv2.rectangle(withBox, (x1,y1), (x2,y2), colors[1], 2)
            if stat[cv2.CC_STAT_AREA] > 200:
                withBox = cv2.putText(withBox, "Green Obstacle", (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1);
                obstacles.append([(int(cent[0]), int(cent[1])),COLOR_GREEN])
            else:
                withBox = cv2.putText(withBox, "Green Player", (x1,y1-5), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 1);
                if (len(player) < 2):
                    player.append([(int(cent[0]),int(cent[1])),COLOR_GREEN])


    flip = False 
    if len(player) == 0:
        print("Player not found")
    elif len(player) == 1:
        #print("Do not flip. Player inside obstacle, or mid-way flip animation")
        dummy = True
    elif len(player) == 2:
        result = [-1,'']
        for side in player:
            for o in obstacles:
                dist = math.sqrt((side[0][0]-o[0][0])**2 + (side[0][1]-o[0][1])**2)
                if (dist < result[0] or result[0] < 0):
                    result = [dist, side[1], o[1]]
        if (result[1] != result[2]):
            flip = True
    else:
        print("Too many player sides detected...")

    #cv2.imshow('With Box', withBox)
    #cv2.waitKey(0)

    return flip

def findScreen(im):
    # Apply edge detection, dilate result
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
    pos = [(x1,y1), (x2,y2)]

    return pos

def flip():
    #ser.write(b'f')
    print('flip')
    return 0

def main():
    currEst = False

    # Create window
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)

    # Load sample data
    cap = cv2.VideoCapture('sample.mp4')

    # Get the first frame
    ret, curr = cap.read()

    # Locate the screen in the frame, and crop video
    screen = findScreen(curr)
    curr = curr[screen[0][1]:screen[1][1],screen[0][0]:screen[1][0]]

    # Main loop
    flipHist = []

    # Timeout to prevent rapid flipping
    timeout = 0
    isTimeout = False
    timeoutMax = 3
    while True:
        prevEst = currEst

        # Update curr and prev
        prev = curr
        _, curr = cap.read()
        curr = curr[screen[0][1]:screen[1][1],screen[0][0]:screen[1][0]]

        # Check for timeout
        if isTimeout:
            if (timeout > 0):
                timeout -= 1
                print("Timeout...")
            else:
                isTimeout = False
        else:
            flipHist.append(shouldFlip(curr))
            if (len(flipHist) > 3):
                flipHist.pop(0)
                currEst = max(set(flipHist), key = flipHist.count)
                if (currEst != prevEst):
                    if (currEst == True):
                        flip()
                        isTimeout = True
                        timeout = timeoutMax

        cv2.imshow('window', curr)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
