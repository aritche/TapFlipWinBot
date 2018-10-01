import numpy as np
import cv2

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

def main():
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
    while True:
        # Update curr and prev
        prev = curr
        _, curr = cap.read()
        curr = curr[screen[0][1]:screen[1][1],screen[0][0]:screen[1][0]]

        cv2.imshow('window', curr)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
