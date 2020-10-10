"""
Detects circles in a video.
"""

from time import sleep
import argparse

import cv2
import numpy as np


def findCircles(sourceImage, drawCircles=True):
    """Finds circles in an image, then optionally draws them to said image.

    Args:
        sourceImage (cv2.Mat): The image to analyze.
        drawCircles (bool, optional): Draw the detected circles. Defaults to True.

    Returns:
        circles, copiedImage: The detected circles and the optionally drawn to 
            source image.
    """
    src = sourceImage.copy()

    # Requirement 2
    grey = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(grey, (9, 9), sigmaX=2, sigmaY=2)

    # Requirement 3
    # Note: I changed param2 to 32 due to the videos not having the best centers.
    #       There's a lot of reflection and motion on the balls, so this little bit
    #       helps with blocking out poor detections.
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, frame.shape[1]/16, 
        param1=100, param2=32)

    # Requirement 4
    if circles is not None and drawCircles:
        circles = np.uint16(np.around(circles))
        for n in circles[0, :]:
            # Draw circle center
            center = (n[0], n[1])
            cv2.circle(src, center, 1, (0, 255, 255), 3)

            # Draw outlining circle
            radius = n[2]
            cv2.circle(src, center, radius, (0, 255, 0), 3)
    
    return circles, src

def main():
    """
    The main function for Part 1.
    """
    # Requirement 1
    parser = argparse.ArgumentParser(description="Find each circle in each frame" + \
        "then draw it to the original frame.")
    parser.add_argument("infile", nargs=1, type=str, default=None, 
        help="The file to read the video from.")
    parser.add_argument("--fps", nargs="?", type=np.uint16, default=30,
        help="The FPS to display the processed video at.")
    args = parser.parse_args()

    # Grab data
    capture = cv2.VideoCapture(str(args.infile[0]))
    showVideo = capture.isOpened()
    if not showVideo:
        print("Error opening video file at \"" + str(args.infile[0]) + "\".")
    
    # Process and display data
    while capture.isOpened():
        ret, src = capture.read()
        if not ret: break

        _, resultImg = findCircles(src, drawCircles=True)

        # Display
        cv2.imshow("part1", resultImg)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        sleep(1/args.fps)
    capture.release()
    
    if showVideo:
        cv2.waitKey(0)
        cv2.destroyWindow("part1")

    return

if __name__ == "__main__":
    main()
