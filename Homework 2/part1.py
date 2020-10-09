"""
Detects circles in a video.
"""

import os
from time import sleep
import argparse

import cv2
import numpy as np


def main():
    print(os.path.dirname(os.path.realpath(__file__)))

    # Requirement 1
    parser = argparse.ArgumentParser(description="Find each circle in each frame" + \
        "then draw it to the original frame.")
    parser.add_argument("infile", nargs=1, type=str, default=None, 
        help="The file to read the video from.")
    parser.add_argument("--fps", nargs="?", type=np.uint16, default=24,
        help="The FPS to display the processed video at.")
    args = parser.parse_args()

    # Grab data
    capture = cv2.VideoCapture(str(args.infile))
    if not capture.isOpened():
        print("Error opening video file.")
    
    # Process and display data
    while capture.isOpened():
        ret, src = capture.read()
        if not ret: continue
    
        # Requirement 2
        grey = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(grey, (9, 9), sigmaX=2, sigmaY=2)

        # Requirement 3
        rows = frame.shape[0]
        circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, rows)

        # Requirement 4
        if circles is not None:
            circles = np.around(circles)
            for n in circles[0, :]:
                # Draw circle center
                center = (n[0], n[1])
                cv2.circle(src, center, 1, (0, 255, 255), 3)

                # Draw outlining circle
                radius = n[2]
                cv2.circle(src, center, radius, (0, 255, 0), 3)

        # Display
        cv2.imshow("part1", src)
        sleep(1/args.fps)
    cv2.waitKey(0)
    cv2.destroyWindow("part1")

    return

if __name__ == "__main__":
    main()
