"""
Uses detected spheres in a video to make estimations 
on positioning.
"""

import os
from time import sleep
import argparse

import cv2
import numpy as np

import part1

BALL_RADIUS_MM = 30

def convertToCameraCoords(x, y, f, cx, cy):
    """Converts image coordinates to camera coordinates.

    Args:
        x (float): x in image coordinates.
        y (float): y in image coordinates.
        f (float): The focal length.
        cx (float): The principal point.x.
        cy (float): The principal point.y.

    Returns:
        rx, ry: The camera coordinates.
    """
    rx = (x - cx) / f
    ry = (y - cy) / f

    return rx, ry

def readCalib(path=None):
    """Reads the calibration file for the camera used to take the videos.

    Args:
        path (string, optional): The path to the calibration file. Defaults to None.

    Returns:
        f, cx, cy: Returns the camera's simplified intrinsic values.
    """
    if path is None:
        path = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "iphone_calib.txt"

    calibFile = open(path, 'r')
    line = calibFile.readline()
    calibFile.close()

    values = line.split(' ')
    f = float(values[0])
    cx = float(values[1])
    cy = float(values[2])

    return f, cx, cy

def main():
    parser = argparse.ArgumentParser(description="Opens a video with OpenCV, " + \
        "then plays through the video estimating the world coordinates for the " + \
        "target balls.")
    parser.add_argument("infile", nargs=1, type=str, default=None, 
        help="The file to read the video from.")
    parser.add_argument("--fps", nargs="?", type=np.uint16, default=30,
        help="The FPS to display the processed video at.")
    parser.add_argument("--calib", nargs="?", type=str, default=None,
        help="An optional way to pass custom calibration parameters.")
    args = parser.parse_args()

    # Grab data
    f, cx, cy = readCalib(args.calib)
    capture = cv2.VideoCapture(str(args.infile[0]))
    showVideo = capture.isOpened()
    if not showVideo:
        print("Error opening video file at \"" + str(args.infile[0]) + "\".")
    
    # Process and display data
    while capture.isOpened():
        ret, src = capture.read()
        if not ret: break

        circles, _ = part1.findCircles(src, drawCircles=False)
        for circle in circles:
            # Requirement 1
            x, y, r = circle[0]
            camX, camY = convertToCameraCoords(x, y, f, cx, cy)
            
            # Requirement 2
            Z = f * (BALL_RADIUS_MM / r)

            # Requirement 3
            X = ((camX * BALL_RADIUS_MM) / r) - cx
            Y = ((camY * BALL_RADIUS_MM) / r) - cy

            # Requirement 4
            cv2.putText(src, str(int(Z)) + "mm", (np.int(x), np.int(y)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                1, (0, 255, 0))
        
        # Display
        cv2.imshow("part2", src)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if args.fps is not None: sleep(1/args.fps)
    capture.release()
    
    if showVideo:
        cv2.waitKey(0)
        cv2.destroyWindow("part2")

if __name__ == "__main__":
    main()