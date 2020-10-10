"""
Checks the accuracy of the 3D projections from the previous parts in the lab.
"""

import argparse

import cv2
import numpy as np

import part2
import part1

OPTIMAL_DISTANCE = 360

def main():
    """
    The main function of the file.
    """
    parser = argparse.ArgumentParser("Checks the accuracy of the projections created " + \
        "in the previous parts of the lab.")
    parser.add_argument("infile", nargs=1, type=str, default=None, 
        help="The file to read the video from.")
    parser.add_argument("--calib", nargs="?", type=str, default=None,
        help="An optional way to pass custom calibration parameters.")
    args = parser.parse_args()

    # Grab data
    f, cx, cy = part2.readCalib(args.calib)
    capture = cv2.VideoCapture(str(args.infile[0]))
    if not capture.isOpened():
        print("Error opening video file at \"" + str(args.infile[0]) + "\".")
    
    # Process data
    missedFrames = 0
    frames = 0
    distances = []
    while capture.isOpened():
        ret, src = capture.read()
        if not ret: break
        frames += 1

        # Find 2 circles in the frame
        circles, _ = part1.findCircles(src, drawCircles=False)
        if circles is None or len(circles[0, :]) != 2: 
            missedFrames += 1
            continue
        circles = circles[0, :]

        # Compute distance
        coordA = part2.findCircleWorldCoords(circles[0], f, cx, cy)
        coordB = part2.findCircleWorldCoords(circles[1], f, cx, cy)

        distance = 0
        for i in range(3):
            distance += (coordA[i] - coordB[i])**2
        distances.append(np.sqrt(distance))

    # Analyze
    if len(distances) == 0:
        print("No wand detected.")
        return

    mean = np.mean(distances)
    stdDev = np.std(distances)

    # Display
    print("Mean: " + str(mean) + "mm")
    print("Standard Deviation: " + str(stdDev) + "mm")
    accuracy = 100 - np.abs(mean-OPTIMAL_DISTANCE)*100/OPTIMAL_DISTANCE
    print("Mean accuracy: %" + str(accuracy))
    
    if (mean+stdDev) >= OPTIMAL_DISTANCE and (mean-stdDev) <= OPTIMAL_DISTANCE:
        print("Within standard deviation.")
    else:
        print("Outside of standard deviation.")
    print(str(missedFrames) + " out of " + str(frames) + " frames missed.")

if __name__ == "__main__":
    main()
