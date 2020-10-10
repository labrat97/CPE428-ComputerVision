"""
Draws boxes around the balls detected in the videos, making estimations.
"""

from time import sleep
import argparse

import cv2
import numpy as np

import part2
import part1


def worldToImageCoords(X, Y, Z, f, cx, cy):
    """Convert 3D world coordinates to image coordinates. This also satisfies
    requirement 1.

    Args:
        X (float): The world x-point.
        Y (float): The world y-point.
        Z (float): The world z-point.
        f (float): The focal length of the camera.
        cx (float): The principle x-point.
        cy (float): The principle y-point.

    Returns:
        x, y: The image space coordinates.
    """
    x = (f * (X / Z)) + cx
    y = (f * (Y / Z)) + cy

    return x, y


def drawProjectedLine(sourceImage, A, B, f, cx, cy, color=(0, 0, 255)):
    """Draws a line that's been projected from world coordinates to the image
    space coordinates. This also satisfies requirement 2.

    Args:
        sourceImage (cv2.Mat): The image to draw on (will be copied).
        A (float[3]): X, Y, Z in world space coordinates as the first point of the line.
        B (float[3]): X, Y, Z in world space coordinates as the second point of the line.
        f (float): The focal length of the camera.
        cx (float): The principle x-point.
        cy (float): The principle y-point.
        color (uint8[3]): The color in BGR to color the projected line. Defaults to (0, 0, 255).

    Returns:
        cv2.Mat: The image that has been drawn on.
    """
    src = sourceImage.copy()

    # Convert to image space coordinates
    xA, yA = worldToImageCoords(A[0], A[1], A[2], f, cx, cy)
    xB, yB = worldToImageCoords(B[0], B[1], B[2], f, cx, cy)

    cv2.line(src, (int(xA), int(yA)), (int(xB), int(yB)), color, thickness=1)
    return src


def main():
    """
    The main function of the file.
    """
    parser = argparse.ArgumentParser("Mark bounding boxes around the circles detected " + \
        "in each frame.")
    parser.add_argument("infile", nargs=1, type=str, default=None, 
        help="The file to read the video from.")
    parser.add_argument("--fps", nargs="?", type=np.uint16, default=30,
        help="The FPS to display the processed video at.")
    parser.add_argument("--calib", nargs="?", type=str, default=None,
        help="An optional way to pass custom calibration parameters.")
    args = parser.parse_args()

    # Grab data
    f, cx, cy = part2.readCalib(args.calib)
    capture = cv2.VideoCapture(str(args.infile[0]))
    showVideo = capture.isOpened()
    if not showVideo:
        print("Error opening video file at \"" + str(args.infile[0]) + "\".")
    
    # Process and display data
    while capture.isOpened():
        ret, src = capture.read()
        if not ret: break

        circles, _ = part1.findCircles(src, drawCircles=False)
        if circles is not None:
            for circle in circles[0, :]:
                # Requirement 3
                x, y, _ = circle
                X, Y, Z = part2.findCircleWorldCoords(circle, f, cx, cy)
                R = part2.BALL_RADIUS_MM

                for i in range(2):
                    i = (i * 2) - 1
                    for j in range(2):
                        j = (j * 2) - 1

                        src = drawProjectedLine(src, (X+(i*R), Y+(j*R), Z-R), (X+(i*R), Y+(j*R), Z+R), f, cx, cy)
                        src = drawProjectedLine(src, (X+(i*R), Y-R, Z+(j*R)), (X+(i*R), Y+R, Z+(j*R)), f, cx, cy)
                        src = drawProjectedLine(src, (X-R, Y+(i*R), Z+(j*R)), (X+R, Y+(i*R), Z+(j*R)), f, cx, cy)

                cv2.putText(src, str(int(Z)) + "mm", (np.int(x), np.int(y)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 255, 255))
        
        # Display
        cv2.imshow("part3", src)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if args.fps is not None: sleep(1/args.fps)
    capture.release()

    if showVideo:
        cv2.waitKey(0)
        cv2.destroyWindow("part3")

if __name__ == "__main__":
    main()
