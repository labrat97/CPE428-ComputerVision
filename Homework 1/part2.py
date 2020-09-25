"""
Creates a background image by averaging together all of the frames provided.
"""
import os
import argparse
from time import sleep

import cv2
import numpy as np


def averageBackground(path="frames", showVideo=False, usePathPrefix=True, hz=None):
    """Averages out a set of frames into the greyscale background.

    Args:
        path (str, optional): The directory to load the frames from. Defaults to "frames".
        showVideo (bool, optional): Show the video during processing and after. Defaults to False.
        usePathPrefix (bool, optional): Use the path prefix of the module. Defaults to True.

    Returns:
        An opencv image.
    """
    pathPrefix = os.path.dirname(os.path.realpath(__file__))
    if usePathPrefix: path = pathPrefix + os.path.sep + path

    # Load all frames then average
    cap = cv2.VideoCapture(path + os.path.sep + "%06d.jpg")
    frames = []
    while cap.isOpened:
        _, frame = cap.read()
        if frame is None: break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grey)

        # Optionally show video
        if not showVideo: continue
        cv2.imshow('frame', grey)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if hz is not None: sleep(1/hz)
    cap.release()
    cv2.waitKey(0)
    
    # Final compute using numpy
    avg = np.mean(frames, axis=0).astype('uint8')
    if showVideo:
        cv2.imshow('frame', avg)

        # Close window
        cv2.waitKey(0)
        cv2.destroyWindow('frame')
    
    return avg


def main():
    """
    The main function of the module.
    """
    parser = argparse.ArgumentParser(description="Opens a video with OpenCV " + \
        "then plays through the video in greyscale saving an average.")
    parser.add_argument("indir", nargs="?", type=str, default=None, \
        help="The directory to load the video frames from.")
    parser.add_argument("outfile", nargs="?", type=str, default="p2-output.png", \
        help="The file to write the output image to.")
    args = parser.parse_args()

    avg = None
    if args.indir is None:
        avg = averageBackground(showVideo=True, hz=24)
    else:
        avg = averageBackground(path=args.indir, showVideo=True, usePathPrefix=False, hz=24)

    cv2.imwrite(args.outfile, avg)
    return

if __name__ == "__main__":
    main()
