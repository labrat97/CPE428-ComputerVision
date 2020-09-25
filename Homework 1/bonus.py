"""
Extends the work of module part3 to the entirety of the frames and draws
bounding boxes around each car.
"""
import os
import argparse
from time import sleep

import cv2

import part2
import part3

def processVideo(path='frames', useGauss=True, usePathPrefix=True, hz=None):
    """Get the threshold of a video then draw bounding boxes on it.

    Args:
        path (str, optional): The path to load the frames from. Defaults to 'frames'.
        useGauss (bool, optional): Use a guassian blur prior to Otsu's method. Defaults to True.
        usePathPrefix (bool, optional): Use the path prefix of this file. Defaults to True.
        hz (float, optional): The refresh rate of the video (approx). Defaults to None.
    """
    pathPrefix = os.path.dirname(os.path.realpath(__file__))
    if usePathPrefix: path = pathPrefix + os.path.sep + path

    # Gather average, process video with Otsu's method, then display
    avg = part2.averageBackground(path=path, usePathPrefix=False)
    cap = cv2.VideoCapture(path + os.path.sep + "%06d.jpg")
    otsuFrames = []
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None: break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, _, otsu = part3.backgroundDifference(grey, avg, useGauss=useGauss)

        otsuFrames.append(otsu)
        cv2.imshow('bonus', otsu)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if hz is not None: sleep(1/hz)
    cap.release()
    cv2.waitKey(0)

    # Draw bounding boxes and display
    for otsu in otsuFrames:
        contours, _ = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

        bgr = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 50: continue
            bgr = cv2.rectangle(bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('bonus', bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):break
        if hz is not None: sleep(1/hz)
    cv2.waitKey(0)
    cv2.destroyWindow('bonus')

def main():
    """
    The main function of the module.
    """
    parser = argparse.ArgumentParser(description="Runs the background difference over " + \
        "the entire set of frames. Also draws bounding boxes around detected contours.")
    parser.add_argument("inpath", nargs="?", type=str, default=None, help="The directory " + \
        "to load the video from.")
    parser.add_argument("--useGauss", nargs="?", type=bool, default=True, \
        help="Apply a gaussian filter prior to using Otsu's method.")
    args = parser.parse_args()

    if args.inpath is None:
        processVideo(usePathPrefix=args.useGauss, hz=24)
    else:
        processVideo(path=args.inpath, usePathPrefix=False, useGauss=args.useGauss, hz=24)

    return

if __name__ == "__main__":
    main()
