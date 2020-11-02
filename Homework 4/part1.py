"""
Simply runs and draws SIFT computations.
"""
import cv2
import argparse
import time
from time import sleep

WINDOW_NAME = "Part 1 - SIFT Example"

def main():
    """
    The main function of the file.
    """
    parser = argparse.ArgumentParser(description="Runs sift for an image and a video.")
    parser.add_argument("inputImage", nargs=1, type=str, default=None,
        help="The image to run sift on.")
    parser.add_argument("inputVideo", nargs=1, type=str, default=None,
        help="The video to search through.")
    parser.add_argument("--fps", nargs=1, type=int, default=30, required=False,
        help="The frames per second to display the video at.")
    args = parser.parse_args()

    # Load the target image, detect features
    targetImage = cv2.imread(args.inputImage[0], cv2.IMREAD_GRAYSCALE)
    if targetImage is None:
        print("Could not open image at \"" + args.inputImage[0] + "\"")
        exit()
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(targetImage, None)
    writtenImage = None
    writtenImage = cv2.drawKeypoints(targetImage, keypoints, writtenImage,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display
    cv2.imshow(WINDOW_NAME, writtenImage)
    cv2.waitKey(0)

    # Do the same to the video
    capture = cv2.VideoCapture(args.inputVideo[0])
    if not capture.isOpened():
        print("Could not open the video at \"" + args.inputVideo[0] + "\"")
        exit()
    while capture.isOpened():
        startTime = time.time()
        ret, videoImg = capture.read()
        if not ret: break

        keypoints, _ = sift.detectAndCompute(videoImg, None)
        writtenImage = cv2.drawKeypoints(videoImg, keypoints, writtenImage,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display
        cv2.imshow(WINDOW_NAME, writtenImage)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if args.fps is not None: 
            sleepTime = (1 / args.fps) - (time.time() - startTime)
            if sleepTime > 0: sleep(sleepTime)
    capture.release()

    cv2.imshow(WINDOW_NAME, writtenImage)
    cv2.waitKey(0)
    cv2.destroyWindow(WINDOW_NAME)

    exit()


if __name__ == "__main__":
    main()
