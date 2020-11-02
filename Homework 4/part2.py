import cv2
import argparse
import time
from time import sleep

WINDOW_NAME = "Part 2 - SIFT Matching"
RATIO_THRESHOLD = 0.7

def main():
    """
    The main function of the file.
    """
    parser = argparse.ArgumentParser(description="Runs sift for an image over a video with matching.")
    parser.add_argument("targetImage", nargs=1, type=str, default=None,
        help="The image to search for in the video.")
    parser.add_argument("inputVideo", nargs=1, type=str, default=None,
        help="The video to search inside of.")
    parser.add_argument("--fps", nargs=1, type=int, default=[30], required=False,
        help="The frames per second to display the video at.")
    parser.add_argument("--ratioThresh", nargs=1, type=float, default=[RATIO_THRESHOLD], required=False,
        help="The threshold to use for the ratio test during feature matching.")
    args = parser.parse_args()

    # Load the target image, detect features
    targetImage = cv2.imread(args.targetImage[0], cv2.IMREAD_GRAYSCALE)
    if targetImage is None:
        print("Could not open image at \"" + args.targetImage[0] + "\"")
        exit()
    sift = cv2.SIFT_create()
    targetPoints, targetDescriptions = sift.detectAndCompute(targetImage, None)

    # Create the matching system and match features for each frame
    matcher = cv2.BFMatcher()
    writtenImage = None
    capture = cv2.VideoCapture(args.inputVideo[0])
    if not capture.isOpened():
        print("Could not open the video at \"" + args.inputVideo[0] + "\"")
        exit()
    while capture.isOpened():
        startTime = time.time()
        ret, videoImg = capture.read()
        if not ret: break
        keypoints, descriptions = sift.detectAndCompute(videoImg, None)
        
        # Run the matching system
        matches = matcher.knnMatch(targetDescriptions, descriptions, k=2)

        # Apply ratio test
        good = []
        for match, other in matches:
            if match.distance < args.ratioThresh[0]*other.distance:
                good.append([match])

        # Draw, display
        writtenImage = cv2.drawMatchesKnn(targetImage, targetPoints, videoImg, keypoints,
            good, writtenImage, singlePointColor=(255, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(WINDOW_NAME, writtenImage)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if args.fps is not None: 
            sleepTime = (1 / args.fps[0]) - (time.time() - startTime)
            if sleepTime > 0: sleep(sleepTime)
    capture.release()

    cv2.imshow(WINDOW_NAME, writtenImage)
    cv2.waitKey(0)
    cv2.destroyWindow(WINDOW_NAME)

    exit()

if __name__ == "__main__":
    main()
