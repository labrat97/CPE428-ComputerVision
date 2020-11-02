# TODO - This is currently just a copy of part 3
"""
Detects matches from a target image in a video, then draws them.
"""
import cv2
import numpy as np
import argparse
import time
from time import sleep

WINDOW_NAME = "Bonus - AR Overlay"
RATIO_THRESHOLD = 0.7
MINIMUM_GOOD_POINTS = 10

def main():
    """
    The main function of the file.
    """
    parser = argparse.ArgumentParser(description="Trys to draw an overlay in an AR like fashion over the target image " +
        "in the video provided with the overlay provided.")
    parser.add_argument("targetImage", nargs=1, type=str, default=None,
        help="The image to search for in the video.")
    parser.add_argument("inputVideo", nargs=1, type=str, default=None,
        help="The video to search inside of.")
    parser.add_argument("overlayImage", nargs=1, type=str, default=None,
        help="The image to overlay over the target image.")
    parser.add_argument("--fps", nargs=1, type=int, default=[30], required=False,
        help="The frames per second to display the video at.")
    parser.add_argument("--ratioThresh", nargs=1, type=float, default=[RATIO_THRESHOLD], required=False,
        help="The threshold to use for the ratio test during feature matching.")
    args = parser.parse_args()

    # Logging
    inlierCount = 0
    matchCount = 0

    # Load the target image, detect features
    targetImage = cv2.imread(args.targetImage[0], cv2.IMREAD_GRAYSCALE)
    if targetImage is None:
        print("Could not open image at \"" + args.targetImage[0] + "\"")
        exit()
    overlayImage = cv2.imread(args.overlayImage[0], cv2.IMREAD_COLOR)
    if overlayImage is None:
        print("Cound not open image at \"" + args.overlayImage[0] + "\"")
    overlayMaskBase = np.ones_like(overlayImage)
    sift = cv2.SIFT_create()
    targetPoints, targetDescriptions = sift.detectAndCompute(targetImage, None)

    # Create the matching system and match features for each frame
    matcher = cv2.BFMatcher()
    capture = cv2.VideoCapture(args.inputVideo[0])
    videoImg = None
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
                good.append(match)
        matchCount += len(good)

        # Calculate homography
        targetHPoints = np.float32([targetPoints[match.queryIdx].pt for match in good]).reshape(-1,1,2)
        hPoints = np.float32([keypoints[match.trainIdx].pt for match in good]).reshape(-1,1,2)
        targetObject, mask = cv2.findHomography(targetHPoints, hPoints, cv2.RANSAC)
        matchesMask = mask.ravel().tolist()

        # Get the count of the inliers
        for matchResult in matchesMask:
            if matchResult == 1: inlierCount += 1

        # Draw, show object
        if len(good) > MINIMUM_GOOD_POINTS:
            # Mask and warp
            dataShape = videoImg.shape[:2][::-1]
            overlayImg = cv2.warpPerspective(overlayImage, targetObject, dataShape)
            overlayMask = cv2.warpPerspective(overlayMaskBase, targetObject, dataShape)

            videoImg = (videoImg*(1-overlayMask)) + (overlayImg*overlayMask)
        
        cv2.imshow(WINDOW_NAME, videoImg)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        if args.fps is not None: 
            sleepTime = (1 / args.fps[0]) - (time.time() - startTime)
            if sleepTime > 0: sleep(sleepTime)
    capture.release()

    cv2.imshow(WINDOW_NAME, videoImg)
    cv2.waitKey(0)
    cv2.destroyWindow(WINDOW_NAME)

    print("Inlier count: \t\t" + str(inlierCount))
    print("Total match count: \t" + str(matchCount))
    print("Inlier percentage: \t%" + str(inlierCount*100/matchCount))

    exit()

if __name__ == "__main__":
    main()
