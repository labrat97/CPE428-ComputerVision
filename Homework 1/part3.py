"""
Uses the work done in modules part2 and part1 to create a background difference
calculation.
"""
import argparse
import cv2

import part1
import part2


def backgroundDifference(targetImg, backgroundImg, threshVal=127, useGauss=True):
    """Generates three different images from comparisons of the target and background
    image. One is the absolute difference between the images, one is the threshold
    of the images with the provided value and binary masking, and finally there is
    the threshold image from Otsu's method (optionally with gaussian blur applied
    before analysis).

    Args:
        targetImg (cv2.Mat): The target image to compare against the background.
        backgroundImg (cv2.Mat): The background image to compare against.
        threshVal (int, optional): The threshold value for the binary mask. Defaults to 127.
        useGauss (bool, optional): Apply a gaussian blur to the image prior to 
            using Otsu's method. Defaults to True.

    Returns:
        The absolute difference, threshold image, and threshold image using Otsu's method.
    """
    absDiff = cv2.absdiff(targetImg, backgroundImg)
    _, threshImg = cv2.threshold(absDiff, threshVal, 255, cv2.THRESH_BINARY)

    otsuImg = None
    if useGauss:
        blur = cv2.GaussianBlur(absDiff, (7, 7), 0)
        _, otsuImg = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        _, otsuImg = cv2.threshold(absDiff, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return absDiff, threshImg, otsuImg

def main():
    """
    The main function of the module.
    """
    parser = argparse.ArgumentParser(description="Uses the result of modules part1 and part2 to " + \
        "compute the absolute difference, the threshold image using binary masking, and the " + \
        "threshold image using Otsu's method.")
    parser.add_argument("infile", nargs="?", type=str, default=None, help="The input image file.")
    parser.add_argument("indir", nargs="?", type=str, default=None, \
        help="The directory to load the video frames from.")
    parser.add_argument("--useGauss", nargs="?", type=bool, default=False, help="Use gaussian " + \
        "blurring prior to using Otsu's method.")
    args = parser.parse_args()

    # I know I'm not using the IMREAD_GRAYSCALE option here. I know exactly what it does
    # and how it works, I just wanted reusability. (cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    sourceImage = None
    backgroundImage = None
    if args.infile is None:
        sourceImage = part1.greyscaleImg(printShape=True)
    else:
        sourceImage = part1.greyscaleImg(path=args.infile, printShape=True, usePathPrefix=False)
    if args.indir is None:
        backgroundImage = part2.averageBackground()
    else:
        backgroundImage = part2.averageBackground(path=args.indir, usePathPrefix=False)

    absDiffImg, threshImg, otsuImg = backgroundDifference(sourceImage, backgroundImage, threshVal=50, useGauss=args.useGauss)

    cv2.imshow('difference', absDiffImg)
    cv2.waitKey(0)
    cv2.imshow('difference', threshImg)
    cv2.waitKey(0)
    cv2.imshow('difference', otsuImg)
    cv2.waitKey(0)
    cv2.destroyWindow('difference')

    return

if __name__ == "__main__":
    main()
