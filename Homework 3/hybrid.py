"""
Makes a hybrid of two provided images.
"""
import cv2
import numpy as np
import argparse

KERNEL_SIZE = (31, 31)
KERNEL_SIGMA = 5
DEFAULT_EXT = '.png'

def normalizeImage(image, maxRange=255, minRange=0):
    """Scales an image to fit in the range of [0, 1].

    Args:
        image (cv2.Mat): The image to modify.
        maxRange (float, optional): The high range of values from the image. Defaults to 255.
        minRange (float, optional): The low range of values from the image. Defaults to 0.

    Returns:
        cv2.Mat: The image normalized to the range of [0, 1].
    """
    spread = maxRange - minRange
    return (image - minRange) / spread

def denornalizeImage(image, maxRange=255, minRange=0):
    """Scales an image back up to a usable range.

    Args:
        image (cv2.Mat): The image to modify.
        maxRange (float, optional): The high range of values from the original image. Defaults to 255.
        minRange (float, optional): The low range of values from the original image. Defaults to 0.

    Returns:
        cv2.Mat: The image re-scaled to the range of [minRange, maxRange].
    """
    spread = maxRange - minRange
    return (image * spread) + minRange

def main():
    """
    The main function for the file.
    """
    parser = argparse.ArgumentParser(description="Create a hybrid of images " +
        "based off the features found in them.")
    parser.add_argument("featureFile", nargs=1, type=str, default=None, 
        help="The file to read the features, viewable in a full size depiction " + 
        "of the image, to merge into the hybrid image.")
    parser.add_argument("colorFile", nargs=1, type=str, default=None, 
        help="The file to read the colors out of, viewable in a preview like " + 
        "depiction, to merge into the hybrid image.")
    parser.add_argument("output", nargs="?", type=str, default="hybrid.png",
        help="The hybrid image output.")
    args = parser.parse_args()

    # Open the images
    featureImage = cv2.imread(args.featureFile[0], cv2.IMREAD_COLOR)
    colorImage = cv2.imread(args.colorFile[0], cv2.IMREAD_COLOR)
    if featureImage is None:
        print('Error opening feature image.')
        parser.print_usage()
        return -1
    if colorImage is None:
        print('Error opening color image.')
        parser.print_usage()
        return -1

    # Step 1
    featureNorm = normalizeImage(featureImage)
    colorNorm = normalizeImage(colorImage)

    # Step 2
    kernel = cv2.getGaussianKernel(KERNEL_SIZE, KERNEL_SIGMA)
    lowpassKernel = kernel * kernel.transpose()

    # Step 3
    highpassKernel = 1 - lowpassKernel

    # Step 4
    lowImage = cv2.filter2D(colorNorm, -1, lowpassKernel)
    highImage = cv2.filter2D(featureNorm, -1, highpassKernel)

    # Step 5
    outputImage = denornalizeImage(lowImage + highImage)
    outfileName = args.output[0]
    if outfileName[-4:] != DEFAULT_EXT:
        outfileName += DEFAULT_EXT
    
    return cv2.imwrite(outfileName, outputImage)

if __name__ == "__main__":
    main()
