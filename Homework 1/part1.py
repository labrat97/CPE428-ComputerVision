"""
Open an image, convert it to greyscale, and save it.
"""
import os
import argparse

import cv2


def greyscaleImg(path="frames/000000.jpg", printShape=False, showImage=False, usePathPrefix=True):
    """Loads an image into greyscale, optionally printing parameters or showing
    the image.

    Args:
        path (str, optional): The path of the image to load. Defaults to 'frames/000000.jpg'.
        printShape (bool, optional): Print the parameters. Defaults to True.
        showImage (bool, optional): Show the image on the screen. Defaults to False.
        usePathPrefix (bool, optional): Uses the directory of the module for relative path sourcing. Defaults to True.

    Returns:
        An opencv image.
    """
    pathPrefix = os.path.dirname(os.path.realpath(__file__))
    if usePathPrefix: path = pathPrefix + os.path.sep + path

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if printShape: print(img.shape)
    if showImage:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyWindow('image')

    return img


def main():
    """
    The main function of the module.
    """
    parser = argparse.ArgumentParser(description="Opens an image with OpenCV " + \
        "then prints and shows the image as greyscale. Afterwards it is saved out as " + \
        "a png with the default name \"p1-output.png\" in your current working directory.")
    parser.add_argument("infile", nargs="?", type=str, default=None, help="The input image file.")
    parser.add_argument("outfile", nargs="?", type=str, default="p1-output.png", \
        help="The output image file to write to.")
    args = parser.parse_args()
    
    img = None
    if args.infile is None:
        img = greyscaleImg(printShape=True, showImage=True)
    else:
        img = greyscaleImg(args.infile, printShape=True, showImage=True, usePathPrefix=False)
    
    cv2.imwrite(args.outfile, img)
    return

if __name__ == "__main__":
    main()
