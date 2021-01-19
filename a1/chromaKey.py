import cv2 as cv
import numpy as np
import sys


args = sys.argv
param_1 = args[1]
image_path = args[2]


# different color space convert
class ColorSpaceCvt:
    # initialize
    def __init__(self, image):
        self.image = image
        self.process_image = np.zeros(shape=self.image.shape, dtype=np.uint8)
        self.image_width = 540

    # XYZ color space convert
    def ColorCvtXyz(self):
        self.process_image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)

    # YCrCb color space convert
    def ColorCvtYcrcb(self):
        self.process_image = cv.cvtColor(self.image, cv.COLOR_BGR2YCrCb)

    # LAB color space convert
    def ColorCvtLab(self):
        self.process_image = cv.cvtColor(self.image, cv.COLOR_BGR2LAB)

    # HSV color space convert
    def ColorCvtHsv(self):
        self.process_image = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)

    # merge different image
    def displayImage(self):
        # different color space channel initialization
        channel_0 = np.zeros(shape=self.process_image.shape, dtype=np.uint8)
        channel_1 = np.zeros(shape=self.process_image.shape, dtype=np.uint8)
        channel_2 = np.zeros(shape=self.process_image.shape, dtype=np.uint8)

        # transfer color space channel 0 to image, image should be 3 channels
        channel_0[:, :, 0] = self.process_image[:, :, 0]
        channel_0[:, :, 1] = self.process_image[:, :, 0]
        channel_0[:, :, 2] = self.process_image[:, :, 0]

        # transfer color space channel 1 to image, image should be 3 channels
        channel_1[:, :, 0] = self.process_image[:, :, 1]
        channel_1[:, :, 1] = self.process_image[:, :, 1]
        channel_1[:, :, 2] = self.process_image[:, :, 1]

        # transfer color space channel 2 to image, image should be 3 channels
        channel_2[:, :, 0] = self.process_image[:, :, 2]
        channel_2[:, :, 1] = self.process_image[:, :, 2]
        channel_2[:, :, 2] = self.process_image[:, :, 2]

        # resize four images while keeping aspect ratio
        orig_image = image_resize(self.image, width=self.image_width)
        channel_0 = image_resize(channel_0, width=self.image_width)
        channel_1 = image_resize(channel_1, width=self.image_width)
        channel_2 = image_resize(channel_2, width=self.image_width)

        # merge top two images horizontally
        image_top = np.hstack((orig_image, channel_0))

        # merge bottom two images horizontally
        image_bot = np.hstack((channel_1, channel_2))

        # merge four images together
        display_image = np.vstack((image_top, image_bot))

        return display_image


# blend green screen image and scenic image
class BackBlend:
    """
    This class is used for blend green screen image with scenic image together.
    """
    # initialize
    def __init__(self, green_image, scenic_image):
        """
        Initialization. HSV and LAB color space conversion is performed here.
        :param green_image: green screen image
        :param scenic_image: scenic image
        """
        # initalize image width
        self.image_width = 540

        self.green_image = green_image
        self.scenic_image = scenic_image

        # process image to HSV channel, for further subject extraction
        self.green_image_hsv = cv.cvtColor(self.green_image, cv.COLOR_BGR2HSV)

        # process image to LAB channel, did not use here since the result is not as good as HSV convert
        self.green_image_lab = cv.cvtColor(self.green_image, cv.COLOR_BGR2LAB)

        # resize scenic image to the size of green screen image, for mask extraction and blending
        self.scenic_image = cv.resize(self.scenic_image, (self.green_image.shape[1], self.green_image.shape[0]))

        # initialization
        self.mask = np.zeros(shape=self.green_image.shape, dtype=np.uint8)
        self.foreground = np.zeros(shape=self.green_image.shape, dtype=np.uint8)
        self.background = np.zeros(shape=self.green_image.shape, dtype=np.uint8)
        self.final = np.zeros(shape=self.green_image.shape, dtype=np.uint8)

    # extract mask
    def mask_extract(self):
        """
        Extract mask based on HSV conversion of green screen image.
        """

        # define the background color for mask extraction
        lower_green = (30,80,80)
        upper_green = (150,255,255)

        # find green background, green is 255
        mask = cv.inRange(self.green_image_hsv, lower_green, upper_green)
        # convert mask, green is 0
        mask = 255 - mask

        # apply morphology opening to mask
        kernel = np.ones((1, 1), np.uint8)
        # erode edge of the mask, background of the mask here is 0 (black)
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)
        # remove the back points in the foreground of mask
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        # antialias mask
        self.mask = cv.GaussianBlur(mask, (1, 1), sigmaX=1, sigmaY=1, borderType=cv.BORDER_DEFAULT)  # mask, green is 0


    def mask_extract_lab(self):
        """
        Extract mask based on LAB conversion of green screen image. (did not used in this assignment)
        """

        L = self.green_image_lab[:, :, 0]
        A = self.green_image_lab[:, :, 1]
        B = self.green_image_lab[:, :, 2]

        # negate A
        A = (255 - A)

        # multiply negated A by B
        nAB = 255 * (A / 255) * (B / 255)
        nAB = np.clip((nAB), 0, 255)
        nAB = np.uint8(nAB)

        lower_green = 90
        upper_green = 160
        mask = cv.inRange(nAB, lower_green, upper_green)  # find green background, green is 255
        mask = 255 - mask  # convert mask, green is 0

        # apply morphology opening to mask
        kernel = np.ones((1, 1), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        # antialias mask
        self.mask = cv.GaussianBlur(mask, (1, 1), sigmaX=1, sigmaY=1, borderType=cv.BORDER_DEFAULT)  # mask, green is 0

    # extract foreground from green screen image by mask
    def gene_foreground(self):
        """
        Extract foreground from green screen image.
        """

        self.foreground = cv.bitwise_and(self.green_image, self.green_image, mask=self.mask)

    # extract background from scenic image by mask
    def gene_background(self):
        """
        Extract background from green screen image.
        """

        # convert mask, otherwise cannot extract background since self.mask is 0 on ground
        mask = 255 - self.mask
        self.background = cv.bitwise_and(self.scenic_image, self.scenic_image, mask=mask)

    # blend background and foreground images together
    def combine_back_fore(self):
        """
        Combine foreground and background.
        """

        self.final = cv.bitwise_or(self.foreground, self.background)

    # merge images
    def displayImage(self):
        """
        Creat display image: green screen image, white screen image, scenic image, blended image.
        """

        # create the foreground image with white background
        channel_0 = self.green_image.copy()
        channel_0[self.mask==0] = (255, 255, 255)

        # resize images keeping aspect ratio
        orig_image = image_resize(self.green_image, width=self.image_width)
        channel_0 = image_resize(channel_0, width=self.image_width)
        scenic = image_resize(self.scenic_image, width=self.image_width)
        final = image_resize(self.final, width=self.image_width)

        # merge images
        image_top = np.hstack((orig_image, channel_0))
        image_bot = np.hstack((scenic, final))
        display_image = np.vstack((image_top, image_bot))

        return display_image


# comment: resize image with aspect ratio
def image_resize(image, width=None, height=None, inter=cv.INTER_LINEAR):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def colorSplit(color_space, path):
    # read image
    image = cv.imread(path)

    # call color space convert class
    ColorCvt = ColorSpaceCvt(image)

    # determine which color space is expected
    if color_space == '-RGB':
        ColorCvt.ColorCvtXyz()
    if color_space == '-YCrCb':
        ColorCvt.ColorCvtYcrcb()
    if color_space == '-Lab':
        ColorCvt.ColorCvtLab()
    if color_space == '-HSB':
        ColorCvt.ColorCvtHsv()

    img = ColorCvt.displayImage()

    # display merged image
    cv.imshow('Color Space {}'.format(color_space), img)
    cv.waitKey()


def subject_blend(greenpath, backgourndpath):

    # read green screen images and scenic image
    green_image = cv.imread(greenpath)
    scenic_image = cv.imread(backgourndpath)

    # image processing
    blendBack = BackBlend(green_image, scenic_image)

    # extract mask basic on green screen image
    blendBack.mask_extract()

    # extract foreground and background image
    blendBack.gene_foreground()
    blendBack.gene_background()

    # merge background and foreground images
    blendBack.combine_back_fore()

    # merge images and display
    display_image = blendBack.displayImage()
    cv.imshow('Subjects blend', display_image)
    cv.waitKey()


# if the first input is not a jpg file, color space convert
if len(param_1.rsplit('.', 1)) > 1:
    subject_blend(param_1, image_path)
# if the first input is jpg file, blend subjects
else:
    colorSplit(param_1, image_path)





