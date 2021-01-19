import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plt

args = sys.argv


# comment: resize image with aspect ratio
def image_resize(image, inter=cv.INTER_LINEAR):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    (h, w) = image.shape[:2]
    max_size = max(h, w)

    if max_size < 600:
        return image
    else:
        # check to see if the width is None
        if h > w:
            # calculate the ratio of the height and construct the
            # dimensions
            r = 600 / max_size
            dim = (int(w * r), 600)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = 600 / max_size
            dim = (600, int(h * r))

        # resize the image
        resized = cv.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized


class SiftImage:
    """
    Task 1
    """

    def __init__(self, image):
        self.image = image
        self.process_image = np.zeros(shape=self.image.shape, dtype=np.uint8)
        self.detected_image = np.zeros(shape=self.image.shape, dtype=np.uint8)
        self.detected_image_marker = np.zeros(shape=self.image.shape, dtype=np.uint8)
        self.keypoints = []
        self.descriptors = np.array([])

    # make sure which Y is required in the assignment. YCrCb or YUV???
    def YCrCb_cvt(self):
        self.process_image = cv.cvtColor(self.image, cv.COLOR_BGR2YCrCb)

    # def YUV_cvt(self):
    #     self.process_image = cv.cvtColor(self.image, cv.COLOR_BGR2YUV)

    def Sift(self):
        sift = cv.SIFT_create()
        # self.keypoints = sift.detect(self.process_image[:, :, 0], None)
        # self.detected_image = cv.drawKeypoints(self.process_image[:, :, 0], self.keypoints, self.detected_image,
        #                                        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.keypoints, self.descriptors = sift.detectAndCompute(self.process_image[:, :, 0], None)
        self.detected_image = cv.drawKeypoints(image=self.process_image[:, :, 0], outImage=self.detected_image,
                                               keypoints=self.keypoints, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for keypoint in self.keypoints:
            x = np.int(keypoint.pt[0])
            y = np.int(keypoint.pt[1])
            self.detected_image_marker = cv.drawMarker(self.detected_image, (x, y), (0,255,0),
                                                       markerType=cv.MARKER_CROSS,
                                                       markerSize=4,
                                                       thickness=1)

    # merge image
    def displayImage(self):
        # merge top two images horizontally
        display_image = np.hstack((self.image, self.detected_image_marker))
        return display_image


class BowImage:
    """
    Task 2
    """

    def __init__(self, image):
        # initialization
        self.image_path = image
        self.image = []
        self.input_num = len(image)
        self.keypoints = [0]
        self.descriptors = np.array([])
        self.center = np.array([])
        self.label = np.array([])
        self.histgram = []
        self.dissimilar = np.zeros((self.input_num, self.input_num))

        # sift process image one by one
        for i in range(self.input_num):
            # initialize
            tmp_image = image[i]  # get image path
            tmp_image = cv.imread(tmp_image)  # read image
            tmp_image = image_resize(tmp_image)  # resize to the right image dimensions
            self.process_image = np.zeros(shape=tmp_image.shape, dtype=np.uint8)
            self.detected_image = np.zeros(shape=tmp_image.shape, dtype=np.uint8)

            # process image
            self.YCrCb_cvt(tmp_image)  # process image with YUV
            self.Sift()

        self.num_center = self.center.shape[0]

    # make sure which Y is required in the assignment. YCrCb or YUV???
    def YCrCb_cvt(self, img):
        self.process_image = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
        self.image.append(self.process_image)

    # def YUV_cvt(self, img):
    #     self.process_image = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    #     self.image.append(self.process_image)

    def Sift(self):
        sift = cv.SIFT_create()
        # sift = cv.xfeatures2d.SiftFeatureDetector()
        keypoints, dsp = sift.detectAndCompute(self.process_image[:, :, 0], None)

        # save SIFT details of all image
        self.keypoints.append(len(keypoints))
        if self.descriptors.shape[0] == 0:
            self.descriptors = dsp
        else:
            self.descriptors = np.concatenate((self.descriptors, dsp), axis=0)

    def KMeans(self, percentage):
        self.center = np.array([])
        self.label = np.array([])
        # define criteria, number of clusters(K) and apply kmeans()
        K = int(percentage * self.descriptors.shape[0] / 100)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, self.label, self.center = cv.kmeans(self.descriptors, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.1)
        # ret, self.label, self.center = cv.kmeans(self.descriptors, K, None, criteria, 25, cv.KMEANS_RANDOM_CENTERS)

    def HistConstruct(self):
        passed_label = 0
        self.histgram = []
        for i in range(self.input_num):
            # slice label into image
            slice_label = self.label[passed_label: passed_label + self.keypoints[i + 1]]
            hist, _ = np.histogram(slice_label, bins=self.center.shape[0], range=(0, self.center.shape[0]))
            self.histgram.append(hist)
            passed_label += self.keypoints[i + 1]

    def DistCal(self):
        # channels = [0]  # only use Y channel to calculate histogram
        # histSize = [self.center.shape[0]]
        # ranges = [0, 256]
        for i in range(self.input_num):
            # img1 = self.image[i]
            # hist_img1 = cv.calcHist([img1], channels, None, histSize, ranges, accumulate=False)
            # hist_img1 = cv.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
            # hist_img1 = np.zeros_like(self.histgram[i]).astype('float32')
            # hist_img1 = cv.normalize(self.histgram[i].astype('float32'), hist_img1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

            hist_img1 = (self.histgram[i] / self.descriptors.shape[0]).astype('float32')
            # hist_img1 = (self.histgram[i] / self.keypoints[i + 1]).astype('float32')
            # print(sum(hist_img1))

            for j in range(self.input_num):
            # for j in range(i, self.input_num):
                # img2 = self.image[j]
                # hist_img2 = cv.calcHist([img2], channels, None, histSize, ranges, accumulate=False)
                # hist_img2 = cv.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
                # hist_img2 = np.zeros_like(self.histgram[j]).astype('float32')
                # hist_img2 = cv.normalize(self.histgram[j].astype('float32'), hist_img2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

                hist_img2 = (self.histgram[j] / self.descriptors.shape[0]).astype('float32')
                # hist_img2 = (self.histgram[j] / self.keypoints[j + 1]).astype('float32')

                self.dissimilar[i, j] = cv.compareHist(hist_img1, hist_img2, method=cv.HISTCMP_CHISQR_ALT) / 4
                # self.dissimilar[i, j] = cv.compareHist(hist_img1, hist_img2, method=1) / 2
            #     print(sum(hist_img2))
            # print()

    def OutputKeypoints(self):
        # print keypoints number for every jpg
        for i in range(self.input_num):
            print('# of keypoints in {} is {}'.format(self.image_path[i], self.keypoints[i + 1]))
        print()

    def OutputMatrix(self, percent, name_list):
        print('\nK = {}% * {} = {}'.format(percent, self.descriptors.shape[0],
                                           int(percent * self.descriptors.shape[0] / 100)))
        print('Dissimilarity Matrix')

        for i in range(self.input_num + 1):
            template = "{:^15}" * (self.input_num + 1)
            name = []
            if i == 0:
                name.append('\t')
            else:
                name.append(name_list[i - 1])
            for j in range(self.input_num):
            # for j in range(self.input_num):
                if i == 0:
                    name.append(name_list[j])
                else:
                    name.append(str(round(self.dissimilar[i-1, j], 6)))
            print(template.format(*name))




# run python program

def optimize_path(paths):
    # optimize the name of input (for the readable matrix output)
    optim_name = []
    for path in paths:
        if len(path.rsplit('\\', 1)) > 1:
            optim_name.append(path.rsplit('\\', 1)[-1])
        else:
            optim_name.append(path)
    return optim_name


# TASK 1
def find_keypoints(image_path):
    image = image_path[0]
    img = cv.imread(image)
    img_resized = image_resize(img)
    SiftImg = SiftImage(img_resized)
    SiftImg.YCrCb_cvt()
    SiftImg.Sift()
    display = SiftImg.displayImage()

    # output in command
    print('# of keypoints in {} is {}'.format(image, len(SiftImg.keypoints)))

    # display merged image
    cv.imshow('Display image', display)
    cv.waitKey()


# TASK 2
def find_dissimilarity(image_path, optimized_path):
    percentages = [5, 10, 20]
    BagImage = BowImage(image_path)
    BagImage.OutputKeypoints()  # output number of keypoints for every jpg
    for p in percentages:
        BagImage.KMeans(p)  # K means cluster
        BagImage.HistConstruct()  # construct histograms of occurrence of the visual words
        BagImage.DistCal()  # calculate the Chi-Square distance
        BagImage.OutputMatrix(p, optimized_path)  # output the matrix to readable format

    #     if p == 5:
    #         hist_1 = BagImage.histgram[0]
    #     if p == 10:
    #         hist_2 = BagImage.histgram[0]
    # print(hist_1 == hist_2)


params = args[1:]
# params = [r'D:\Others\CSCI935\a2\img01.jpg']
# params = [r'D:\Others\CSCI935\a2\img01.jpg', r'D:\Others\CSCI935\a2\img02.jpg',
#           r'D:\Others\CSCI935\a2\img03.jpg', r'D:\Others\CSCI935\a2\img04.jpg']
if len(params) > 1:
    op_name = optimize_path(params)
    find_dissimilarity(params, op_name)

else:
    find_keypoints(params)


























