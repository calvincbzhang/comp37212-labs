import cv2
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from scipy import ndimage, spatial


def show(img):
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    # plt.imshow(img, cmap='gray')
    # plt.show()


def test(img):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location, not size and orientation
    img_kp = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    show(img_kp)


def HarrisPointsDetector(img):
    # keep a copy of the original image
    original = img.copy()
    # if intermediary steps should be shown
    show_img = 0
    # list of keypoints to return
    keypoints = []

    # smooth image
    img = cv2.GaussianBlur(img, (9,9), 3)

    # calculate derivatives using Sobel operator
    Ix = cv2.Sobel(img, -1, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(img, -1, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    if show_img:
        show(Ix)
        show(Iy)

    # compute second derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    if show_img:
        show(Ixx)
        show(Iyy)
        show(Ixy)

    # apply Gaussian mask
    g_Ixx = cv2.GaussianBlur(Ixx, (5,5), 0.5, borderType=cv2.BORDER_REFLECT)
    g_Iyy = cv2.GaussianBlur(Iyy, (5,5), 0.5, borderType=cv2.BORDER_REFLECT)
    g_Ixy = cv2.GaussianBlur(Ixy, (5,5), 0.5, borderType=cv2.BORDER_REFLECT)

    if show_img:
        show(g_Ixx)
        show(g_Iyy)
        show(g_Ixy)

    # compute the gradient orientations
    gradients = np.arctan2(Ix, Iy) * 180 / np.pi

    # compute corner strength function
    detM = (g_Ixx * g_Iyy) - (g_Ixy ** 2)
    traceM = g_Ixx + g_Iyy
    harris_img = detM - 0.1 * (traceM ** 2)

    if show_img:
        show(harris_img)

    # select maxima (keypoints)
    max_img = ndimage.maximum_filter(harris_img, size=(7,7))
    harris_img[harris_img < 255] = harris_img.min()-1
    max_img = (harris_img == max_img).astype(np.uint8) * 255

    if show_img:
        show(max_img)

    # show keypoints
    for y in range(max_img.shape[1]):
        for x in range(max_img.shape[0]):
            if max_img[x, y] > 0:
                keypoints.append(cv2.KeyPoint(y, x, gradients[x, y]))
                
    img_kp = cv2.drawKeypoints(original, keypoints, None, color=(0,255,0), flags=0)
    
    if show_img:
        show(img_kp)

    return keypoints


def featureDescriptor(img, kp):
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location, not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    show(img2)

    # Harris
    kp_harris = orb.detect(img, None)
    kp_harris, des_harris = orb.compute(img, kp_harris)
    img3 = cv2.drawKeypoints(img, kp_harris, None, color=(0,255,0), flags=0)
    show(img3)
    
    # FAST
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    kp_fast = orb.detect(img, None)
    kp_fast, des_fast = orb.compute(img, kp_fast)
    img4 = cv2.drawKeypoints(img, kp_fast, None, color=(0,255,0), flags=0)
    show(img4)

    return des, kp, des_harris, kp_harris, des_fast, kp_fast


def SSDFeatureMatcher(des1, des2):
    matches = []

    distances = sp.spatial.distance.cdist(des1, des2, 'euclidean')

    for idx, list in enumerate(des1):
        # get minimum distance
        min = np.argmin(distances[idx])
        # create match object
        match = cv2.DMatch()
        match.queryIdx = idx
        match.trainIdx = int(min)
        match.distance = distances[idx, int(min)]
        matches.append(match)

    return matches


def RatioFeatureMatcher(des1, des2):
    matches = []

    distances = sp.spatial.distance.cdist(des1, des2, 'euclidean')

    for idx, list in enumerate(des1):
        # sort by distance
        sorted_idx = np.argsort(distances[idx])
        # get best match and second-best match
        best_match = distances[idx, sorted_idx[0]]
        second_best = distances[idx, sorted_idx[1]]

        match = cv2.DMatch()
        match.queryIdx = idx
        match.trainIdx = int(sorted_idx[0])
        ratio = best_match / float(second_best)
        match.distance = ratio
        matches.append(match)

    return matches


if __name__ == '__main__':
    # open image in gray scale
    img_path = './images/'

    filename = 'bernieSanders.jpg'
    img = cv2.imread(img_path + filename, cv2.IMREAD_GRAYSCALE)

    # check for success
    if img is None:
        print('Error: failed to open', filename)
        sys.exit()

    filename2 = 'bernie180.jpg'
    img2 = cv2.imread(img_path + filename2, cv2.IMREAD_GRAYSCALE)

    # check for success
    if img2 is None:
        print('Error: failed to open', filename2)
        sys.exit()

    keypoints = HarrisPointsDetector(img)
    des, kp, des_harris, kp_harris, des_fast, kp_fast = featureDescriptor(img, keypoints)

    keypoints2 = HarrisPointsDetector(img2)
    des2, kp2, des_harris2, kp_harris2, des_fast2, kp_fast2 = featureDescriptor(img2, keypoints2)

    ssd_matches = SSDFeatureMatcher(des_harris, des_harris2)
    ssd_matches_img = cv2.drawMatches(img, kp_harris, img2, kp_harris2, ssd_matches, None)
    show(ssd_matches_img)

    ratio_matches = RatioFeatureMatcher(des_harris, des_harris2)
    ratio_matches_img = cv2.drawMatches(img, kp_harris, img2, kp_harris2, ratio_matches, None)
    show(ratio_matches_img)