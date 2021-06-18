import cv2
import sys
import os
import numpy as np
import scipy as sp
import logging as log
import matplotlib.pyplot as plt

from scipy import ndimage, spatial


def show(img):
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    # plt.imshow(img, cmap='gray')
    # plt.show()


def import_img(name):
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)

    # check for success
    if img is None:
        print('Error: failed to open', name)
        sys.exit()
    
    return img


def HarrisPointsDetector(img, gauss_size, sigma, thresh, show_img=0):
    # keep a copy of the original image
    original = img.copy()
    # list of keypoints to return
    keypoints = []

    # smooth image
    if gauss_size:
        img = cv2.GaussianBlur(img, (gauss_size,gauss_size), sigma, borderType=cv2.BORDER_REFLECT)

    # calculate derivatives using Sobel operator
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)

    if show_img: show(Ix), show(Iy)

    # compute second derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    if show_img: show(Ixx), show(Iyy), show(Ixy)

    # apply Gaussian mask
    g_Ixx = cv2.GaussianBlur(Ixx, (5,5), 0.5, borderType=cv2.BORDER_REFLECT)
    g_Iyy = cv2.GaussianBlur(Iyy, (5,5), 0.5, borderType=cv2.BORDER_REFLECT)
    g_Ixy = cv2.GaussianBlur(Ixy, (5,5), 0.5, borderType=cv2.BORDER_REFLECT)

    if show_img: show(g_Ixx), show(g_Iyy), show(g_Ixy)

    # compute the gradient orientations
    gradients = np.arctan2(Ix, Iy) * 180 / np.pi

    # compute corner strength function
    detM = (g_Ixx * g_Iyy) - (g_Ixy ** 2)
    traceM = g_Ixx + g_Iyy
    harris_img = detM - 0.1 * (traceM ** 2)

    if show_img: show(harris_img)

    print(f'Max Harris response: {harris_img.max()}')
    log.info(f'Max Harris response: {harris_img.max()}')
    
    # select maxima (keypoints)
    max_img = ndimage.maximum_filter(harris_img, size=(7,7))
    max_img = ((harris_img == max_img) & (harris_img > thresh)).astype(np.uint8) * 255

    if show_img: show(max_img)

    # show keypoints
    for y in range(max_img.shape[1]):
        for x in range(max_img.shape[0]):
            if max_img[x, y] > 0:
                # giving a keypoint size, otherwise a random number is initialized
                keypoints.append(cv2.KeyPoint(y, x, 31, gradients[x, y]))
                
    img_kp = cv2.drawKeypoints(original, keypoints, None, color=(0,255,0), flags=0)
    
    if show_img: show(img_kp)

    print(f'Number of keypoints: {len(keypoints)}')
    log.info(f'Number of keypoints: {len(keypoints)}')

    return keypoints, img_kp


def featureDescriptor(img, kp):
    # Initiate ORB detector
    orb = cv2.ORB_create()

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location, not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    # show(img2)

    # Harris
    kp_harris = orb.detect(img, None)
    kp_harris, des_harris = orb.compute(img, kp_harris)
    img3 = cv2.drawKeypoints(img, kp_harris, None, color=(0,255,0), flags=0)
    # show(img3)
    
    # FAST
    orb.setScoreType(cv2.ORB_FAST_SCORE)
    kp_fast = orb.detect(img, None)
    kp_fast, des_fast = orb.compute(img, kp_fast)
    img4 = cv2.drawKeypoints(img, kp_fast, None, color=(0,255,0), flags=0)
    # show(img4)

    return des, kp, img2, des_harris, kp_harris, img3, des_fast, kp_fast, img4


def SSDFeatureMatcher(des1, des2):
    matches = []

    distances = spatial.distance.cdist(des1, des2, 'euclidean')

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

    distances = spatial.distance.cdist(des1, des2, 'euclidean')

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
        if match.distance < 0.9:
            matches.append(match)

    return matches


if __name__ == '__main__':

    img_path = './images/'
    res_path = './results/'
    find_params = 0
    plot = 0
    logname = 'find_params' if find_params else 'matching'

    logfile = res_path + logname + '.log'
    log.basicConfig(filename=logfile,
                    level=log.DEBUG,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
    log.getLogger('matplotlib.font_manager').disabled = True

    # Gaussian filters to test
    gauss_sizes = [None, 3, 5, 7, 9, 11]
    sigmas = [0, 0.5, 1, 2, 3]
    # thresholds to test
    thresholds = [1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5,
                  1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9]
    # number of combinations
    num_comb = len(gauss_sizes) * len(sigmas) * len(thresholds)

    # image to compare with
    bernie = 'bernieSanders.jpg'
    bernie_noext = os.path.splitext(bernie)[0]
    img = import_img(img_path+bernie)

    # find best params for bernie
    if find_params:
        print(f'***** {bernie_noext} *****')
        log.info(f'***** {bernie_noext} *****')
        for gs in gauss_sizes:
            for s in sigmas:
                for t in thresholds:
                    print(f'Gaussian filter: ({gs}, {gs}) \t Sigma: {s} \t Threshold: {t}')
                    log.info(f'Gaussian filter: ({gs}, {gs}) \t Sigma: {s} \t Threshold: {t}')
                    keypoints, img_kp = HarrisPointsDetector(img, gs, s, t)
                    cv2.imwrite(res_path + bernie_noext + '_' + str(gs) + '_' + 
                                str(s) + '_' + str(t) +'.jpg', img_kp)
                    if len(keypoints) == 0:
                        break
                if gs is None:
                    break
    else:   # best bernie
        best_gs = 5
        best_sigma = 0
        best_thresh = 1e7

        print(f'***** {bernie_noext} *****')
        log.info(f'***** {bernie_noext} *****')
        
        print(f'Gaussian filter: ({best_gs}, {best_gs}) \t Sigma: {best_sigma} \t Threshold: {best_thresh}')
        log.info(f'Gaussian filter: ({best_gs}, {best_gs}) \t Sigma: {best_sigma} \t Threshold: {best_thresh}')
        kp_bernie, img_kp = HarrisPointsDetector(img, best_gs, best_sigma, best_thresh)

        if plot:
            num_kp = []
            for t in thresholds:
                print(f'Threshold: {t}')
                keypoints, img_kp = HarrisPointsDetector(img, best_gs, best_sigma, t)
                num_kp.append(len(keypoints))
            plt.plot(thresholds, num_kp)
            plt.xscale('log')
            plt.xlabel('Threshold')
            plt.ylabel('Keypoints')
            plt.savefig(res_path + bernie)
            plt.close()

    # all other images
    filenames = os.listdir(img_path)
    filenames.remove(bernie)

    imgs = []
    for f in filenames:
        imgs.append(import_img(img_path+f))

    # find best params for all other images
    if find_params:
        for i in range(len(filenames)):
            image_noext = os.path.splitext(filenames[i])[0]
            print(f'***** {image_noext} *****')
            log.info(f'***** {image_noext} *****')
            for gs in gauss_sizes:
                for s in sigmas:
                    for t in thresholds:
                        print(f'Gaussian filter: ({gs}, {gs}) \t Sigma: {s} \t Threshold: {t}')
                        log.info(f'Gaussian filter: ({gs}, {gs}) \t Sigma: {s} \t Threshold: {t}')
                        keypoints, img_kp = HarrisPointsDetector(imgs[i], gs, s, t)
                        cv2.imwrite(res_path + image_noext + '_' + str(gs) + '_' + 
                                    str(s) + '_' + str(t) +'.jpg', img_kp)
                        if len(keypoints) == 0:
                            break
                    if gs is None:
                        break
        exit()
    else:   # best for others
        kp_others = []
        best_gss = [5, 5, 9, 5, 5, 7, None, 5, 7]
        best_sigmas = [0, 0.5, 2, 0, 2, 2, None, 0, 3]
        best_threshs = [1e7, 1e9, 5e5, 1e2, 5e6, 1e6, 5e7, 1e7, 1e6]

        for i in range(len(filenames)):
            image_noext = os.path.splitext(filenames[i])[0]
            print(f'***** {image_noext} *****')
            log.info(f'***** {image_noext} *****')

            print(f'Gaussian filter: ({best_gss[i]}, {best_gss[i]}) \t Sigma: {best_sigmas[i]} \t Threshold: {best_threshs[i]}')
            log.info(f'Gaussian filter: ({best_gss[i]}, {best_gss[i]}) \t Sigma: {best_sigmas[i]} \t Threshold: {best_threshs[i]}')
            kp, img_kp = HarrisPointsDetector(imgs[i], best_gss[i], best_sigmas[i], best_threshs[i])
            kp_others.append(kp)

            if plot:
                num_kp = []
                for t in thresholds:
                    print(f'Threshold: {t}')
                    keypoints, img_kp = HarrisPointsDetector(imgs[i], best_gss[i], best_sigmas[i], t)
                    num_kp.append(len(keypoints))
                plt.plot(thresholds, num_kp)
                plt.xscale('log')
                plt.xlabel('Threshold')
                plt.ylabel('Keypoints')
                plt.savefig(res_path + filenames[i])
                plt.close()

    # best for each of the other images

    # compute points using ORB
    des, kp, img_mine, des_harris, kp_harris, img_harris, des_fast, kp_fast, img_fast = featureDescriptor(img, kp_bernie)
    cv2.imwrite(res_path + bernie_noext + '_harris.jpg', img_mine)
    cv2.imwrite(res_path + bernie_noext + '_ORBharris.jpg', img_harris)
    cv2.imwrite(res_path + bernie_noext + '_ORBfast.jpg', img_fast)

    dess, des_harriss, des_fasts = [], [], []
    kps, kp_harriss, kp_fasts = [], [], []

    for i in range(len(filenames)):
        d, k, i_1, dh, kh, i_2, df, kf, i_3 = featureDescriptor(imgs[i], kp_others[i])
        dess.append(d)
        kps.append(k)
        des_harriss.append(dh)
        kp_harriss.append(kh)
        des_fasts.append(df)
        kp_fasts.append(kf)

        image_noext = os.path.splitext(filenames[i])[0]
        cv2.imwrite(res_path + image_noext + '_harris.jpg', i_1)
        cv2.imwrite(res_path + image_noext + '_ORBharris.jpg', i_2)
        cv2.imwrite(res_path + image_noext + '_ORBfast.jpg', i_3)

    # draw matches with all three techniques
    for i in range(len(filenames)):
        image_noext = os.path.splitext(filenames[i])[0]

        ssd_matches = SSDFeatureMatcher(des, dess[i])
        ssd_matches_img = cv2.drawMatches(img, kp, imgs[i], kps[i], ssd_matches, None)
        cv2.imwrite(res_path + image_noext + '_harris_SSDmatches.jpg', ssd_matches_img)

        ratio_matches = RatioFeatureMatcher(des, dess[i])
        ratio_matches_img = cv2.drawMatches(img, kp, imgs[i], kps[i], ratio_matches, None)
        cv2.imwrite(res_path + image_noext + '_harris_ratiomatches.jpg', ratio_matches_img)

    for i in range(len(filenames)):
        image_noext = os.path.splitext(filenames[i])[0]

        ssd_matches = SSDFeatureMatcher(des_harris, des_harriss[i])
        ssd_matches_img = cv2.drawMatches(img, kp_harris, imgs[i], kp_harriss[i], ssd_matches, None)
        cv2.imwrite(res_path + image_noext + '_ORBharris_SSDmatches.jpg', ssd_matches_img)

        ratio_matches = RatioFeatureMatcher(des_harris, des_harriss[i])
        ratio_matches_img = cv2.drawMatches(img, kp_harris, imgs[i], kp_harriss[i], ssd_matches, None)
        cv2.imwrite(res_path + image_noext + '_ORBharris_ratiomatches.jpg', ratio_matches_img)

    for i in range(len(filenames)):
        image_noext = os.path.splitext(filenames[i])[0]

        ssd_matches = SSDFeatureMatcher(des_fast, des_fasts[i])
        ssd_matches_img = cv2.drawMatches(img, kp_fast, imgs[i], kp_fasts[i], ssd_matches, None)
        cv2.imwrite(res_path + image_noext + '_ORBfast_SSDmatches.jpg', ssd_matches_img)

        ratio_matches = RatioFeatureMatcher(des_fast, des_fasts[i])
        ratio_matches_img = cv2.drawMatches(img, kp_fast, imgs[i], kp_fasts[i], ssd_matches, None)
        cv2.imwrite(res_path + image_noext + '_ORBfast_ratiomatches.jpg', ratio_matches_img)