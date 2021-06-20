'''
Perform thresholding of the edge strength image, and hence display the major
edges of the image.
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def convolve(image, kernel):

    # convolved image (both the x and y axis loose 2 pixels because we are not padding)
    # we can hardcode the dimensions because we know we are using a 3 × 3 kernel
    conv_img = np.zeros((image.shape[0]-2, image.shape[1]-2))

    # convolve image
    for y in range(conv_img.shape[1]):  # rows
        for x in range(conv_img.shape[0]):  # columns
            # matrix * matrix is an element-wise multiplication
            conv_img[x, y] = (kernel * image[x: x+kernel.shape[0], y: y+kernel.shape[1]]).sum()

    return (conv_img / np.max(conv_img)) * 255  # range from 0 to 255


# open image in gray scale
filename = 'kitty.bmp'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# check for success
if img is None:
    print('Error: failed to open', filename)
    sys.exit()

# kernels
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# perform convolutions
prewitt_x_img = convolve(img, prewitt_x)
prewitt_y_img = convolve(img, prewitt_y)
sobel_x_img = convolve(img, sobel_x)
sobel_y_img = convolve(img, sobel_y)

# gradient magnitude
prewitt_grad_img = np.sqrt(np.power(prewitt_x_img, 2) + np.power(prewitt_y_img, 2))
prewitt_grad_img = ((prewitt_grad_img / np.max(prewitt_grad_img)) * 255).astype('uint8')
sobel_grad_img = np.sqrt(np.power(sobel_x_img, 2) + np.power(sobel_y_img, 2))
sobel_grad_img = ((sobel_grad_img / np.max(sobel_grad_img)) * 255).astype('uint8')

# image histograms
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist = hist.reshape(256)
plt.bar(np.linspace(0,255,256), hist)
plt.title('Historgram for Intensities in the Original Image')
plt.ylabel('Frequency')
plt.xlabel('Grey Level')
plt.show()

hist = cv2.calcHist([prewitt_grad_img], [0], None, [256], [0, 256])
hist = hist.reshape(256)
plt.bar(np.linspace(0,255,256), hist)
plt.title('Histogram for Prewitt Edge-Strength Image')
plt.ylabel('Frequency')
plt.xlabel('Grey Level')
plt.show()

hist = cv2.calcHist([sobel_grad_img], [0], None, [256], [0, 256])
hist = hist.reshape(256)
plt.bar(np.linspace(0,255,256), hist)
plt.title('Histogram for Sobel Edge-Strength Image')
plt.ylabel('Frequency')
plt.xlabel('Grey Level')
plt.show()

# threshold the gradient magnitude
threshold = 60

img_thresh = np.zeros((img.shape[0], img.shape[1]))
for y in range(img.shape[1]):
    for x in range(img.shape[0]):
        if img[x, y] > threshold:
            img_thresh[x, y] = 255

prewitt_thresh = np.zeros((prewitt_grad_img.shape[0], prewitt_grad_img.shape[1]))
for y in range(prewitt_grad_img.shape[1]):
    for x in range(prewitt_grad_img.shape[0]):
        if prewitt_grad_img[x, y] > threshold:
            prewitt_thresh[x, y] = 255

sobel_thresh = np.zeros((sobel_grad_img.shape[0], sobel_grad_img.shape[1]))
for y in range(sobel_grad_img.shape[1]):
    for x in range(sobel_grad_img.shape[0]):
        if sobel_grad_img[x, y] > threshold:
            sobel_thresh[x, y] = 255

# save images
cv2.imwrite('./results/prewitt_thresh_' + filename, prewitt_thresh)
cv2.imwrite('./results/sobel_thresh_' + filename, sobel_thresh)
cv2.imwrite('./results/img_thresh_' + filename, img_thresh)