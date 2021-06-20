'''
Write a short function that performs the convolution between an image, and a
3 × 3 structuring element, by performing an explicit looping over the image
pixels. You should pad the edges of the input image with zeros to deal with the
edges and corners of the original image.
'''
import cv2
import numpy as np


def convolve(image, kernel):

    kernel_sum = kernel.sum()

    # pad image with zeros
    pad_img = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    pad_img[1:image.shape[0]+1, 1:image.shape[1]+1] = image

    # convolved image (here we know the dimension is the same as the original
    # because we are using a 3 × 3 kernel with zero padding and stride of 1)
    conv_img = np.zeros((image.shape[0], image.shape[1]))

    # convolve image
    for y in range(image.shape[1]):  # rows
        for x in range(image.shape[0]):  # columns
            # matrix * matrix is an element-wise multiplication
            conv_img[x, y] = (kernel * pad_img[x: x+kernel.shape[0], y: y+kernel.shape[1]]).sum()

    return conv_img/kernel_sum


# open image in gray scale
filename = 'kitty.bmp'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# check for success
if img is None:
    print('Error: failed to open', filename)
    sys.exit()

# kernels
avg_kernel = np.ones((3, 3))
wavg_kernel = np.array([[.5, 1, .5], [1, 2, 1], [.5, 1, .5]])

# perform convolutions
avg_img = convolve(img, avg_kernel)
wavg_img = convolve(img, wavg_kernel)

# save images
cv2.imwrite('./results/avg_' + filename, avg_img)
cv2.imwrite('./results/wavg_' + filename, wavg_img)