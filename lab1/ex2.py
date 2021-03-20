'''
By convolving the original image with the appropriate kernels, compute the
horizontal and vertical gradient images, and then find the edge strength image
given by the gradient magnitude (combined image).
'''
import cv2
import numpy as np


def convolve(image, kernel):

    # convolved image (both the x and y axis loose 2 pixels because we are not padding)
    # we do can hardcode the dimensions because we know we are using a 3 × 3 kernel
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
prewitt_grad_img = (prewitt_grad_img / np.max(prewitt_grad_img)) * 255
sobel_grad_img = np.sqrt(np.power(sobel_x_img, 2) + np.power(sobel_y_img, 2))
sobel_grad_img = (sobel_grad_img / np.max(sobel_grad_img)) * 255

# save images
cv2.imwrite('prewitt_x_' + filename, prewitt_x_img)
cv2.imwrite('prewitt_y_' + filename, prewitt_y_img)
cv2.imwrite('sobel_x_' + filename, sobel_x_img)
cv2.imwrite('sobel_y_' + filename, sobel_y_img)
cv2.imwrite('prewitt_grad_' + filename, prewitt_grad_img)
cv2.imwrite('sobel_grad_' + filename, sobel_grad_img)