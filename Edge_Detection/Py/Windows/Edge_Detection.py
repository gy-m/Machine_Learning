import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def convolve(img, kernel):

    image_matrix_calculated = np.zeros_like(img)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            window_im = img[i - 1:i + 2, j - 1:j + 2]
            prod_sum = np.sum(window_im * kernel)
            image_matrix_calculated[i, j] = prod_sum
    return image_matrix_calculated


def plot_imgs(imgs, fig_rows, fig_cols):
    # create figure (will hold the sub plots of the images)
    fig = plt.figure()

    for i in range(len(imgs)):
        sub_plot_num = i
        fig.add_subplot(fig_rows, fig_cols, sub_plot_num+1)
        if i == 0:
            # for colored images (only the firt image is colored)
            plt.imshow(imgs[i])
        else:
            plt.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title("Image: " + str(i + 1))

    plt.show()


if __name__ == '__main__':

    imgs = []
    print("Starting Time of the program: " + time.ctime().format())

    # Read image from your local file system
    image_original = mpimg.imread('image_original.jpg')
    imgs.append(image_original)

    # Convert color image to grayscale
    image_grayscale = cv.cvtColor(image_original, cv.COLOR_BGR2GRAY).astype(np.float32) / 255
    imgs.append(image_grayscale)

    # create a kernel named "Laplacian Filter"
    laplacian_Filter = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float32)

    image_edges_Laplacian = convolve(image_grayscale, laplacian_Filter)
    imgs.append(image_edges_Laplacian)
    print("Time of creation of Laplacian image: " + time.ctime().format())

    # create a kernel named "Sobel Filter"
    Sobel_Filter = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float32)

    image_edges_Sobel = convolve(image_grayscale, Sobel_Filter)
    imgs.append(image_edges_Sobel)
    print("Time of creation of Sobel image: " + time.ctime().format())

    plot_imgs(imgs, fig_rows=1, fig_cols=len(imgs))


