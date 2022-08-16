#include "opencv2/core/core.hpp"
#include "opencv2/core/mat.hpp"
#include <opencv2/core/mat.inl.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"


#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;



void show_image(string image_name, Mat img)
{
	namedWindow(image_name, WINDOW_NORMAL);
	resizeWindow(image_name, 800, 800);
	imshow(image_name, img);
}


// Driver Code
int main(int argc, char** argv)
{
	// Create an image of size
	// (B, G, R) : (255, 255, 255)
	Mat image_original = imread("image_original.jpg", IMREAD_COLOR);

	// Check if the image is created successfully.
	if (!image_original.data) {
		cout << "Could not open or find the image" << std::endl;
		return 0;
	}

	// converte to a greyscale image
	Mat image_grey_scale;
	cvtColor(image_original, image_grey_scale, COLOR_BGR2GRAY);

	// converte to a sobel image, meaning soble filter implemented (sobel image)
	Mat sobelx;
	Sobel(image_grey_scale, sobelx, CV_32F, 1, 0);
	double minVal, maxVal;
	minMaxLoc(sobelx, &minVal, &maxVal);		//find minimum and maximum intensities
	cout << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;

	Mat image_sobel;
	sobelx.convertTo(image_sobel, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

	// converte to a laplacian image, meaning Laplacian filter implemented (laplacian image)
	Mat src, src_gray, dst;
	Mat image_laplacian;
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Laplacian(image_grey_scale, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(dst, image_laplacian);

	// Show our images
	show_image("Original", image_original);
	show_image("Greyscale", image_grey_scale);
	show_image("Sobel", image_sobel);
	show_image("Laplacian", image_laplacian);
	waitKey(0);

	return 0;
}
