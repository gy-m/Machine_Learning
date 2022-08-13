// C++ program to demonstrate the
// above approach
#include <iostream>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Library to include for
// drawing shapes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;


void show_image(string image_name, Mat img)
{
	namedWindow("xxx", WINDOW_NORMAL);
	resizeWindow("xxx", 800, 800);
	imshow("xxx", img);
	waitKey(0);
}

// Driver Code
int main(int argc, char** argv)
{
	// Create an image of size
	// (B, G, R) : (255, 255, 255)
	Mat image_original = imread("image_original.jfif", IMREAD_COLOR);

	// Check if the image is created successfully.
	if (!image_original.data) {
		cout << "Could not open or find the image" << std::endl;
		return 0;
	}

	// Show our image inside a window (source image)
	namedWindow("Original", WINDOW_NORMAL);
	resizeWindow("Original", 800, 800);
	imshow("Original", image_original);
	waitKey(0);

	// converte to a greyscale image
	Mat image_grey_scale;
	cvtColor(image_original, image_grey_scale, COLOR_BGR2GRAY);
	// Show our image inside a window (greyscale image)
	/*
	namedWindow("Greyscale", WINDOW_NORMAL);
	resizeWindow("Greyscale", 800, 800);
	imshow("Greyscale", image_grey_scale);
	waitKey(0);
	*/

	show_image("Greyscale", image_grey_scale);

	// converte to a sobel image, meaning soble filter implemented (sobel image)
	Mat sobelx;
	Sobel(image_grey_scale, sobelx, CV_32F, 1, 0);
	double minVal, maxVal;
	minMaxLoc(sobelx, &minVal, &maxVal);		//find minimum and maximum intensities
	cout << "minVal : " << minVal << endl << "maxVal : " << maxVal << endl;

	Mat image_sobel;
	sobelx.convertTo(image_sobel, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

	// Show our image inside a window (sobel image)
	namedWindow("Sobel", WINDOW_NORMAL);
	resizeWindow("Sobel", 800, 800);
	imshow("Sobel", image_sobel);
	waitKey(0);

	// converte to a laplacian image, meaning Laplacian filter implemented (laplacian image)
	Mat src, src_gray, dst;
	Mat image_laplacian;
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Laplacian(image_grey_scale, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(dst, image_laplacian);

	// Show our image inside a window (laplacian image)
	namedWindow("Laplacian", WINDOW_NORMAL);
	resizeWindow("Laplacian", 800, 800);
	imshow("Laplacian", image_laplacian);
	waitKey(0);

	return 0;
}
