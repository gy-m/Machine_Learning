// C++ program to demonstrate the
// above approach
#include <iostream>
#include <opencv2/core/core.hpp>

// Library to include for
// drawing shapes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

// Driver Code
int main(int argc, char** argv)
{
	// Create a blank image of size
	// (500 x 500) with white background
	// (B, G, R) : (255, 255, 255)
	Mat image = imread("image_original.jfif", IMREAD_COLOR);

	// Check if the image is
	// created successfully.
	if (!image.data) {
		cout << "Could not open or"
			<< " find the image" << std::endl;
		return 0;
	}

	// Show our image inside a window.
	namedWindow("Output", WINDOW_NORMAL);
	resizeWindow("Output", 800, 800);
	imshow("Output", image);
	waitKey(0);

	return 0;
}
