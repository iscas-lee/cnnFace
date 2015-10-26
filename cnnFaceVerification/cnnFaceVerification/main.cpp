#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp> 

#include "cnnFace.h"

using namespace std;
using namespace cv;

void main(int argc, char* argv)
{
	Mat imgFace1 = imread("D:\\test\\Aaron_Eckhart_0001.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgFace2 = imread("D:\\test\\Aaron_Guiel_0001.bmp", CV_LOAD_IMAGE_GRAYSCALE);
}
