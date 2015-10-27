#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp> 

#include <ctime>

#include "cnnFace.h"

using namespace std;
using namespace cv;

void main(int argc, char* argv)
{
	Mat imgFace1 = imread("D:\\test\\Aaron_Peirsol_0002.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgFace2 = imread("D:\\test\\Aaron_Peirsol_0003.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	char* modelPath = "D:\\code\\cnnFace\\model\\cnnFace.bin";

	clock_t start, end;
	double time;

	start = clock();
	cnnFace cnn(modelPath, 44);
	if (cnn.cnnFaceInit() != 0) {
		return;
	}
	cnn.faceVerification(imgFace1, imgFace2);
	end = clock();
	time = (double)(end -  start) / CLOCKS_PER_SEC;

	cout << "The score is " << cnn.getScore() << "\nTime is " << time << endl;

	cnn.~cnnFace();
}
