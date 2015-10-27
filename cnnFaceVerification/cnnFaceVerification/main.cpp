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
	const int layerIdx = 44;
	const int len = 256;

	float* feat1 = (float*)malloc(len*sizeof(float));
	float* feat2 = (float*)malloc(len*sizeof(float));

	clock_t start, end;
	double time;

	start = clock();
	cnnFace cnn(modelPath, layerIdx, len);
	if (cnn.cnnFaceInit() != 0) {
		return;
	}


	cnn.getFeature(imgFace1, feat1);
	cnn.getFeature(imgFace2, feat2);

	float score = cnn.getScore(feat1, feat2);

	end = clock();
	time = (double)(end -  start) / CLOCKS_PER_SEC;

	cout << "The score is " << score << "\nTime is " << time << endl;

	cnn.~cnnFace();
	free(feat1);
	free(feat2);
}
