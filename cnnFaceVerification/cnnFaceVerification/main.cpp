#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp> 

#include <ctime>

#include "cnnFace.h"

#define _MAX_LEN 1024

using namespace std;
using namespace cv;

void main(int argc, char* argv[])
{
	char *filename1 = (char *)malloc(_MAX_LEN*sizeof(char));
	char *filename2 = (char *)malloc(_MAX_LEN*sizeof(char));
	char *modelPath  = (char *)malloc(_MAX_LEN*sizeof(char));
	filename1 = argv[1];
	filename2 = argv[2];
	modelPath = argv[3];
	
	Mat imgFace1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgFace2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);
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
