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
	/*char *filename1 = (char *)malloc(_MAX_LEN*sizeof(char));
	char *filename2 = (char *)malloc(_MAX_LEN*sizeof(char));
	char *modelPath  = (char *)malloc(_MAX_LEN*sizeof(char));
	filename1 = argv[1];
	filename2 = argv[2];
	modelPath = argv[3];
	
	Mat imgFace1 = imread(filename1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgFace2 = imread(filename2, CV_LOAD_IMAGE_GRAYSCALE);*/


	Mat imgFace1 = imread("D:\\test\\Aaron_Peirsol_0002.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgFace2 = imread("D:\\test\\Aaron_Peirsol_0003.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	char* modelPath = "D:\\code\\cnnFace\\model\\cnnFace.bin";

	// VGG
	/*resize(imgFace1, imgFace1, Size(224, 224), CV_INTER_LINEAR);
	resize(imgFace2, imgFace2, Size(224, 224), CV_INTER_LINEAR);*/

	const int layerIdx = 44;
	const int len = 256;

	float* feat1 = (float*)malloc(len*sizeof(float));
	float* feat2 = (float*)malloc(len*sizeof(float));

	clock_t start, end;
	double time;


	cnnFace cnn(modelPath, layerIdx, len);
	if (cnn.cnnFaceInit() != 0) {
		return;
	}

	start = clock();
	//cnn.getFeature(imgFace1, feat1);

	int w = imgFace1.rows;
	int h = imgFace1.cols;
	int c = imgFace1.channels();
	int cnt = w * h * c;

	float* data1 = (float *)malloc(cnt * sizeof(float));
	for (int i = 0; i < cnt; i++) {
		data1[i] = static_cast<float>(imgFace1.data[i]) / 255.0;
	}
	cnn.getFeature(data1, feat1, w, h, c);
	end = clock();
	


	w = imgFace2.rows;
	h = imgFace2.cols;
	c = imgFace2.channels();
	cnt = w * h * c;

	float* data2 = (float *)malloc(cnt * sizeof(float));
	for (int i = 0; i < cnt; i++) {
		data2[i] = static_cast<float>(imgFace2.data[i]) / 255.0;
	}
	cnn.getFeature(data2, feat2, w, h, c);

	float score = cnn.getScore(feat1, feat2);

	time = (double)(end -  start) / CLOCKS_PER_SEC;
	cout << "The score is " << score << "\nTime is " << time << endl;

	cnn.~cnnFace();
	free(feat1);
	free(feat2);

	system("pause");
}
