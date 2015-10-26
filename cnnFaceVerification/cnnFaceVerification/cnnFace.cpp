#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp> 

#include "cnn.h"
#include "cnnFace.h"

using namespace std;
using namespace cv;

int cnnFace::cnnFaceInit() {

	int ret;  
	ret = cnnFaceNet.LoadFromFile(_modelPath);
	if (ret != 0) {
		cout << "[Error] Loading model is error!\n";
	}
	return ret;
	}

int cnnFace::faceVerification() {
	int w1, h1, c1;
	int w2, h2, c2;

	w1 = _faceData1.rows;
	h1 = _faceData1.cols;
	c1 = _faceData1.channels();

	w2 = _faceData2.rows;
	h2 = _faceData2.cols;
	c2 = _faceData2.channels();

	float* data1 = (float *)malloc(w1 * h1 * c1 * sizeof(float));
	float* data2 = (float *)malloc(w2 * h2 * c2 * sizeof(float));


	return 0;
}