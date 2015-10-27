#ifndef _CNN_FACE_H
#define _CNN_FACE_H

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp> 

#include "cnn.h"

using namespace std;
using namespace cv;

class cnnFace{
public:
	cnnFace(const char* modelPath, const int layerIdx, const int len):
		_modelPath(modelPath),
	    _layerIdx(layerIdx), 
	    _len(len){};

	~cnnFace() {
		_cnnFaceNet.~Net();
	};
	
	int cnnFaceInit();
	int getFeature(Mat &faceImg, float* feat);
	float getScore(float* feat1, float* feat2);
	//int faceVerification(Mat &faceData1, Mat &faceData2);
	
private:
	const char* _modelPath;
	const int _layerIdx;
	
	int _len;
	Net _cnnFaceNet;
	
};
#endif