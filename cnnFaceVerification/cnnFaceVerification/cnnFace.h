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
	cnnFace(const char* modelPath, const int layerIdx):
		_modelPath(modelPath),
	    _layerIdx(layerIdx) {};

	~cnnFace() {
		_cnnFaceNet.~Net();
	};
	
	int cnnFaceInit();
	int faceVerification(Mat &faceData1, Mat &faceData2);
	float getScore() {return _score; }

private:
	const char* _modelPath;
	const int _layerIdx;

	Net _cnnFaceNet;
	float _score;
};

#endif