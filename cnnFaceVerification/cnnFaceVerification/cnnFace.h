#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp> 

#include "cnn.h"

using namespace std;
using namespace cv;

class cnnFace{
public:
	cnnFace(Mat &faceData1, Mat &faceData2, const char* modelPath, const int layerNum):
		_faceData1(faceData1),
		_faceData2(faceData2),
		_modelPath(modelPath),
	    _layerNum(layerNum) {};

	~cnnFace() {
		_cnnFaceNet.~Net();
	};
	
	int cnnFaceInit();
	int faceVerification();
	float getScore() {return _score; }

private:
	const Mat _faceData1;
	const Mat _faceData2;
	const char* _modelPath;
	const int _layerNum;

	Net _cnnFaceNet;
	float _score;

};