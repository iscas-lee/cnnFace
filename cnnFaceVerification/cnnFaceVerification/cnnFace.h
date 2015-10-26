#include <stdio.h>
#include <stdlib.h>

#include "cnn.h"

class cnnFace{
public:
	cnnFace(const float* faceData1, const float* faceData2, const char* modelPath):
		_faceData1(faceData1),
		_faceData2(faceData2),
		_modelPath(modelPath) {};

	~cnnFace() {
		if (_faceData1 != NULL)
			delete _faceData1;
		if (_faceData2 != NULL)
			delete _faceData2;
	};
	
	int cnnInit();
	int faceVerification();

private:
	const float* _faceData1;
	const float* _faceData2;
	const char* _modelPath;

	float score;

};