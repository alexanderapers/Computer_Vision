//// main.cpp
//
//
//
#include "precomp.h"


int main(int argc, char** argv)
{
    //ScreenCapture::streamWebcamFeed();

    CameraCalibration::calibrate(argc, argv);

    return 0;
}