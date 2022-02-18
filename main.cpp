//// main.cpp
//
//
//
#include "precomp.h"


int main(int argc, char** argv)
{
    //ScreenCapture::streamWebcamFeed();
    
    // get recalibration error of using all images
    CameraCalibration::calibrate(argc, argv);

    // for image in calibimages run calibration without this images
        // get recalibration error using n - 1 images
        // if new error < old error:
            // disregard new



    return 0;
}