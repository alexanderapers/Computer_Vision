#pragma once
#include "common.h"

#include <iostream>
#include <sstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <format>
#include <Windows.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

// ----------------------------------------------------------------------------------
// PROJECT SOURCE HEADERS
// ----------------------------------------------------------------------------------

// Screen capturing
#include "src/screen_capture/screen_capture.h"

// Calibration
#include "src/calibration/calibration_settings.h"
#include "src/calibration/camera_calibration.h"
