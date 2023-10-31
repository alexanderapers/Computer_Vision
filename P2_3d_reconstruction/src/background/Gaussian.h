#pragma once

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <stddef.h>
#include "../controllers/Video.h"
#include <cassert>
#include <iostream>

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{
class Gaussian
{
public:
	Gaussian(Video* background_vid);
	Video* m_background_vid;
	tuple<Mat, Mat> calculateGaussian();
	int m_frame_count;
	int m_width;
	int m_height;
};

}
