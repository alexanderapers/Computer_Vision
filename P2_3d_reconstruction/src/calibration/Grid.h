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
#include <cassert>
#include <iostream>

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

class Grid
{
public:
	vector<Point> m_four_Corners;
	Grid(vector<Point> four_Corners, Size grid_size);
	vector<Point>* getAllPoints();
	void interpolate(Point p1, Point p2, vector<Point>* points);

private:
	Size m_grid_size;
};
}