#pragma once

#include "precomp.h"

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

#include "Grid.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

	Grid::Grid(vector<Point> four_Corners, Size grid_size)
		: m_four_Corners(four_Corners),
		m_grid_size(grid_size)
	{}

	vector<Point>* Grid::getAllPoints()
	{
		vector<Point>* all_points = new vector<Point>();

		Point a = m_four_Corners.at(0);
		Point b = m_four_Corners.at(1);
		Point c = m_four_Corners.at(2);
		Point d = m_four_Corners.at(3);

		double delta_x_ad = d.x - a.x;
		delta_x_ad /= m_grid_size.height - 1;

		double delta_y_ad = d.y - a.y;
		delta_y_ad /= m_grid_size.height - 1;

		double delta_x_bc = c.x - b.x;
		delta_x_bc /= m_grid_size.height - 1;

		double delta_y_bc = c.y - b.y;
		delta_y_bc /= m_grid_size.height - 1;


		for (int i = 0; i < m_grid_size.height; i++)
		{
			Point p(a.x + i * delta_x_ad, a.y + i * delta_y_ad);
			Point q(b.x + i * delta_x_bc, b.y + i * delta_y_bc);

			interpolate(p, q, all_points);
		}
		
		return all_points;
	}

	void Grid::interpolate(Point p1, Point p2, vector<Point>* points)
	{
		double delta_x = p2.x - p1.x;
		delta_x /= m_grid_size.width - 1;

		double delta_y = p2.y - p1.y;
		delta_y /= m_grid_size.width - 1;

		for (int i = 0; i < m_grid_size.width; i++)
		{
			Point p(p1.x + i * delta_x, p1.y + i * delta_y);
			points->push_back(p);
		}
	}

	
}