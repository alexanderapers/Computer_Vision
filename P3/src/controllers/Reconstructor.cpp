/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "precomp.h"
#include "Reconstructor.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <cassert>
#include <iostream>
#include <filesystem>

#include "../utilities/General.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs) :
				m_cameras(cs),
				m_height(2048),
				m_step(32)
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const size_t edge = 2 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

	initialize();
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(static) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	cout << "done!" << endl;
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	m_visible_voxels.clear();
	std::vector<Voxel*> visible_voxels;

	int v;
#pragma omp parallel for schedule(static) private(v) shared(visible_voxels)
	for (v = 0; v < (int) m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);
		}
	}

	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());

	cluster();
}

void Reconstructor::cluster()
{
	m_groundCoordinates = vector<Point2f>(m_visible_voxels.size());
	vector<int> bestLabels(m_visible_voxels.size());

	for (int i = 0; i < (int)m_visible_voxels.size(); i++)
	{
		m_groundCoordinates[i] = Point2f(m_visible_voxels[i]->x, m_visible_voxels[i]->y);
	}

	int K = 4;
	int reruns = 10;
	vector<Point2f> centers(K);
	TermCriteria termination_criteria;
	termination_criteria.epsilon = 0.1; //??
	double compactness = kmeans(m_groundCoordinates, K, bestLabels, termination_criteria, reruns, KmeansFlags(), centers);
	//cout << std::format("Clustered {0} times with K={1} final compactness measure of {2}", reruns, K, compactness) << endl;
	m_clusterLabels = bestLabels;
	m_centers = centers;

	vector<vector<int>> clusters = vector<vector<int>>(K);

	for (int i = 0; i < m_visible_voxels.size(); i++)
	{
		clusters[m_clusterLabels[i]].push_back(i);
	}

	m_clusters = clusters;
}

void Reconstructor::buildOfflineColorModels()
{
	int camera = 3;
	Mat current_frame = m_cameras[camera]->getFrame();
	Mat mask = Mat(current_frame.size(), CV_8UC3, Scalar(0, 0, 0));
	if (!std::filesystem::exists("GMMS"))
	{
		std::filesystem::create_directory("./GMMS");
	}
	imwrite("GMMS/frame.png", current_frame);

	cvtColor(current_frame, current_frame, CV_BGR2HSV); // convert to HSV color space
	vector<Ptr<EM>> GMMS;

	// for each cluster
	for (int k = 0; k < (int)m_clusters.size(); k++)
	{
		std::unordered_set<Point> points;
		Mat colors;

		// for each voxel in that cluster
		for (int i = 0; i < (int)m_clusters[k].size(); i++)
		{
			Voxel* voxel = m_visible_voxels[m_clusters[k][i]];

			if (voxel->z > LOWER_GMM_LIMIT && voxel->z < UPPER_GMM_LIMIT && voxel->valid_camera_projection[camera])
			{ 
				Point point = voxel->camera_projection[camera];

				if (!points.contains(point))
				{
					Vec3b color = current_frame.at<Vec3b>(point);
					points.insert(point);
					Mat col(1, 3, CV_8UC1);
					for (int m = 0; m < 3; m++)
						col.at<char>(0, m) = color[m];
					colors.push_back(col);
					mask.at<Vec3b>(point) = col;
				}
			}	
		}

		Ptr<EM> GMM = EM::create();

		// set cluster numbers, covariance types and termination criteria
		GMM->setClustersNumber(3);
		GMM->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);
		GMM->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));

		// train GMM
		GMM->trainEM(colors, noArray(), noArray(), noArray());

		interpretGMM(k, GMM);

		GMM->save(std::format("GMMS/GMM_{}.yaml", k+1));

		GMMS.push_back(GMM);
	}

	cvtColor(mask, mask, CV_HSV2BGR);
	imwrite("GMMS/masks.png", mask);
}

void Reconstructor::interpretGMM(int GMM_number, Ptr<EM> GMM)
{
	Mat means = GMM->getMeans();

	Vec3b a = means.row(0);
	Vec3b b = means.row(1);
	Vec3b c = means.row(2);
	Mat x = Mat(100, 500, CV_8UC3, a);
	Mat y = Mat(100, 500, CV_8UC3, b);
	Mat r = Mat(100, 500, CV_8UC3, c);
	Mat z;
	Mat p;
	hconcat(x, y, z);
	hconcat(z, r, p);
	cvtColor(p, p, CV_HSV2BGR);
	imwrite(std::format("GMMS/cluster_{}_means.png", GMM_number + 1), p);
	waitKey(10);

	vector<Mat> covs;
	GMM->getCovs(covs);

	FileStorage fs_means(std::format("GMMS/cluster_{}_means.xml", GMM_number + 1), FileStorage::WRITE);
	fs_means << "means" << means;
	fs_means.release();

	FileStorage fs_covs(std::format("GMMS/cluster_{}_covs.xml", GMM_number + 1), FileStorage::WRITE);
	for (int i = 0; i < covs.size(); i++)
	{
		fs_covs << std::format("covs{}", i) << covs[i];
	}

	fs_covs.release();
}

} /* namespace nl_uu_science_gmt */
