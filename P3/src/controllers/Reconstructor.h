/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#pragma once

#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>

#include "Camera.h"

namespace nl_uu_science_gmt
{

class Reconstructor
{
public:
	/*
	 * Voxel structure
	 * Represents a 3D pixel in the half space
	 */
	struct Voxel
	{
		int x, y, z;                               // Coordinates
		cv::Scalar color;                          // Color
		std::vector<cv::Point> camera_projection;  // Projection location for camera[c]'s FoV (2D)
		std::vector<int> valid_camera_projection;  // Flag if camera projection is in camera[c]'s FoV
	};

private:

	const std::vector<Camera*> &m_cameras;  // vector of pointers to cameras
	const int m_height;                     // Cube half-space height from floor to ceiling
	const int m_step;                       // Step size (space between voxels)

	std::vector<cv::Point3f*> m_corners;    // Cube half-space corner locations

	size_t m_voxels_amount;                 // Voxel count
	cv::Size m_plane_size;                  // Camera FoV plane WxH

	std::vector<Voxel*> m_voxels;           // Pointer vector to all voxels in the half-space
	std::vector<Voxel*> m_visible_voxels;   // Pointer vector to all visible voxels
	std::vector<Point2f> m_groundCoordinates;
	std::vector<int> m_clusterLabels;
	std::vector<Point2f> m_centers;
	std::vector<vector<int>> m_clusters;
	std::vector<Ptr<cv::ml::EM>> m_GMMS;

	vector<Mat> m_cluster_paths = { Mat(), Mat(), Mat(), Mat() };
	vector<Mat> m_cluster_paths_y = { Mat(), Mat(), Mat(), Mat() };

	void initialize();

public:
	Reconstructor(
			const std::vector<Camera*> &);
	virtual ~Reconstructor();

	void update();

	void setPath(int cluster_id, Point2f position, int matched_label) {
		m_cluster_paths[matched_label].push_back(position);
	}

	void writePaths() {
		for (int cluster_id = 0; cluster_id < m_clusters.size(); cluster_id++) {
			cv::FileStorage fs_paths(std::format("paths/cluster_{}_path.xml", cluster_id + 1), cv::FileStorage::WRITE);
			fs_paths << "positions" << m_cluster_paths[cluster_id];
			fs_paths.release();
			cout << "Wrote path " << cluster_id << endl;
		}
	}

	const std::vector<Voxel*>& getVisibleVoxels() const
	{
		return m_visible_voxels;
	}

	const std::vector<int>& getClusterLabels() const
	{
		return m_clusterLabels;
	}

	const std::vector<vector<int>>& getClusters() const
	{
		return m_clusters;
	}

	const std::vector<Point2f>& getClusterCenters() const
	{
		return m_centers;
	}

	const std::vector<Voxel*>& getVoxels() const
	{
		return m_voxels;
	}

	void setVisibleVoxels(
			const std::vector<Voxel*>& visibleVoxels)
	{
		m_visible_voxels = visibleVoxels;
	}

	void setVoxels(
			const std::vector<Voxel*>& voxels)
	{
		m_voxels = voxels;
	}

	const std::vector<cv::Point3f*>& getCorners() const
	{
		return m_corners;
	}

	int getSize() const
	{
		return m_height;
	}

	const cv::Size& getPlaneSize() const
	{
		return m_plane_size;
	}

	std::vector<Ptr<cv::ml::EM>>& getGMMS()
	{
		vector<Ptr<cv::ml::EM>> GMMS;
		for (int i = 0; i < 4; i++)
		{
			Ptr<cv::ml::EM> GMM = cv::Algorithm::load<cv::ml::EM>(std::format("GMMS/GMM_{}.yaml", i + 1));
			GMMS.push_back(GMM);
		}

		m_GMMS = GMMS;

		return m_GMMS;
	}

	void cluster();
	void buildOfflineColorModels();
	void interpretGMM(int GMM_number, Ptr<cv::ml::EM> GMM);
};

} /* namespace nl_uu_science_gmt */
