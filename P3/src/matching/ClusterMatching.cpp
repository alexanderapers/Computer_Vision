#include "precomp.h"
#include "ClusterMatching.h"

#include "matrix.h"
#include "munkres.h"
#include <src/controllers/Camera.h>
#include <src/controllers/Scene3DRenderer.h>
#include <src/controllers/Reconstructor.h>

#include <unordered_set>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <common.h>

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt {

vector<unordered_set<Point>> ClusterMatching::get_cluster_projections(int camera_id, Camera*camera, Reconstructor* reconstructor) {
	// Get the cluster/voxel data.
	vector<Reconstructor::Voxel*> voxels = reconstructor->getVisibleVoxels();
	vector<vector<int>> clusters = reconstructor->getClusters();

	vector<std::unordered_set<Point>> cluster_points;

	// For each cluster, get the unique points and colors.
	for (vector<int> cluster : clusters) {

		std::unordered_set<Point> points;

		for (int voxel_id : cluster) {
			Reconstructor::Voxel* voxel = voxels[voxel_id];

			// Check whether we want this voxel, and whether it is on screen.
			if (voxel->z <= LOWER_GMM_LIMIT || voxel->z >= UPPER_GMM_LIMIT || !voxel->valid_camera_projection[camera_id]) {
				continue;
			}

			Point point = voxel->camera_projection[camera_id];

			// If we already got this color, skip it.
			if (points.contains(point))
			{
				continue;
			}

			// Store the point in the unique set.
			points.insert(point);
		}

		cluster_points.push_back(points);
	}
	return cluster_points;
}

vector<pair<float, int>> ClusterMatching::sort_clusters_by_distance(Camera* camera, Reconstructor* reconstructor) {

	// Get camera and cluster center positions.
	Point3f cam_position = camera->getCameraLocation();
	Point2f cam_position_2d = Point2f(cam_position.x, cam_position.y);
	vector<Point2f> cluster_centers = reconstructor->getClusterCenters();

	// Calculate the distance from the camera to each cluster center for occlusion purposes.
	vector <pair<float, int>> cluster_cam_distances;
	for (int cluster_id = 0; cluster_id < cluster_centers.size(); cluster_id++) {

		// Euclidean distance
		float distance = cv::norm(cam_position_2d - cluster_centers[cluster_id]);
		cluster_cam_distances.push_back(pair<float, int>(distance, cluster_id));
	}

	// Sort the camera distances in ascending order. We now know which clusters are the closest to the camera.
	sort(cluster_cam_distances.begin(), cluster_cam_distances.end());

	return cluster_cam_distances;
}

void ClusterMatching::resolve_occlusion(vector<unordered_set<Point>>& cluster_points, vector<pair<float, int>>& cluster_cam_distances) {
	// From closest to farthest, get the intersections of the closest element and the farthest ones,
	// then remove those shared points from the farther ones. Continue until you reach the last element.
	for (int i = 0, j = 1; j < cluster_cam_distances.size(); i++, j++) {
		int closest_cluster_id = cluster_cam_distances[i].second;
		std::unordered_set<Point> closest_cluster_points = cluster_points[closest_cluster_id];

		for (int k = j; k < cluster_cam_distances.size(); k++) {
			int farther_cluster_id = cluster_cam_distances[k].second;
			std::unordered_set<Point> farther_cluster_points = cluster_points[farther_cluster_id];

			// Loop over both sets, remove shared points from the right cluster.
			for (Point x : farther_cluster_points)
				if (closest_cluster_points.contains(x))
					cluster_points[farther_cluster_id].erase(x);
		}
	}
}

vector<Mat> ClusterMatching::get_cluster_projection_colors(Camera* camera, vector<unordered_set<Point>>& cluster_points) {
	Mat current_frame = camera->getFrame();
	cvtColor(current_frame, current_frame, CV_BGR2HSV); // convert to HSV
	//Mat masks = Mat(current_frame.size(), current_frame.type(), Scalar(0, 0, 0));

	vector<Mat> colors(cluster_points.size());

	for (int z = 0; z < cluster_points.size(); z++)
	{
		colors[z] = Mat();

		for (Point p : cluster_points[z])
		{
			Vec3b color = current_frame.at<Vec3b>(p);
			Mat col(1, 3, CV_8UC1);

			//Vec3b mask_col;
			//if (z == 0)
			//	mask_col = { 255, 0, 0 };
			//if (z == 1)
			//	mask_col = { 0, 255, 0 };
			//if (z == 2)
			//	mask_col = { 0, 0, 255 };
			//if (z == 3)
			//	mask_col = { 120, 120, 0 };

			for (int m = 0; m < 3; m++)
				col.at<char>(0, m) = color[m];
			colors[z].push_back(col);

			//masks.at<Vec3b>(p) = mask_col;
		}
	}

	//imshow("frame", masks);
	//waitKey(1);

	return colors;
}

map<int, int> ClusterMatching::match_clusters(Reconstructor* reconstructor, Scene3DRenderer* scene3D) {

	// Get the camera and the current frame.
	int camera_id = 3;
	Camera* camera = scene3D->getCameras()[camera_id];

	// Get the cluster projections, sort the clusters by distance, and then resolve occclusion.
	vector<std::unordered_set<Point>> cluster_points = get_cluster_projections(camera_id, camera, reconstructor);
	vector<pair<float, int>> cluster_cam_distances = sort_clusters_by_distance(camera, reconstructor);
	resolve_occlusion(cluster_points, cluster_cam_distances);

	vector<Mat> colors = get_cluster_projection_colors(camera, cluster_points);

	//Mat match_matrix(4, 4, CV_32SC1, Scalar(0));
	Matrix<int> match_matrix(4, 4);

	vector<Ptr<cv::ml::EM>> GMMS = reconstructor->getGMMS();

	for (int i = 0; i < colors.size(); i++)
	{
		for (int j = 0; j < colors[i].rows; j++)
		{
			int most_likely = -1;
			double log_likelihood = -DBL_MAX;

			for (int k = 0; k < 4; k++)
			{
				Vec2d likelihood = GMMS[k]->predict2(colors[i].row(j), noArray());

				if (likelihood[0] > log_likelihood)
				{
					most_likely = k;
					log_likelihood = likelihood[0];
				}
			}

			//match_matrix.at<int>(Point(i, most_likely))++;
			match_matrix(i, most_likely) = match_matrix(i, most_likely) + 1;
		}
	}

	//cout << match_matrix << endl;

	Munkres<int> munkres;
	munkres.solve(match_matrix);

	map<int, int> matching;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (match_matrix(i, j) == 0)
			{
				matching[i] = j;
			}
		}
	}

	return matching;
}


}