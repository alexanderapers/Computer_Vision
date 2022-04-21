#pragma once
#include <src/controllers/Scene3DRenderer.h>

namespace nl_uu_science_gmt {

//class Scene3DRenderer;
class ClusterMatching
{
public:
	static map<int, int> match_clusters(Reconstructor* reconstructor, Scene3DRenderer* scene3D);
private:
	static vector<unordered_set<Point>> get_cluster_projections(int camera_id, Camera* camera, Reconstructor* reconstructor);
	static vector<pair<float, int>> sort_clusters_by_distance(Camera* camera, Reconstructor* reconstructor);
	static void resolve_occlusion(vector<unordered_set<Point>>& cluster_points, vector<pair<float, int>>& cluster_cam_distances);
	static vector<Mat> get_cluster_projection_colors(Camera* camera, vector<unordered_set<Point>>& cluster_points);
};

}

