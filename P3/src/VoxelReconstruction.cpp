/*
 * VoxelReconstruction.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#include "precomp.h"
#include "VoxelReconstruction.h"
#include "background/mog2/MOG2BackgroundSubtraction.h"
#include "background/gaussian/GaussianBackgroundSubtraction.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <stddef.h>
#include <cassert>
#include <iostream>
#include <sstream>

#include "controllers/Glut.h"
#include "controllers/Reconstructor.h"
#include "controllers/Scene3DRenderer.h"
#include "utilities/General.h"
#include <opencv2/imgproc.hpp>

using namespace nl_uu_science_gmt;
using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Main constructor, initialized all cameras
 */
VoxelReconstruction::VoxelReconstruction(const string &dp, const int cva) :
		m_data_path(dp), m_cam_views_amount(cva)
{
	const string cam_path = m_data_path + "cam";

	for (int v = 0; v < m_cam_views_amount; ++v)
	{
		stringstream full_path;
		full_path << cam_path << (v + 1) << PATH_SEP;

		/*
		 * Assert that there's a background image or video file and \
		 * that there's a video file
		 */
		std::cout << full_path.str() << General::BackgroundImageFile << std::endl;
		std::cout << full_path.str() << General::VideoFile << std::endl;
		assert(
			General::fexists(full_path.str() + General::BackgroundImageFile)
			&&
			General::fexists(full_path.str() + General::VideoFile)
		);

		/*
		 * Assert that if there's no config.xml file, there's an intrinsics file and
		 * a checkerboard video to create the extrinsics from
		 */
		assert(
			(!General::fexists(full_path.str() + General::ConfigFile) ?
				General::fexists(full_path.str() + General::IntrinsicsFile) &&
					General::fexists(full_path.str() + General::CheckerboadVideo)
			 : true)
		);

		m_cam_views.push_back(new Camera(full_path.str(), General::ConfigFile, v));
		
		BackgroundSubtraction* bg_sub = new MOG2BackgroundSubtraction(v + 1, 0.1f);
		//BackgroundSubtraction* bg_sub = new GaussianBackgroundSubtraction(v);
		bg_sub->prepareBgRef();
		m_bg_subtractors.push_back(bg_sub);
	}
}

/**
 * Main destructor, cleans up pointer vector memory of the cameras
 */
VoxelReconstruction::~VoxelReconstruction()
{
	for (size_t v = 0; v < m_cam_views.size(); ++v)
	{
		delete m_cam_views[v];
		delete m_bg_subtractors[v];
	}
		
}

/**
 * What you can hit
 */
void VoxelReconstruction::showKeys()
{
	cout << "VoxelReconstruction v" << VERSION << endl << endl;
	cout << "Use these keys:" << endl;
	cout << "q       : Quit" << endl;
	cout << "p       : Pause" << endl;
	cout << "b       : Frame back" << endl;
	cout << "n       : Next frame" << endl;
	cout << "r       : Rotate voxel space" << endl;
	cout << "s       : Show/hide arcball wire sphere (Linux only)" << endl;
	cout << "v       : Show/hide voxel space box" << endl;
	cout << "g       : Show/hide ground plane" << endl;
	cout << "c       : Show/hide cameras" << endl;
	cout << "i       : Show/hide camera numbers (Linux only)" << endl;
	cout << "o       : Show/hide origin" << endl;
	cout << "t       : Top view" << endl;
	cout << "1,2,3,4 : Switch camera #" << endl << endl;
	cout << "Zoom with the scrollwheel while on the 3D scene" << endl;
	cout << "Rotate the 3D scene with left click+drag" << endl << endl;
}

/// <summary>
/// Calculates the quality measure of two (grayscale) images.
/// </summary>
inline float generate_quality_measure(Mat img1, Mat img2) {
	// Get the total pixel count.
	int pixel_count = img1.total();

	// Calculate the amount of pixels that are white when they should be black, and vice versa.
	Mat XOR_diff = Mat();
	cv::bitwise_xor(img1, img2, XOR_diff);
	unsigned int diff_count = countNonZero(XOR_diff);

	// Report quality measure variables.
	double success = (pixel_count - diff_count) / (float)pixel_count;

	// Return the amount of differing pixels, and the success rate.
	return success;
}

/// <summary>
/// Using an arbitrary camera view, this function tunes the HSV thresholds of a given renderer.
/// </summary>
void VoxelReconstruction::tuneRenderer(Scene3DRenderer& renderer, Camera * camera) {
	cout << "Tuning renderer HSV thresholds... (This may take a while)" << endl;

	// Take an arbitrary camera, find the optimal parameters.
	int cam_id = 1;
	string cam_path = std::format("./data/cam{}/", cam_id);

	// Take one frame from video.avi
	VideoCapture cap = VideoCapture(std::format("{}video.avi", cam_path));
	Mat frame;
	cap >> frame;

	// TODO: Get SD & mean of background.
	Mat background = cv::imread(std::format("{}background.png", cam_path));

	// Get golden standard to compare against
	Mat golden_standard = cv::imread(std::format("{}golden_standard.png", cam_path), cv::IMREAD_GRAYSCALE);

	// For all channels, generate every possible combination (limit the amount of values to combine).

	// SIMPLE GRID SEARCH
	std::vector<int> possible_h_values(10);
	std::generate(possible_h_values.begin(), possible_h_values.end(), [n = 0]() mutable { return n++; });
	std::vector<int> possible_s_values(25);
	std::generate(possible_s_values.begin(), possible_s_values.end(), [n = 5]() mutable { return n++; });
	std::vector<int> possible_v_values(40);
	std::generate(possible_v_values.begin(), possible_v_values.end(), [n = 10]() mutable { return n += 2; });

	float best_quality_measure = 0;
	vector<int> best_HSV;
	// Loop over all combinations of HSV values.
	for (int H : possible_h_values) for (int S : possible_s_values) for (int V : possible_v_values) {
		// Set renderer values.
		renderer.setHThreshold(H);
		renderer.setSThreshold(S);
		renderer.setVThreshold(V);

		m_bg_subtractors[cam_id]->processForeground(&renderer, camera);
		Mat foreground = camera->getForegroundImage();

		float quality_measure = generate_quality_measure(foreground, golden_standard);

		if (quality_measure > best_quality_measure) {
			best_quality_measure = quality_measure;
			best_HSV = { H, S, V };
		}
	}

	int H = best_HSV[0];
	int S = best_HSV[1];
	int V = best_HSV[2];
	renderer.setHThreshold(H);
	renderer.setSThreshold(S);
	renderer.setVThreshold(V);

	cout << "Best quality measure " << best_quality_measure << endl;
	cout << std::format("Camera {0} best HSV values: {1} {2} {3}", cam_id, H, S, V) << endl;
}

/**
 * - If the xml-file with camera intrinsics, extrinsics and distortion is missing,
 *   create it from the checkerboard video and the measured camera intrinsics
 * - After that initialize the scene rendering classes
 * - Run it!
 */
void VoxelReconstruction::run(int argc, char** argv)
{
	for (int v = 0; v < m_cam_views_amount; ++v)
	{
		bool has_cam = Camera::detExtrinsics(m_cam_views[v]->getDataPath(), General::CheckerboadVideo,
				General::IntrinsicsFile, m_cam_views[v]->getCamPropertiesFile());
		assert(has_cam);
		if (has_cam) has_cam = m_cam_views[v]->initialize();
		assert(has_cam);
	}

	destroyAllWindows();
	namedWindow(VIDEO_WINDOW, CV_WINDOW_KEEPRATIO);

	//// offline
	//Reconstructor reconstructor_offline(m_cam_views);
	//Camera* cam_view_offline = m_cam_views[0];
	//cam_view_offline->advanceVideoFrame();
	//Scene3DRenderer scene3d_offline(reconstructor_offline, m_cam_views, m_bg_subtractors);

	//cvDestroyAllWindows();

 //	scene3d_offline.setHThreshold(9);
	//scene3d_offline.setSThreshold(29);
	//scene3d_offline.setVThreshold(54);
	//
	//scene3d_offline.setCurrentFrame(1000);
	//scene3d_offline.processFrame();
	//reconstructor_offline.update();
	//reconstructor_offline.buildOfflineColorModels();

	Reconstructor reconstructor(m_cam_views);
	Camera* cam_view = m_cam_views[0];
	cam_view->advanceVideoFrame();
	Scene3DRenderer scene3d(reconstructor, m_cam_views, m_bg_subtractors);

	//// Instead of tuning the renderer, we now manually set the previously found HSV values (H=9, S=29, V=54)
	//// tune_renderer(scene3d, cam_view); // Old tuning code.
	scene3d.setHThreshold(9);
	scene3d.setSThreshold(29);
	scene3d.setVThreshold(54);

	// offline
	//// use a suitable frame to build model on
	scene3d.setCurrentFrame(518);
	scene3d.processFrame();
	reconstructor.update();
	reconstructor.buildOfflineColorModels();
	scene3d.setCurrentFrame(0);
	// offline

	Glut glut(scene3d);

#ifdef __linux__
	glut.initializeLinux(SCENE_WINDOW.c_str(), argc, argv);
#elif defined _WIN32
	glut.initializeWindows(SCENE_WINDOW.c_str());
	glut.mainLoopWindows();
#endif
}

} /* namespace nl_uu_science_gmt */
