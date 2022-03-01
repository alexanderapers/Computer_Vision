#include <cstdlib>
#include <string>
#include <iostream>
#include <filesystem>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include "VoxelReconstruction.h"
#include "utilities/General.h"
#include "Video.h"

using namespace nl_uu_science_gmt;
using namespace cv;
using namespace std;

/// <summary>
/// Calculates the quality measure of two (grayscale) images.
/// </summary>
float generate_quality_measure(Mat img1, Mat img2) {
	// Get the total pixel count.
	int pixel_count = img1.total();

	// Calculate the amount of pixels that are white when they should be black, and vice versa.
	Mat XOR_diff = Mat();
	cv::bitwise_xor(img1, img2, XOR_diff);
	unsigned int diff_count = countNonZero(XOR_diff);

	// Report quality measure variables.
	double success = (pixel_count - diff_count) / (float)pixel_count;
	//cout << "Number of differing pixels " << diff_count << endl;
	//cout << "Quality measure: " << success << endl;
	//cout << "Loss: " << 1 - success << endl;

	// Return the amount of differing pixels, and the success rate.
	return success;
}

/// <summary>
/// Given a video, this function saves an amount of frames in the corresponding camera folder.
/// </summary>
/// <param name="video_path">The path to the video file.</param>
/// <param name="output_filename">The name of the output file.</param>
/// <param name="n_frames">jthe amount of frames to save as images.</param>
/// <param name="output_folder">The name of the output folder.</param>
/// <param name="n_cams">The number of cameras.</param>
void save_frames(string video_path, string output_filename, int n_frames = 1, string output_folder = "", int n_cams = 4) {
	if (output_folder != "") {
		output_folder = std::format("/{}", output_folder);
	}

	for (int i = 1; i <= n_cams; i++)
	{
		Video vid = Video(std::format("./data/cam{}", i), video_path);
		vid.getFrames(n_frames, std::format("./data/cam{0}{1}", i, output_folder), output_filename);
	}
}

vector<Mat> split_hsv_channels(Mat img) {
	Mat hsv_img;
	cvtColor(img, hsv_img, CV_BGR2HSV);  // from BGR to HSV color space

	vector<Mat> img_channels;

	// Split the HSV-channels for further analysis
	split(hsv_img, img_channels);
	return img_channels;
}

/** Copied from Scene3DRenderer
 * Separate the background from the foreground
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
Mat create_foreground(vector<Mat> frame_channels, vector<Mat> bg_channels, int h, int s, int v)
{
	// Background subtraction H
	Mat tmp, foreground, background;
	absdiff(frame_channels[0], bg_channels[0], tmp);
	threshold(tmp, foreground, h, 255, CV_THRESH_BINARY);

	// Background subtraction S
	absdiff(frame_channels[1], bg_channels[1], tmp);
	threshold(tmp, background, s, 255, CV_THRESH_BINARY);
	bitwise_and(foreground, background, foreground);

	// Background subtraction V
	absdiff(frame_channels[2], bg_channels[2], tmp);
	threshold(tmp, background, v, 255, CV_THRESH_BINARY);
	bitwise_or(foreground, background, foreground);

	// Improve the foreground image
	return foreground;
}

vector<vector<int>> search_algorithm() {
	vector<vector<int>> cam_HSV_thresholds;

	// For each camera, find the optimal parameters.
	for (int cam_id = 1; cam_id < 5; cam_id++) {
		string cam_path = std::format("./data/cam{}/", cam_id);
		
		// Take one frame from video.avi
		VideoCapture cap = VideoCapture(std::format("{}video.avi", cam_path));
		Mat frame;
		cap >> frame;
		
		// TODO: Get SD & mean of background.
		Mat background = cv::imread(std::format("{}background.png", cam_path));

		// Get golden standard to compare against
		Mat golden_standard = cv::imread(std::format("{}golden_standard.png", cam_path), cv::IMREAD_GRAYSCALE);

		// Split the frame and bg into hsv channels for further analysis.
		vector<Mat> frame_channels = split_hsv_channels(frame);
		vector<Mat> bg_channels = split_hsv_channels(background);

		// TODO: Insert hyperparameter tuning algorithm here
		//		 For all channels, generate every possible combination (limit the amount of values to combine).

		// ATTEMPT AT SIMPLE GRID SEARCH ---------------------------------------------------------------- 
		vector<int> possible_values = { 0, 20, 40, 60, 80, 100, 120 };

		float best_quality_measure = 0;
		vector<int> best_HSV;
		// Loop over all combinations of HSV values.
		for (int H : possible_values) for (int S : possible_values) for (int V : possible_values) {

			Mat foreground = create_foreground(frame_channels, bg_channels, H, S, V);

			float quality_measure = generate_quality_measure(foreground, golden_standard);
			
			if (quality_measure > best_quality_measure) {
				best_quality_measure = quality_measure;
				best_HSV = {H, S, V};
			}
		}
		// -----------------------------------------------------------------------------------------------

		cam_HSV_thresholds.push_back(best_HSV);

		int H = best_HSV[0];
		int S = best_HSV[1];
		int V = best_HSV[2];
		cout << "Best quality measure " << best_quality_measure << endl;
		cout << std::format("Camera {0} best HSV values: {1} {2} {3}", cam_id, H, S, V) << endl;
	}
	return cam_HSV_thresholds;
	// For every combination of channel thresholds:
	// Create a thresholded image using the background image (or mean/sd background image). LOOK AT ProcessForeground function!

	// Compare the thresholded image created by this combination with the gold standard mask using the XOR measure we created.
	// If the quality measure is higher than the previous combination of parameters, overwrite them with the current ones.

	// After finding a set of parameters that generates the closest quality measure to 1 (minimizing loss):
	// Update the final HSV (threshold) values in the 3D renderer.
	// Output parameters.
}

int main(int argc, char** argv)
{
	//// GETS FRAMES FROM INTRINSICS VIDEO AND SAVES THEM INTO FILE
	//save_frames(General::IntrinsicsVideo, "frame", 50, "intrinsics");

	//// USE MIDDLE FRAME OF BACKGROUND.AVI FOR EACH CAMERA
	//save_frames(General::BackgroundVideo, "background", 1);

	/*
	 READS INTRINSICS FROM THE FILES AND WRITES THEM TO INTRINSICS.XML
	for (int i = 1; i < 5; i++)
	{
		const std::string input_file_path = std::format("..\\P1\\cam{}_out_camera_data.xml", i);
		const std::string output_file_path = std::format("./data/cam{}/", i) + General::IntrinsicsFile;
		General::writeIntrinsics(input_file_path, output_file_path);
	}*/

	auto results = search_algorithm();
	
	//VoxelReconstruction::showKeys();
	//VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	//vr.run(argc, argv);

	//Mat golden_standard = cv::imread("./data/cam1/golden_standard.png", cv::IMREAD_GRAYSCALE);
	//Mat two_pixels_off = cv::imread("./data/cam1/two_pixels_off.png", cv::IMREAD_GRAYSCALE);
	//auto measure_values = quality_measure(golden_standard, two_pixels_off);

	while (true) {}
	return EXIT_SUCCESS;	
}