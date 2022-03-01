#include <cstdlib>
#include <string>
#include <iostream>
#include <filesystem>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "Video.h"

using namespace nl_uu_science_gmt;
using namespace cv;
using namespace std;

/// <summary>
/// Calculates the quality measure of two (grayscale) images.
/// </summary>
tuple<int, float> quality_measure(Mat img1, Mat img2) {
	// Get the total pixel count.
	int pixel_count = img1.total();

	// Calculate the amount of pixels that are white when they should be black, and vice versa.
	Mat XOR_diff = Mat();
	cv::bitwise_xor(img1, img2, XOR_diff);
	unsigned int diff_count = countNonZero(XOR_diff);

	// Report quality measure variables.
	double success = (pixel_count - diff_count) / (float)pixel_count;
	cout << "Number of differing pixels " << diff_count << endl;
	cout << "Quality measure: " << success << endl;
	cout << "Loss: " << 1 - success << endl;

	// Return the amount of differing pixels, and the success rate.
	return tuple<int, float>(diff_count, success);
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

int main(
		int argc, char** argv)
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
	
	//VoxelReconstruction::showKeys();
	//VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	//vr.run(argc, argv);

	Mat golden_standard = cv::imread("./data/cam1/golden_standard.png", cv::IMREAD_GRAYSCALE);
	Mat two_pixels_off = cv::imread("./data/cam1/two_pixels_off.png", cv::IMREAD_GRAYSCALE);
	auto measure_values = quality_measure(golden_standard, two_pixels_off);

	while (true) {}
	return EXIT_SUCCESS;	
}