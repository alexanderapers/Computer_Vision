#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

using namespace cv;

namespace nl_uu_science_gmt
{

class Video
{
private:
	int m_frame_count;
	int countFramesManual();

public:
	std::string m_full_data_path;
	const std::string m_file_path;
	const std::string m_file_name;
	cv::VideoCapture m_video_capture;
	Video(const std::string file_path, const std::string file_name);
	void getFrames(int number, const std::string out_location, const std::string file_name);


};

}