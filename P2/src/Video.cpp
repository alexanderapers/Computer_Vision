#include "Video.h"
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <assert.h>
#include <filesystem>

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{
	Video::Video(const std::string file_path, const std::string file_name) :
		m_file_path(file_path), m_file_name(file_name)
	{
		m_full_data_path = std::format("{}/{}", m_file_path, m_file_name);
		m_video_capture = VideoCapture(m_full_data_path);
		m_video_capture.open(m_full_data_path);
		m_frame_count = countFramesManual();

		if (!m_video_capture.isOpened())
		{
			cout << "Error opening video stream or file" << endl;
		}
		
	}

	void Video::getFrames(int number, const std::string out_location, const std::string file_name)
	{
		assert(number < m_frame_count);

		if (!std::filesystem::exists(out_location))
		{
			std::filesystem::create_directory(out_location);
		}

		int frame_jump = m_frame_count / number;

		for (int i = 0; i < number; i++)
		{
			assert(i * frame_jump < m_frame_count);
			m_video_capture.set(1, i * frame_jump);
			Mat frame;
			m_video_capture >> frame;
			const std::string file_path_name = number != 1 ?
				std::format("{0}/{1}_{2}.png", out_location, file_name, i * frame_jump) :
				std::format("{0}/{1}.png", out_location, file_name);

			imwrite(file_path_name, frame);
		}
	}

	int Video::countFramesManual()
	{
		int total = 0;

		while (true)
		{
			Mat frame;
			bool found = m_video_capture.read(frame);
			if (!found)
				break;

			total++;
		}

		return total;
	}






}