#include "Gaussian.h"

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
#include <filesystem>

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{
	Gaussian::Gaussian(Video* background_video) :
		m_background_vid(background_video)
	{
		m_frame_count = m_background_vid->m_frame_count;
		m_height = m_background_vid->m_video_capture.get(CAP_PROP_FRAME_HEIGHT);
		m_width = m_background_vid->m_video_capture.get(CAP_PROP_FRAME_WIDTH);
	}

	tuple<Mat, Mat> Gaussian::calculateGaussian()
	{
		m_background_vid->m_video_capture.set(1, 0);

		Mat running_mean = Mat(m_height, m_width, CV_32FC3, Scalar(0, 0, 0));
		Mat running_std = Mat(m_height, m_width, CV_32FC3, Scalar(0, 0, 0));
		Mat new_mean = Mat(m_height, m_width, CV_32FC3, Scalar(0, 0, 0));
		Mat new_std = Mat(m_height, m_width, CV_32FC3, Scalar(0, 0, 0));

		Mat frame = Mat(m_height, m_width, CV_32SC3);
		Mat hsv_frame = Mat(m_height, m_width, CV_32SC3);

		VideoCapture vid = m_background_vid->m_video_capture;
		for (int i = 0; i < m_frame_count; i++)
		{
			vid.set(1, i);
			vid >> frame;
			//cvtColor(frame, hsv_frame, CV_BGR2HSV);
			//hsv_frame.convertTo(hsv_frame, CV_32FC3);

			frame.convertTo(frame, CV_32FC3);

			new_mean = running_mean + (frame - running_mean) / (i + 1);
			new_std = running_std + (frame - running_mean).mul(frame - new_mean);

			running_mean = new_mean;
			running_std = new_std;
		}

		Mat std = Mat(m_height, m_width, CV_32FC3);
		cv::sqrt(running_std / (m_frame_count - 1), std);


		Mat mean;
		cvtColor(running_mean, mean, CV_HSV2BGR);
		mean.convertTo(mean, CV_32SC3);
		
		cv::imshow("frame", mean);
		waitKey(10000);
		cv::imshow("frame", std);
		waitKey(10000);
	
		return tuple<Mat, Mat>(running_mean, std);
	}
	
}