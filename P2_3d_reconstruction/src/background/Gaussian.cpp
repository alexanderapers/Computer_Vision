#include "precomp.h"
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


		VideoCapture vid = m_background_vid->m_video_capture;
		for (int i = 0; i < m_frame_count; i++)
		{
			Mat frame;
			vid.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, i);
			vid >> frame;

			Mat hsv_frame;
			cvtColor(frame, hsv_frame, CV_BGR2HSV);
			hsv_frame.convertTo(hsv_frame, CV_32FC3);

			running_mean = running_mean + ((hsv_frame - running_mean) / (float)(i + 1));
		}

		Mat accumulator = Mat(m_height, m_width, CV_32FC3, Scalar(0, 0, 0));;
		for (int i = 0; i < m_frame_count; i++) {
			Mat frame;
			vid.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, i);
			vid >> frame;

			Mat hsv_frame;
			cvtColor(frame, hsv_frame, CV_BGR2HSV);
			hsv_frame.convertTo(hsv_frame, CV_32FC3);

			accumulator = accumulator + ((running_mean - hsv_frame).mul(running_mean - hsv_frame));
		}
		
		Mat std = Mat(m_height, m_width, CV_32FC3);
		cv::sqrt(accumulator / (float)(m_frame_count - 1), std);


		//Mat mean;
		//running_mean.convertTo(mean, CV_8UC3);
		//std.convertTo(std, CV_8UC3);

		return tuple<Mat, Mat>(running_mean, std);
	}
	
}