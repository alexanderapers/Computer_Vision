#include "precomp.h"
#include "MOG2BackgroundSubtraction.h"
#include "../../controllers/Video.h"
#include "../../controllers/Camera.h"
#include "../../utilities/General.h"

namespace nl_uu_science_gmt {
	/// <summary>
	/// Uses the background video to "train" the MOG2 background model. Uses the learning rate member variable.
	/// </summary>
	void MOG2BackgroundSubtraction::prepareBgRef() const {
		
		// Get background video.
		Video background_video = Video(std::format("./data/cam{}", camera_id), General::BackgroundVideo);
		VideoCapture vid = background_video.m_video_capture;
		int frame_count = background_video.m_frame_count;

		for (int i = 0; i < frame_count; i++) {
			Mat frame;
			vid.set(cv::VideoCaptureProperties::CAP_PROP_POS_FRAMES, i);
			vid >> frame;
			Mat mask;
			pMOG2->apply(frame, mask, learning_rate);
		}
		vid.release();
	}

	/// <summary>
	/// Using a learning rate of zero, applies the current frame to the MOG2 background subtractor to process the foreground image.
	/// </summary>
	void MOG2BackgroundSubtraction::processForeground(Scene3DRenderer* renderer, Camera* camera) const {
		assert(!camera->getFrame().empty());

		Mat frame = camera->getFrame();

		// Apply the MOG2 background subtractor with a learning rate of 0 because the background model doesn't have to get updated.
		Mat foreground;
		pMOG2->apply(frame, foreground, 0);

		// Blur the foreground mask to reduce the effect of noise and false positives
		cv::blur(foreground, foreground, cv::Size(1, 1));

		// Remove the shadow parts and the noise
		cv::threshold(foreground, foreground, 128, 255, cv::THRESH_BINARY);

		// Improve the foreground image
		Mat erode_kernel = Mat(4, 4, CV_8U, Scalar(1, 1, 1));
		Mat dilate_kernel = Mat(2, 2, CV_8U, Scalar(1, 1, 1));
		cv::erode(foreground, foreground, erode_kernel);
		cv::dilate(foreground, foreground, dilate_kernel);


		camera->setForegroundImage(foreground);
	}
}
