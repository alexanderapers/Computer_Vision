#include "precomp.h"
#include "GaussianBackgroundSubtraction.h"
#include "../../controllers/Scene3DRenderer.h"
#include "../../controllers/Camera.h"
#include "../../controllers/Video.h"

namespace nl_uu_science_gmt {

    /// <summary>
    /// Uses the background video to create mean and standard deviation matrices. Writes them to background.xml and background_sd.xml respectively.
    /// </summary>
    void GaussianBackgroundSubtraction::prepareBgRef() const {
        Video background_video = Video(std::format("./data/cam{}", camera_id), General::BackgroundVideo);
        Gaussian gaussian = Gaussian(&background_video);
        tuple<Mat, Mat> mats = gaussian.calculateGaussian();
        Mat average = get<0>(mats);
        Mat sd = get<1>(mats);

        FileStorage fs(std::format("./data/cam{}/background.xml", camera_id), FileStorage::WRITE);
        FileStorage fs2(std::format("./data/cam{}/background_sd.xml", camera_id), FileStorage::WRITE);
    }

	/// <summary>
	/// Uses the given camera's background mean and standard deviation matrices to separate the foreground and background of the current frame.
	/// The HSV thresholds in the renderer determine how many standard deviations a pixel can be off from the mean before being considered foreground.
	/// Applies laplacian smoothing to the standard deviation to prevent divisions by zero.
	/// </summary>
	void GaussianBackgroundSubtraction::processForeground(Scene3DRenderer* renderer, Camera* camera) const {
		assert(!camera->getFrame().empty());
		Mat hsv_image, hsv_3_channels;
		cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space
		hsv_image.convertTo(hsv_3_channels, CV_32FC3);

		vector<Mat> channels;
		split(hsv_3_channels, channels);  // Split the HSV-channels for further analysis

		// Background subtraction H
		Mat sd, diff, tmp, foreground, background;

		absdiff(channels[0], camera->getBgHsvChannels().at(0), diff);
		sd = camera->getBgSdHsvChannels().at(0);
		tmp = diff / (sd + 1);
		threshold(tmp, foreground, renderer->getHThreshold(), 255, CV_THRESH_BINARY);

		// Background subtraction S
		absdiff(channels[1], camera->getBgHsvChannels().at(1), diff);
		sd = camera->getBgSdHsvChannels().at(1);
		tmp = diff / (sd + 1);
		threshold(tmp, background, renderer->getSThreshold(), 255, CV_THRESH_BINARY);
		bitwise_and(foreground, background, foreground);

		// Background subtraction d
		absdiff(channels[2], camera->getBgHsvChannels().at(2), diff);
		sd = camera->getBgSdHsvChannels().at(2);
		tmp = diff / (sd + 1);
		threshold(tmp, background, renderer->getVThreshold(), 255, CV_THRESH_BINARY);
		bitwise_or(foreground, background, foreground);

		foreground.convertTo(foreground, CV_8U);

		// Improve the foreground image
		Mat erode_kernel = Mat(1, 1, CV_8U, Scalar(1, 1, 1));
		Mat dilate_kernel = Mat(3, 3, CV_8U, Scalar(1, 1, 1));
		cv::erode(foreground, foreground, erode_kernel);
		cv::dilate(foreground, foreground, dilate_kernel);

		camera->setForegroundImage(foreground);
    }
}
