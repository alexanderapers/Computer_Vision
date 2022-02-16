//// main.cpp
//
//
//
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

int streamWebcamFeed()
{
    cv::VideoCapture videoCapture(0);

    if (!videoCapture.isOpened())
    {
        std::cout << "Unable to connect to webcam" << std::endl;
        return -1;
    }

    while (true)
    {
        cv::Mat frame;
        videoCapture >> frame;
        if (frame.empty()) break;
        cv::imshow("Camera feed", frame);
        if (cv::waitKey(10) == 27) break;
    }

    videoCapture.release();
    return 0;

}

int main()
{
    //std::string image_path = samples::findFile("starry_night.jpg");
    //Mat img = imread(image_path, IMREAD_COLOR);
    //if (img.empty())
    //{
    //    std::cout << "Could not read the image: " << image_path << std::endl;
    //    return 1;
    //}
    //imshow("Display window", img);
    //int k = waitKey(0); // Wait for a keystroke in the window
    //if (k == 's')
    //{
    //    imwrite("starry_night.png", img);
    //}

    streamWebcamFeed();

    return 0;
}