//// main.cpp
//
//
//
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <format>
#include <Windows.h>
//#include <cstdint>

//namespace fs = std::filesystem;
using namespace cv;

String CAMERA_NAME = "Alex_Camera";

void Log(String input)
{
    std::cout << input << std::endl;
}

String getCurrentDateTime()
{
    time_t t = time(0);
    struct tm now;
    localtime_s(&now, &t);
    char buffer[80];
    strftime(buffer, 80, "%Y-%m-%d-%H-%M-%S", &now);

    return buffer;
}

void makeScreenShot(cv::Mat& frame)
{
    String dirname = std::format("{}_Calibration_Images", CAMERA_NAME);

    // check if directory exists
    if (!std::filesystem::exists(dirname))
    {
        // if not then make it
        std::filesystem::create_directory(dirname);
    }

    // create filename based on current date
    String datetime = getCurrentDateTime();
    String filename = std::format("{0}/{1}.png", dirname, datetime);
    Log(std::format("Created the following file: {0}.png in directory {1}", datetime, dirname));
    cv::imwrite(filename, frame);
}

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

        // press spacebar for screenshot
        if(GetAsyncKeyState(32))
        {
            makeScreenShot(frame);
        }

        if (cv::waitKey(10) == 27) break;
    }

    videoCapture.release();
    return 0;

}
//
//int main()
//{
//    //std::string image_path = samples::findFile("starry_night.jpg");
//    //Mat img = imread(image_path, IMREAD_COLOR);
//    //if (img.empty())
//    //{
//    //    std::cout << "could not read the image: " << image_path << std::endl;
//    //    return 1;
//    //}
//    //imshow("display window", img);
//    //int k = waitKey(0); // wait for a keystroke in the window
//    //if (k == 's')
//    //{
//    //    imwrite("starry_night.png", img);
//    //}
//
//    streamWebcamFeed();
//
//    return 0;
//}