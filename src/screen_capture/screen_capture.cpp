#include "../../precomp.h"

// TODO: put Log and getCurrentDateTime in separate util classes.
inline void Log(String input)
{
    cout << input << endl;
}

inline String getCurrentDateTime()
{
    time_t t = time(0);
    struct tm now;
    localtime_s(&now, &t);
    char buffer[80];
    strftime(buffer, 80, "%Y-%m-%d-%H-%M-%S", &now);

    return buffer;
}

/// <summary>
/// Takes a screenshot of the current camera frame.
/// </summary>
/// <param name="frame">The current camera frame</param>
void ScreenCapture::makeScreenShot(Mat& frame)
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
    imwrite(filename, frame);
}

/// <summary>
/// Streams the webcam feed in a little window and allows for screenshots.
/// </summary>
/// <returns></returns>
int ScreenCapture::streamWebcamFeed()
{
    VideoCapture videoCapture(0);

    if (!videoCapture.isOpened())
    {
        std::cout << "Unable to connect to webcam" << std::endl;
        return -1;
    }

    while (true)
    {
        Mat frame;
        videoCapture >> frame;
        if (frame.empty()) break;
        imshow("Camera feed", frame);

        // press spacebar for screenshot
        if (GetAsyncKeyState(32))
        {
            makeScreenShot(frame);
        }

        if (waitKey(10) == 27) break;
    }

    videoCapture.release();
    return 0;
}