#pragma once
class ScreenCapture
{
public:
	static int streamWebcamFeed();
private:
	static void makeScreenShot(Mat& frame);

};

