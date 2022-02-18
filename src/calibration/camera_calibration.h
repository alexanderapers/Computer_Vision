#pragma once

// ----------------------------------------------------------------------------------------------------//
// CAMERA CALIBRATION CODE FROM: https://docs.opencv.org/4.2.0/d4/d94/tutorial_camera_calibration.html //
// ----------------------------------------------------------------------------------------------------//

static class CameraCalibration {
public:
	static int calibrate(int argc, char* argv[]);
private:
	static double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
		const vector<vector<Point2f> >& imagePoints,
		const vector<Mat>& rvecs, const vector<Mat>& tvecs,
		const Mat& cameraMatrix, const Mat& distCoeffs,
		vector<float>& perViewErrors, bool fisheye);

	static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
		CalibrationSettings::Pattern patternType /*= Settings::CHESSBOARD*/);

	static bool runCalibration(CalibrationSettings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
		vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
		vector<float>& reprojErrs, double& totalAvgErr, vector<Point3f>& newObjPoints,
		float grid_width, bool release_object);

	static bool runCalibrationAndSave(CalibrationSettings& s, Size imageSize, Mat& cameraMatrix, Mat& distCoeffs,
		vector<vector<Point2f> > imagePoints, float grid_width, bool release_object);

	static void saveCameraParams(CalibrationSettings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
		const vector<Mat>& rvecs, const vector<Mat>& tvecs,
		const vector<float>& reprojErrs, const vector<vector<Point2f> >& imagePoints,
		double totalAvgErr, const vector<Point3f>& newObjPoints);
};