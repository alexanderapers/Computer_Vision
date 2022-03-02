/*
 * General.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#include "General.h"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

namespace nl_uu_science_gmt
{

const string General::CBConfigFile         = "checkerboard.xml";
const string General::CalibrationVideo     = "calibration.avi";
const string General::CheckerboadVideo     = "checkerboard.avi";
const string General::BackgroundImageFile  = "background.png";
const string General::BackgroundSDImageFile = "background_sd.png";
const string General::VideoFile            = "video.avi";
const string General::IntrinsicsFile       = "intrinsics.xml";
const string General::CheckerboadCorners   = "boardcorners.xml";
const string General::ConfigFile           = "config.xml";

const string General::IntrinsicsVideo	   = "intrinsics.avi";
const string General::BackgroundVideo      = "background.avi";

/**
 * Linux/Windows friendly way to check if a file exists
 */
bool General::fexists(const std::string &filename)
{
	ifstream ifile(filename.c_str());
	return ifile.is_open();
}

void General::log(const std::string &inp)
{
	std::cout << inp << std::endl;
}

void General::writeIntrinsics(const std::string read_file_path, const std::string write_file_path)
{
	FileStorage fs_read(read_file_path, FileStorage::READ);
	Mat camera_matrix;
	fs_read["camera_matrix"] >> camera_matrix;
	Mat distortion_coeffs;
	fs_read["distortion_coefficients"] >> distortion_coeffs;

	FileStorage fs_write(write_file_path, FileStorage::WRITE);
	fs_write << "CameraMatrix" << camera_matrix;
	fs_write << "DistortionCoeffs" << distortion_coeffs;
}

} /* namespace nl_uu_science_gmt */
