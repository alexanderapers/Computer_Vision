#include <cstdlib>
#include <string>
#include <iostream>
#include <filesystem>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "Video.h"

using namespace nl_uu_science_gmt;
using namespace cv;
using namespace std;

int main(
		int argc, char** argv)
{
	// GETS FRAMES FROM INTRINSICS VIDEO AND SAVES THEM INTO FILE
	/*for (int i = 1; i < 5; i++)
	{
		Video vid = Video(std::format("./data/cam{}", i), General::IntrinsicsVideo);
		vid.getFrames(50, std::format("./data/cam{}/intrinsics", i));
	}*/  


	// READS INTRINSICS FROM THE FILES AND WRITES THEM TO INTRINSICS.XML
	//for (int i = 1; i < 5; i++)
	//{
	//	const std::string input_file_path = std::format("..\\P1\\cam{}_out_camera_data.xml", i);
	//	const std::string output_file_path = std::format("./data/cam{}/", i) + General::IntrinsicsFile;
	//	General::writeIntrinsics(input_file_path, output_file_path);
	//}


	

	//VoxelReconstruction::showKeys();
	//VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	//vr.run(argc, argv);

	return EXIT_SUCCESS;	
}