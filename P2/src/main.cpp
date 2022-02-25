#include <cstdlib>
#include <string>

#include "utilities/General.h"
#include "VoxelReconstruction.h"
#include "Video.h"

using namespace nl_uu_science_gmt;
using namespace cv;
using namespace std;

int main(
		int argc, char** argv)
{
	for (int i = 1; i < 5; i++)
	{
		Video vid = Video(std::format("./data/cam{}", i), General::IntrinsicsVideo);
		vid.getFrames(50, std::format("./data/cam{}/intrinsics", i));
	}

	

	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;	
}