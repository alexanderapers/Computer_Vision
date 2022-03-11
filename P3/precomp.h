/*
 * Precomp.h
 *
 *  Created on: March 5th, 2022
 *      Author: robertoost
 */

// Include any header dependencies in this file to enjoy the benefits of lightning-fast compilation times.

/* USAGE:	- In any source (.cpp) files that may depend on the precompiled header file, 
			add #include precomp.h as the first include line.
			- For readability, add any other dependencies as if this header file doesn't exist.
			This will result in duplicate #include statements, but the usage of #pragma once 
			at the start of other source header files will prevent this from being a problem.

			!!! WARNING: Never include this file in other header files.
*/			

// ===========================================
// PLATFORM SPECIFIC HEADERS
// ===========================================
#ifdef _WIN32
#include <Windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#ifdef __linux__
#include <GL/glut.h>
#include <GL/glu.h>
#endif


// ===========================================
// STANDARD LIBRARY HEADERS
// ===========================================
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <stddef.h>
#include <vector>
#include <unordered_set>

// ===========================================
// OPENCV HEADERS
// ===========================================
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/ml.hpp>

// ===========================================
// NAMESPACES
// ===========================================
// "Leaking" namespaces so they can be used globally.
// -- It's not standard practice, but it makes things easier to use.
using namespace std;
using namespace cv;

// ===========================================
// TEMPLATE API HEADERS
// ===========================================
#include "./src/utilities/General.h" // GENERAL UTILITIES
#include "./src/controllers/arcball.h"
#include "./src/calibration/Grid.h"
#include "./src/controllers/Reconstructor.h"

// ===========================================
// STABLE SOURCE HEADER FILES
// ===========================================
// Include files that are not likely to change.
#include "./src/controllers/Video.h"
#include "./src/controllers/Camera.h"
#include "./src/background/gaussian/Gaussian.h"

// ===========================================
// UNSTABLE PROJECT HEADER FILES
// ===========================================
// Do not include these files until they're not likely to change anymore.

//#include "./src/controllers/Scene3DRenderer.h"
//#include "./src/VoxelReconstruction.h"
//#include "./src/background/gaussian/GaussianBackgroundSubtraction.h"
//#include "./src/background/mog2/MOG2BackgroundSubtraction.h"
