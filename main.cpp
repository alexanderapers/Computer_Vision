//// main.cpp
//
//
//
#include "precomp.h"

void adjustXMLFiles(XMLData& default_file, XMLData& images_file, string file_To_Leave);
double getAvgReprojectionError(XMLData& camera_output);
void fullCalibration(int argc, char** argv);
tuple<Mat, Mat, Mat> getParameters();

static inline void read(const FileNode& node, CalibrationSettings& x, const CalibrationSettings& default_value = CalibrationSettings())
{
    if (node.empty())
        x = default_value;
    else
        x.read(node);
}

int main(int argc, char** argv)
{
    if (string(argv[1]) == "screencapture")
    { 
        ScreenCapture::streamWebcamFeed();
    }

    if (string(argv[1]) == "offline")
    { 
        fullCalibration(argc, argv);
    }

    if (string(argv[1]) == "online")
    {
        CalibrationSettings s;
        FileStorage fs("default.xml", FileStorage::READ);

        fs["Settings"] >> s;
        fs.release();

        // for frame in framestream
        Mat view = s.nextImage();

        tuple<Mat, Mat, Mat> parameters = getParameters();
        auto [camera_matrix, extrinsic_parameters, distCoeffs] = parameters;
        
        vector<Point3f> points;
        points.push_back(Point3d(0, 0, 0));
        points.push_back(Point3d(1, 0, 0));
        points.push_back(Point3d(0, 1, 0));
        points.push_back(Point3d(0, 0, 1));
        points.push_back(Point3d(1, 1, 0));
        points.push_back(Point3d(0, 1, 1));
        points.push_back(Point3d(1, 0, 1));
        points.push_back(Point3d(1, 1, 1));

        Mat p(points);
        // I think something is off with the size of the matrices
        cout << p.channels() << endl; // it seems to make 3 channels instead of a 8x3 matrix
        cout << p.size().height << " " << p.size().width << endl;

        vector<Point2d> pointBuf;
        Mat rvec, tvec;
        bool found = findChessboardCorners(view, s.boardSize, pointBuf);
        Mat i(pointBuf);
        if (found)
        {
            solvePnP(p, i, camera_matrix, distCoeffs, rvec, tvec);
        }

        cout << rvec << endl;
        cout << tvec << endl;

    }

    return 0;
}

tuple<Mat, Mat, Mat> getParameters()
{
    FileStorage fs("out_camera_data.xml", FileStorage::READ);
    
    // get camera matrix
    Mat camera_matrix;
    fs["camera_matrix"] >> camera_matrix;

    // get extrinsic parameters
    Mat extrinsic_parameters;
    fs["extrinsic_parameters"] >> extrinsic_parameters;

    // get distortion coefficients
    Mat distCoeffs;
    fs["distortion_coefficients"] >> distCoeffs;

    fs.release();

    tuple<Mat, Mat, Mat> return_tuple(camera_matrix, extrinsic_parameters, distCoeffs);
    return return_tuple;
}

void fullCalibration(int argc, char** argv)
{
    // get recalibration error of using all images
    Log("Calibrating with all images... \n");
    CameraCalibration::calibrate(argc, argv, "default.xml");
    XMLData camera_output = XMLData::XMLData("./", "out_camera_data.xml", true);
    double original_error = getAvgReprojectionError(camera_output);

    XMLData default_file("./", "default.xml", true);
    XMLData images_file("./", "VID5.xml", true);

    // set up iterator
    FileStorage fs("VID5.xml", FileStorage::READ);
    FileNodeIterator it = fs["images"].begin(), it_end = fs["images"].end();

    Log("\nCalibrating with n-1 images... \n");
    vector<string> toKeep;

    // for image in calibimages run calibration without this images
    for (; it != it_end; ++it)
    {
        adjustXMLFiles(default_file, images_file, (string)*it);

        CameraCalibration::calibrate(argc, argv, "n-1_default.xml");

        XMLData camera_output = XMLData::XMLData("./", "out_camera_data.xml", true);
        double new_error = getAvgReprojectionError(camera_output);

        if (new_error < original_error)
        {
            Log("Removing this image might improve calibration: " + (string)*it);
        }
        else
        {
            toKeep.push_back((string)*it);
        }
    }


    const char* const delim = "\n";
    std::ostringstream imploded;
    std::copy(toKeep.begin(), toKeep.end(),
        std::ostream_iterator<std::string>(imploded, delim));

    Log("\nCalibrating with outliers removed");

    XMLData new_images("./", "n-1_images.xml", true);
    new_images.writeValue("images", imploded.str());
    new_images.save();

    CameraCalibration::calibrate(argc, argv, "n-1_default.xml");
}

void adjustXMLFiles(XMLData& default_file, XMLData& images_file, string file_To_Leave)
{
    XMLData new_default_file = XMLData("./", "n-1_default.xml", default_file);
    XMLData new_images_file = XMLData("./", "n-1_images.xml", images_file);

    // decrement value of number of images in default
    string number_of_images_str = default_file.readValue("Calibrate_NrOfFrameToUse");
    int number_of_images = stoi(number_of_images_str);
    number_of_images--;
    number_of_images_str = to_string(number_of_images);

    new_default_file.writeValue("Calibrate_NrOfFrameToUse", number_of_images_str);

    // adjust file path to images locations
    new_default_file.writeValue("Input", "\"n-1_images.xml\"");

    // leave one image out in images
    file_To_Leave.append("\n");
    string all_images = new_images_file.readValue("images");
    size_t occurrence = all_images.find(file_To_Leave);
    string before = all_images.substr(0, occurrence);
    string after = all_images.substr(occurrence + file_To_Leave.size(), all_images.size());
    new_images_file.writeValue("images", std::format("{0}{1}", before, after));

    new_default_file.save();
    new_images_file.save();
}

double getAvgReprojectionError(XMLData& camera_output)
{
    /* Reads the average reprojection error from an output camera data xml file
    input:
          pathtooutfile: path to directory where out camera data is
    output:
          double: average reprojection error
    */

    string error_str = camera_output.readValue("avg_reprojection_error");
    double error = stod(error_str);

    return error;
}