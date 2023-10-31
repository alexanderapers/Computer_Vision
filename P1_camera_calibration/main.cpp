//// main.cpp
//
//
//
#include "precomp.h"

void adjustXMLFiles(XMLData& default_file, XMLData& images_file, string file_To_Leave);
double getAvgReprojectionError(XMLData& camera_output);
void fullCalibration(int argc, char** argv);
void drawViewAxes(Mat view, CalibrationSettings s, Mat rvec, Mat tvec, Mat camera_matrix, Mat distCoeffs);
void drawCube(Mat view, CalibrationSettings s, Mat rvec, Mat tvec, Mat camera_matrix, Mat distCoeffs);

tuple<Mat, Mat, Mat> getParameters();
//void projectionFromKRt(Mat K, Mat R, Mat t, Mat& P);

/// <summary>
/// Reads the calibration settings.
/// </summary>
static inline void read(const FileNode& node, CalibrationSettings& x, const CalibrationSettings& default_value = CalibrationSettings())
{
    if (node.empty())
        x = default_value;
    else
        x.read(node);
}

/// <summary>
/// Main function, executes the program. Can take three different arguments: 
/// "screencapture": Allows you to take pictures using your camera. (Using spacebar.)
/// "offline": Uses a list of pictures (manually defined in VID5.xml) to do calibration for your camera.
/// "online": Shows the camera feed in realtime, and draws a 3D cube or axes onto a checkerboard. (Can be switched using spacebar.)
/// </summary>
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

        VideoCapture videoCapture(0);

        if (!videoCapture.isOpened())
        {
            std::cout << "Unable to connect to webcam" << std::endl;
            return -1;
        }

        tuple<Mat, Mat, Mat> parameters = getParameters();
        auto [camera_matrix, extrinsic_parameters /* <-- We don't need these yet*/, distCoeffs] = parameters;

        bool drawAxes = false;

        while (true)
        {
            Mat frame;
            videoCapture >> frame;
            if (frame.empty()) {
                break;
            };
            //imshow("View", frame);
            // press spacebar for screenshot
            if (GetAsyncKeyState(32))
            {
                drawAxes = !drawAxes;
            }

            vector<Point2d> imagePoints;
            int chessBoardFlags = (CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE) | CALIB_CB_FAST_CHECK;
            bool found = findChessboardCorners(frame, s.boardSize, imagePoints, chessBoardFlags);

            if (found)
            {
                float grid_width = s.squareSize * (s.boardSize.width - 1);
                vector<vector<Point3f> > objectPoints(1);
                CameraCalibration::calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);
                objectPoints[0][s.boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
                vector<Point3f> newObjPoints = objectPoints[0];
                objectPoints.resize(imagePoints.size(), objectPoints[0]);
                Mat rvec, tvec;
                solvePnPRansac(newObjPoints, imagePoints, camera_matrix, distCoeffs, rvec, tvec);
                if (drawAxes) {
                    drawViewAxes(frame, s, rvec, tvec, camera_matrix, distCoeffs);
                }
                else {
                    drawCube(frame, s, rvec, tvec, camera_matrix, distCoeffs);
                }
            }
            imshow("View", frame);
            if (waitKey(10) == 27) break;
        }

        videoCapture.release();
    }

    return 0;
}

/// <summary>
/// Draws three axes onto a view containing a checkerboard.
/// </summary>
/// <param name="view">The image to draw on.</param>
/// <param name="s">The calibration settings.</param>
/// <param name="rvec">The rotation vector.</param>
/// <param name="tvec">The translation vector.</param>
/// <param name="camera_matrix">The camera matrix.</param>
/// <param name="distCoeffs">The distortion coefficients.</param>
void drawViewAxes(Mat view, CalibrationSettings s, Mat rvec, Mat tvec, Mat camera_matrix, Mat distCoeffs) {
    // Origin (0,0,0)
    // X axis (0,0,0)->(3,0,0)
    // Y axis (0,0,0)->(0,3,0)
    // Z axis (0,0,0)->(0,0,-3)
    vector<Point3d> axes = vector<Point3d>{ Point3d(0,0,0), Point3d(3,0,0), Point3d(0,3,0), Point3d(0,0,-3) };
    Mat axesPoints = s.squareSize * Mat(axes);
    vector<Point2d> axesImagePoints;
    projectPoints(axesPoints, rvec, tvec, camera_matrix, distCoeffs, axesImagePoints);

    cv::arrowedLine(view, axesImagePoints[0], axesImagePoints[1], Scalar(255, 0, 0), 3);
    cv::arrowedLine(view, axesImagePoints[0], axesImagePoints[2], Scalar(0, 255, 0), 3);
    cv::arrowedLine(view, axesImagePoints[0], axesImagePoints[3], Scalar(0, 0, 255), 3);
}

typedef std::tuple<float, vector<Point>, Scalar> face_data;
inline bool comparator(const face_data& l, const face_data& r)
{
    return get<0>(l) < get<0>(r);
}

/// <summary>
/// Draws a cube out of filled polygons onto a view containing a checkerboard.
/// </summary>
/// <param name="view">The image to draw on.</param>
/// <param name="s">The calibration settings.</param>
/// <param name="rvec">The rotation vector.</param>
/// <param name="tvec">The translation vector.</param>
/// <param name="camera_matrix">The camera matrix.</param>
/// <param name="distCoeffs">The distortion coefficients.</param>
void drawCube(Mat view, CalibrationSettings s, Mat rvec, Mat tvec, Mat camera_matrix, Mat distCoeffs) {
    vector<Point3d> cube = vector<Point3d>{
    Point3d(0,0,0), Point3d(3,0,0), Point3d(0,3,0), Point3d(0,0,-3),
    Point3d(3,3,0), Point3d(0,3,-3), Point3d(3,0,-3), Point3d(3,3,-3) };
    Mat cube_points = s.squareSize * Mat(cube);
    vector<Point2d> cube_image_points;
    projectPoints(cube_points, rvec, tvec, camera_matrix, distCoeffs, cube_image_points);

    vector<Point> bottom = { cube_image_points[0], cube_image_points[1], cube_image_points[4], cube_image_points[2] };
    vector<Point> top = { cube_image_points[3], cube_image_points[5], cube_image_points[7], cube_image_points[6] };
    vector<Point> back = { cube_image_points[0], cube_image_points[1], cube_image_points[6], cube_image_points[3] };
    vector<Point> front = { cube_image_points[4], cube_image_points[2], cube_image_points[5], cube_image_points[7] };
    vector<Point> left = { cube_image_points[2], cube_image_points[0], cube_image_points[3], cube_image_points[5] };
    vector<Point> right = { cube_image_points[1], cube_image_points[4], cube_image_points[7], cube_image_points[6] };

    Mat rotation;
    Rodrigues(rvec, rotation);
    Mat transformed_cam_pos = -rotation.t() * tvec;

    Point3d point = Point3d(transformed_cam_pos);

    Point3d bottom_center = (cube[0] + cube[1] + cube[4] + cube[2]) / 4;
    Point3d top_center = (cube[3] + cube[5] + cube[7] + cube[6]) / 4;
    Point3d back_center = (cube[0] + cube[1] + cube[6] + cube[3]) / 4;
    Point3d front_center = (cube[4] + cube[2] + cube[5] + cube[7]) / 4;
    Point3d left_center = (cube[2] + cube[0] + cube[3] + cube[5]) / 4;
    Point3d right_center = (cube[1] + cube[4] + cube[7] + cube[6]) / 4;
    
    // Gather distances to face centers along with their respective image coordinates and color values.
    vector<face_data> sorted_face_distances = {
        {cv::norm(point - bottom_center), bottom, Scalar(0, 255, 255)}, // The bottom face is yellow and will never be shown due to occlusion.
        {cv::norm(point - top_center), top, Scalar(255, 0, 0)},
        {cv::norm(point - back_center), back, Scalar(0, 255, 0)},
        {cv::norm(point - front_center), front, Scalar(255, 255, 0)},
        {cv::norm(point - left_center), left, Scalar(255, 0, 255)},
        {cv::norm(point - right_center), right, Scalar(0, 0, 255)},
    };

    // Sort the faces so that the least distant face is drawn last.
    std::sort(sorted_face_distances.rbegin(), sorted_face_distances.rend(), comparator);

    for (face_data face_d : sorted_face_distances) {
        fillPoly(view, get<1>(face_d), get<2>(face_d));
    }

    // Wireframe cube code
    //cv::line(view, cube_image_points[0], cube_image_points[1], Scalar(255, 0, 0), 2);
    //cv::line(view, cube_image_points[1], cube_image_points[4], Scalar(255, 0, 0), 2);
    //cv::line(view, cube_image_points[4], cube_image_points[2], Scalar(255, 0, 0), 2);
    //cv::line(view, cube_image_points[2], cube_image_points[0], Scalar(255, 0, 0), 2);


    //cv::line(view, cube_image_points[0], cube_image_points[3], Scalar(0, 0, 255), 2);
    //cv::line(view, cube_image_points[1], cube_image_points[6], Scalar(0, 0, 255), 2);
    //cv::line(view, cube_image_points[4], cube_image_points[7], Scalar(0, 0, 255), 2);
    //cv::line(view, cube_image_points[2], cube_image_points[5], Scalar(0, 0, 255), 2);

    //cv::line(view, cube_image_points[3], cube_image_points[5], Scalar(0, 255, 0), 2);
    //cv::line(view, cube_image_points[5], cube_image_points[7], Scalar(0, 255, 0), 2);
    //cv::line(view, cube_image_points[7], cube_image_points[6], Scalar(0, 255, 0), 2);
    //cv::line(view, cube_image_points[6], cube_image_points[3], Scalar(0, 255, 0), 2);
}

/// <summary>
/// Gets the camera matrix, extrinsic parameters, and distortion coefficients from the calibration output.
/// </summary>
/// <returns>A tuple containing:
/// 1: The camera matrix.
/// 2: The extrinsic parameters.
/// 3: The distortion coeficcients.
/// </returns>
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

/// <summary>
/// Runs a full camera calibration. First using all N images, then N-1 for each image. Then finally with N-rejected images.
/// </summary>
void fullCalibration(int argc, char** argv)
{
    // get reprojection error of using all images
    Log("Calibrating with all images... \n");
    CameraCalibration::calibrate(argc, argv, "default.xml");
    XMLData camera_output = XMLData::XMLData("./", "cam1_out_camera_data.xml", true);
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

        XMLData camera_output = XMLData::XMLData("./", "cam1_out_camera_data.xml", true);
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

/// <summary>
/// Adjusts the various xml files used during calibration.
/// </summary>
/// <param name="default_file">The default calibration settings file used by calibration.cpp.</param>
/// <param name="images_file">The images file that the default file links to.</param>
/// <param name="file_To_Leave">The image file path that needs to be left out for the next n-1 calibration run.</param>
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

/// <summary>
/// Returns the average reprojection error from the camera output.
/// </summary>
/// <param name="camera_output">The xml file containing the camera calibration output.</param>
/// <returns></returns>
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