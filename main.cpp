//// main.cpp
//
//
//
#include "precomp.h"

//double getAvgReprojectionError(string pathtooutfile);
//void removeCalibrationImage(string pathtoimagesfile, string fileToRemove);
//void adjustDefaultXML(const string pathtodefault);
//void createXMLFile(const string filename, const string content);
//string readContentFromFile(string pathttofile);
void adjustXMLFiles(XMLData& default_file, XMLData& images_file, string file_To_Leave);
double getAvgReprojectionError(XMLData& camera_output);


int main(int argc, char** argv)
{
    //ScreenCapture::streamWebcamFeed();

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


    return 0;
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