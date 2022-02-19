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

void myprint(const string input)
{
    std::cout << input << std::endl;
}

int main(int argc, char** argv)
{
    //XMLData testFile = XMLData("./", "testXML.xml");
    //testFile.AddTag("opencv_storage");
    //testFile.AddTag("images", "opencv_storage");
    //testFile.writeValue("images", "\n\t\tRobert_Camera_Calibration_Images/2022-02-18-15-57-23.png\n");
    //testFile.save();

    //cout << testFile.readValue("images") << endl;

    //ScreenCapture::streamWebcamFeed();

    // get recalibration error of using all images
    CameraCalibration::calibrate(argc, argv);
    XMLData camera_output = XMLData("./", "out_camera_data.xml", true);
    string original_error_str = camera_output.readValue("avg_reprojection_error");
    double original_error = stod(original_error_str);
    cout << original_error << endl;

    //
    //FileStorage fs("VID5.xml", FileStorage::READ);
    //FileNodeIterator it = fs["images"].begin(), it_end = fs["images"].end();

    //adjustDefaultXML("default.xml");

    //// for image in calibimages run calibration without this images
    //for (; it != it_end; ++it)
    //{
    //    removeCalibrationImage("VID5.xml", (string)*it);
    //    
    //    CameraCalibration::calibrate(argc, argv);
    //    double new_error = getAvgReprojectionError("out_camera_data.xml");

    //    string content = readContentFromFile("VID5.xml");
    //    createXMLFile("n-1c.xml", content);

    //    if (new_error < original_error)
    //    {
    //        myprint("Removing this image might improve calibration: " + (string)*it);
    //    }
    //}



    return 0;
}

//void adjustDefaultXML(const string pathtodefault)
//{
//    string content = readContentFromFile(pathtodefault);
//    size_t found1 = content.find("<Calibrate_NrOfFrameToUse>");
//    int offset1 = found1 + string("<Calibrate_NrOfFrameToUse>").size();
//    string value1 = content.substr(offset1, 2);
//    int val = stoi(value1);
//    val--;
//    value1 = to_string(val);
//    for (int i = 0; i < value1.length(); i++)
//    {
//        content[offset1 + i] = value1[i];
//    }
//
//    size_t found2 = content.find("<Input>");
//    int offset2 = found2 + string("<Input>").size() + 1;
//    string value2 = content.substr(offset2, 8);
//    value2 = "n-1c.xml";
//
//    for (int j = 0; j < value2.length(); j++)
//    {
//        content[offset2 + j] = value2[j];
//    }
//
//    createXMLFile("default.xml", content);
//}

//double getAvgReprojectionError(const string pathtooutfile)
//{
//    /* Reads the average reprojection error from an output camera data xml file
//    input:
//          pathtooutfile: path to directory where out camera data is
//    output:
//          double: average reprojection error
//    */
//
//    FileStorage fs(pathtooutfile, FileStorage::READ);
//    double avg_reprojection_error;
//    fs["avg_reprojection_error"] >> avg_reprojection_error;
//    fs.release();
//
//    return avg_reprojection_error;
//}

//void removeCalibrationImage(const string pathtoimagesfile, const string fileToRemove)
//{
//   /* Creates a new.xml file with list of images that will be used for n-1 calibration
//    input: 
//        pathtoimagesfile: path to directory where original image list is
//        fileToRemove: name of file that should be removed during n-1 calibration
//    output:
//  */
//
//   FileStorage fs_read(pathtoimagesfile, FileStorage::READ);
//   FileNodeIterator it = fs_read["images"].begin(), it_end = fs_read["images"].end();
//
//   string alltext = 
//       "<?xml version = \"1.0\" ?>\n"
//       "<opencv_storage>\n"
//       "<images>\n";
//
//   for (; it != it_end; ++it)
//   {
//       if ((string)*it != fileToRemove)
//       {
//           alltext.append("\t");
//           alltext.append((string)*it);
//           alltext.append("\n");
//       }
//   }
//
//   fs_read.release();
//
//   alltext.append(
//       "</images>\n"
//       "</opencv_storage>\n"
//   );
//
//   createXMLFile("n-1c.xml", alltext);
//}