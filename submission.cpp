/*
 * File:      submission.cpp
 * Author:    Richard Purcell
 * Date:      2020-08-15
 * Version:   1.0
 * Purpose:   Detect and track a soccerball in the soccer-ball.mp4
 * Usage:     $ ./trackBall  ./pathToFile/soccer-ball.mp4
 * Notes:     Created for OpenCV's Computer Vision 1
 *            Using YOLO and KCF
 *            Based on supplied files:
 *                trackSingleObject.cpp
 *                object_detection_yolo.cpp
 */

#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

// parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // DNN's Width input
int inpHeight = 416; // DNN's Height input
vector<string> classes;
int detected = 0, tracked = 0;

/*
 * Name:         drawPred
 * Purpose:      Draw the predicted bounding box
 * Arguments:    int classId, float conf, int left, int top, int right, int bottom, Mat& frame
 * Outputs:      none
 * Modifies:     frame
 * Returns:      none
 * Assumptions:  none
 * Bugs:         ?
 * Notes:        based on provided file object_detection_yolo.cpp
 */
void 
drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

    //Get the label for the class name and its confidence
    string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);

    rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
}

/*
 * Name:         postprocess
 * Purpose:      Remove the bounding boxes with low confidence using non-maxima suppression
 * Arguments:    Mat& frame, const vector<Mat>& outs
 * Outputs:      none
 * Modifies:     frame (indirectly)
 * Returns:      none
 * Assumptions:  none
 * Bugs:         ?
 * Notes:        based on provided file object_detection_yolo.cpp
 */
void 
postprocess(Mat& frame, const vector<Mat>& outs)
{
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        if(classes[classIds[idx]] == "sports ball")
        {
            //tracked = 1;
            drawPred(classIds[idx], confidences[idx], box.x, box.y,
                box.x + box.width, box.y + box.height, frame);
        }     
    }
}

// Get the names of the output layers
/*
 * Name:         getOutputsNames
 * Purpose:      Get the names of the output layers
 * Arguments:    const Net& net
 * Outputs:      none
 * Modifies:     none
 * Returns:      names
 * Assumptions:  none
 * Bugs:         ?
 * Notes:        based on provided file object_detection_yolo.cpp
 */
vector<String> 
getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

int
main(int argc, char * argv[])
{
    // Load names of classes
    string classesFile = "./data/models/coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Give the configuration and weight files for the model
    String modelConfiguration = "./data/models/yolov3.cfg";
    String modelWeights = "./data/models/yolov3.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    
    // Open a video file.
    string str, outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;
    try 
    {
        outputFile = "submission_out.avi";
        // Open the video file (toDo: change hard coded path)
        str = "./data/images/soccer-ball.mp4";
        ifstream ifile(str);
        if (!ifile) throw("error");
        cap.open(str);
        str.replace(str.end()-4, str.end(), "_submission_out.avi");
        outputFile = str; 
    }
    catch(...) 
    {
        cout << "Could not open the input video stream" << endl;
        return 0;
    }

    // initialize video writer
    video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

    // Create a window
    static const string kWinName = "Detect and track soccer ball";
    namedWindow(kWinName, WINDOW_NORMAL);

    // Process frames.
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
        }

        if(detected == 0 && tracked == 0)
        {
            // Create a 4D blob from a frame.
            blobFromImage(frame, blob, 1/255.0, Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);

            //Sets the input to the network
            net.setInput(blob);

            // Runs the forward pass to get output of the output layers
            vector<Mat> outs;
            net.forward(outs, getOutputsNames(net));

            // Remove the bounding boxes with low confidence
            postprocess(frame, outs);
        }

        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        video.write(detectedFrame);

        imshow(kWinName, frame);

    }

    cap.release();
    video.release();

    return 0;
}
