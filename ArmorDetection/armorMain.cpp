#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cstring>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "armorDetect.h"
#define DEBUG 1
#define VIDEO 1
using namespace cv;
using namespace std;

void setLabel(cv::Mat &im, const std::string label, const cv::Point & pos, const cv::Scalar &colpos)
{
    const int fontface = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.75;
    const int thickness = 2;
    int baseline = 0;
    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, pos + cv::Point(0, baseline), pos + cv::Point(text.width, -text.height), CV_RGB(0,0,0), FILLED);
    cv::putText(im, label, pos, fontface, scale, colpos, thickness, 8);
}

int main (int argvc, char ** argv)
{
    char text_buffer[100];
#if VIDEO
    cout << "Camera ID = ";
    int camera_id;
    cin >> camera_id;
    VideoCapture cap;
    cap.open(camera_id);
    while(!cap.isOpened())
    {
        cout << "Unable to open the specified camera." << endl;
        cout << "(Ctrl-C to exit) Retry. Camera ID = ";
        cin >> camera_id;
        cap.open(camera_id);
    }
    vector<pair<vector<Point2f>, float>> ROI;
    Mat inputFrame;
    ArmorDetect detector = ArmorDetect();
    while (true)
    {
        double t = (double)getTickCount();
        if(!cap.read(inputFrame))
        {
            cout << "Fail to read frame" << endl;
            return 1;
        }
        detector.process(inputFrame, ROI);
        const double latency = ((double)getTickCount() - t) / getTickFrequency();
        const double fps = 1 / latency;
#if DEBUG
        if (ROI.size() != 0)
        {
            for (int i = 0; i < 1; i++)
            {
                line( inputFrame, ROI[i].first[0], ROI[i].first[1], Scalar (0, 255, 255), 3, 8 );
                line( inputFrame, ROI[i].first[1], ROI[i].first[3], Scalar (0, 255, 255), 3, 8 );
                line( inputFrame, ROI[i].first[3], ROI[i].first[2], Scalar (0, 255, 255), 3, 8 );
                line( inputFrame, ROI[i].first[2], ROI[i].first[0], Scalar (0, 255, 255), 3, 8 );
            }
        }
        sprintf(text_buffer, "Latency = %.3f ms", latency * 1000);
        setLabel(inputFrame, text_buffer, Point(50, 50), Scalar(255, 255, 255));
        sprintf(text_buffer, "%.1f FPS", fps);
        setLabel(inputFrame, text_buffer, Point(50, 80), Scalar(255, 255, 255));
        namedWindow ("debug4", WINDOW_NORMAL);
        imshow ("debug4", inputFrame);
#endif
        if(waitKey(1) == 27)
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
        cout << "fps: " << fps << endl;
    }
#else
    Mat inputImage;
    vector<pair<vector<Point2f>, float>> ROI;
    inputImage = imread (argv[1], IMREAD_COLOR);
    ArmorDetect detector = ArmorDetect();
    detector.process ( inputImage, ROI );
#if DEBUG
    for (int i = 0; i < ROI.size(); i++)
    {
        line( inputImage, ROI[i].first[0], ROI[i].first[1], Scalar (0, 255, 255), 3, 8 );
        line( inputImage, ROI[i].first[1], ROI[i].first[3], Scalar (0, 255, 255), 3, 8 );
        line( inputImage, ROI[i].first[3], ROI[i].first[2], Scalar (0, 255, 255), 3, 8 );
        line( inputImage, ROI[i].first[2], ROI[i].first[0], Scalar (0, 255, 255), 3, 8 );
    }
    namedWindow ("debug4", WINDOW_NORMAL);
    imshow ("debug4", inputImage);
#endif
#endif
    waitKey(0);
    return 0;

}
