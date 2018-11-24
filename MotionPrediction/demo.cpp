#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <cstring>
#include <string>

void setLabel(cv::Mat &im, const std::string label, const cv::Point & pos, const cv::Scalar &colpos)
{
    const int fontface = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.75;
    const int thickness = 2;
    int baseline = 0;
    cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
    cv::rectangle(im, pos + cv::Point(0, baseline), pos + cv::Point(text.width, -text.height), CV_RGB(0,0,0), CV_FILLED);
    cv::putText(im, label, pos, fontface, scale, colpos, thickness, 8);
}

void drawCross(cv::Mat &img, const cv::Point &center, const cv::Scalar &color, const int d)
{
    cv::line(img, cv::Point( center.x - d, center.y - d ), cv::Point( center.x + d, center.y + d ), color, 2, CV_AA, 0);
    cv::line(img, cv::Point( center.x + d, center.y - d ), cv::Point( center.x - d, center.y + d ), color, 2, CV_AA, 0);
}

int main(int argc, char *argv[])
{
    if(argc < 6)
    {
        std::cout << "Usage: ./a.out <video file name> <x> <y> <width> <height>" << std::endl;
        return 1;
    }
    cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
    cv::VideoCapture video(argv[1]);
    if(!video.isOpened())
    {
        std::cout << "Could not read video file" << std::endl;
        return 1;
    }
    cv::Mat frame;
    video.read(frame);
    int x, y, width, height;
    x = atoi(argv[2]);
    y = atoi(argv[3]);
    width = atoi(argv[4]);
    height = atoi(argv[5]);
    cv::Rect2d bbox(x, y, width, height);

    // Initialize Kalman Filter
    cv::KalmanFilter kf(4, 2, 0);
    kf.transitionMatrix = (cv::Mat_<float>(4, 4) << 1,0,1,0,   0,1,0,1,  0,0,1,0,  0,0,0,1);
    cv::Mat_<float> measurement(2, 1);
    measurement.setTo(cv::Scalar(0));
    kf.statePre.at<float>(0) = x + width / 2;
    kf.statePre.at<float>(1) = y + height / 2;
    kf.statePre.at<float>(2) = 0.0f;
    kf.statePre.at<float>(3) = 0.0f;
    cv::setIdentity(kf.measurementMatrix);
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(10));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(0.1));
    measurement(0) = bbox.x + bbox.width / 2;
    measurement(1) = bbox.y + bbox.height / 2;
    kf.correct(measurement);

    float last_position[2];
    last_position[0] = measurement(0);
    last_position[1] = measurement(1);

    cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
    cv::imshow("Tracker & Prediction Demo M1", frame);
    cv::waitKey(0);
    tracker->init(frame, bbox);
    char buffer[100];
    while(video.read(frame))
    {
        // KCF Tracker
        double timer = (double)cv::getTickCount();
        bool intrack = tracker->update(frame, bbox);
        double tracker_latency = ((double)cv::getTickCount() - timer) / cv::getTickFrequency();
        
        measurement(0) = bbox.x + bbox.width / 2;
        measurement(1) = bbox.y + bbox.height / 2;

        // Kalman Filter Prediction
        cv::Mat prediction = kf.predict();
        drawCross(frame, cv::Point(prediction.at<float>(0), prediction.at<float>(1)), cv::Scalar(0, 255, 0), 5);
        cv::Rect2d prediction_box(prediction.at<float>(0) - bbox.width / 2, prediction.at<float>(1) - bbox.height / 2, bbox.width, bbox.height);
        cv::rectangle(frame, prediction_box, cv::Scalar(0, 255, 0), 2, 1);

        // Naive Prediction
        float dx = measurement(0) - last_position[0], dy = measurement(1) - last_position[1];
        drawCross(frame, cv::Point(measurement(0) + dx * 5, measurement(1) + dy * 5), cv::Scalar(0, 0, 255), 5);
        cv::Rect2d naive_box(measurement(0) + dx * 5 - bbox.width / 2, measurement(1) + dy * 5 - bbox.height / 2, bbox.width, bbox.height);
        cv::rectangle(frame, naive_box, cv::Scalar(0, 0, 255), 2, 1);
        last_position[0] = measurement(0);
        last_position[1] = measurement(1);

        if(intrack)
        {
            // Draw KCF Tracker box
            setLabel(frame, "KCF Tracker: In tracking", cv::Point(100, 100), cv::Scalar(255, 0, 0));
            cv::rectangle(frame, bbox, cv::Scalar(255, 0, 0), 2, 1);
            drawCross(frame, cv::Point(measurement(0), measurement(1)), cv::Scalar(255, 0, 0), 5);
            
            // Kalman Filter Correction
            kf.correct(measurement);
        }
        else
        {
            setLabel(frame, "KCF Tracker: Lost focus", cv::Point(100, 100), cv::Scalar(255, 0, 0));
        }

        sprintf(buffer, "KCF Tracker: Latency: %.4f sec", tracker_latency);
        setLabel(frame, buffer, cv::Point(100, 125), cv::Scalar(255, 0, 0));
        setLabel(frame, "Kalman Filter Prediction", cv::Point(100, 150), cv::Scalar(0, 255, 0));
        setLabel(frame, "Naive Prediction", cv::Point(100, 175), cv::Scalar(0, 0, 255));
        cv::imshow("Tracker & Prediction Demo M1", frame);
        if(cv::waitKey(0) == 27)
            break;
    }
    return 0;
}
