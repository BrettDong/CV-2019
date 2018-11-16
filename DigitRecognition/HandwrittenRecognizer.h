#pragma once
#ifndef _HANDWRITTENRECOGNIZER_H_
#define _HANDWRITTENRECOGNIZER_H_

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

class HandwrittenRecognizer
{
    private:
        cv::Ptr<cv::ml::SVM> svm;
    public:
        HandwrittenRecognizer();
        ~HandwrittenRecognizer();
        int recognize(const cv::Mat &image);
};

#endif
