#include "HandwrittenRecognizer.h"

HandwrittenRecognizer::HandwrittenRecognizer()
{
    svm = cv::Algorithm::load<cv::ml::SVM>("HOG_SVM_DATA.xml");
}

HandwrittenRecognizer::~HandwrittenRecognizer()
{
    svm.release();
}

int HandwrittenRecognizer::recognize(const cv::Mat &image)
{
    cv::Mat img = image.clone();
    cv::resize(img, img, cv::Size(28, 28));
    cv::HOGDescriptor *hog = new cv::HOGDescriptor(cv::Size(28, 28), cv::Size(14, 14), cv::Size(7, 7), cv::Size(7, 7), 9);
    std::vector<float> descriptors;
    hog->compute(img, descriptors, cv::Size(1, 1), cv::Size(0, 0));
    cv::Mat train_mat(1, descriptors.size(), CV_32FC1);
    for(int i = 0; i < descriptors.size(); i++)
        train_mat.at<float>(0, i) = descriptors[i];
    delete hog;
    return svm->predict(train_mat);
}
