#include <iostream>
#include "HandwrittenRecognizer.h"
using namespace std;

int main(int argc, char *argv[])
{
    HandwrittenRecognizer recognizer;
    cv::Mat image;
    if(argc == 1)
        image = cv::imread("test.jpg");
    else
        image = cv::imread(argv[1]);
    cout << recognizer.recognize(image) << endl;
    return 0;
}
