#ifndef ARMORFIND_H
#define ARMORFIND_H

#include <iostream>
#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

static inline bool RotateRectSort(RotatedRect a1,RotatedRect a2)
{
    return a1.center.x < a2.center.x;
}

static inline bool roiSort(pair<vector<Point2f>, float> A, pair<vector<Point2f>, float> B)
{
    return A.second > B.second;
}

class ArmorDetect
{
public:
    ArmorDetect();
    void process(const Mat &input, vector<pair<vector<Point2f>, float>> &resultROI);
private:
    void preprocess( const Mat &input );
    void ContourCenter(const vector<Point> contour,Point &center);
    double Pointdis(const Point &p1,const Point &p2);
    void Clear(void);
    void GetArmorLights(void);
    void GetArmors();
    double GetK(Point2f L1,Point2f L2);

public:
    Mat bgr[3], binary;
    vector<pair<vector<Point2f>, float>> roi, roiOld;
private:
    int ArmorLostDelay;
    Mat kernel;
    vector<vector<Point> > contours;
    vector<vector<RotatedRect> > Armorlists;
    vector<int> CellMaxs;
    vector<Rect> Target;
    vector<Point> centers;
    vector<RotatedRect> RectfirstResult,RectResults;
};


#endif // ARMORFIND_H

