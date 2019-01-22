#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include "armorDetect.h"
#include <utility>
#define COLOR 0
#define DEBUG 0

#define BINTHRES 150

#define SIZEUPPER 10000
#define MINDIST 10

#define K 0.5
#define DIVSCALE 3.0
#define MAXANGLE 10.0
#define HWDIV 5.0
#define YDISRATIO 0.4
#define ANGLEABS 10.0//7.0

using namespace cv;
using namespace std;

ArmorDetect::ArmorDetect()
{
    ArmorLostDelay = 0;
    kernel = getStructuringElement(MORPH_ELLIPSE,Size(9,9));
}

void ArmorDetect::process(const Mat &input, vector<pair<vector<Point2f>, float>> &resultROI){
    Clear();
    preprocess(input);
    RotatedRect RRect;
    for(int i=0; i<contours.size(); i++){
        RRect = minAreaRect(contours[i]);
        if((fabs(RRect.angle) < 45.0 && RRect.size.height > RRect.size.width)|| (fabs(RRect.angle) > 45.0 && RRect.size.width > RRect.size.height)){
                RectfirstResult.push_back(RRect);
        }
    }
    #if DEBUG
    	Mat debug2 = input.clone();
    	for(int i=0; i<RectfirstResult.size(); i++){
    		Point2f rect_points[4]; 
    		RectfirstResult[i].points( rect_points );
       		for( int j = 0; j < 4; j++ )
          		line( debug2, rect_points[j], rect_points[(j+1)%4], Scalar (0, 0, 255), 5, 8 );
    	}
    	namedWindow ("debug2", WINDOW_NORMAL);
    	imshow ("debug2", debug2);
    #endif
    
    if(RectfirstResult.size() < 2){
        roi.clear();
        return;
    }
    sort(RectfirstResult.begin(),RectfirstResult.end(),RotateRectSort);
    GetArmorLights();
    sort(RectResults.begin(),RectResults.end(),RotateRectSort);
    #if DEBUG
    	Mat debug3 = input.clone();
    	for(int i=0; i<RectResults.size(); i++){
    		Point2f rect_points[4]; 
    		RectResults[i].points( rect_points );
       		for( int j = 0; j < 4; j++ )
          		line( debug3, rect_points[j], rect_points[(j+1)%4], Scalar (0, 0, 255), 5, 8 );
    	}
    	namedWindow ("debug3", WINDOW_NORMAL);
    	imshow ("debug3", debug3);
    #endif

    GetArmors();
    sort(roi.begin(),roi.end(),roiSort);
    if (roi.size() != 0) {
    	resultROI.clear();
    	resultROI = roi;
    }
}

void ArmorDetect::preprocess( const Mat &input ){
    Mat mask;
    split(input,bgr);
    cvtColor(input, binary ,COLOR_BGR2GRAY);

    threshold(binary,binary,100,255,THRESH_BINARY);
    
    #if COLOR == 1
        subtract(bgr[2],bgr[0],mask);
        threshold(mask,mask,BINTHRES,255,THRESH_BINARY);// red
    #else
        subtract(bgr[0],bgr[2],mask);
        threshold(mask,mask,BINTHRES,255,THRESH_BINARY);// blue
    #endif
    
    dilate (mask, mask, kernel);
    binary = binary & mask;
	findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	/*Can add closing and opening to further refine*/
	#if DEBUG
		Mat debug1 = input.clone();
		for( size_t i = 0; i< contours.size(); i++ )
    	{
        	drawContours( debug1, contours, (int)i, Scalar (0, 0, 255) , 10, LINE_8 );
    	}
    	namedWindow ("debug1", WINDOW_NORMAL);
    	imshow ("debug1", debug1);
	#endif
}

void ArmorDetect::GetArmorLights(){
    size_t size = RectfirstResult.size();
    vector<RotatedRect> Groups;
    int cellmaxsize;
    Groups.push_back(RectfirstResult[0]);
    cellmaxsize = RectfirstResult[0].size.height * RectfirstResult[0].size.width;
    if(cellmaxsize > SIZEUPPER) cellmaxsize = 0;
    int maxsize;
    
    /*Add y axis filter*/
    for(int i=1;i<size;i++){
        if (RectfirstResult[i].center.x - RectfirstResult[i-1].center.x < MINDIST) {
            maxsize = RectfirstResult[i].size.height * RectfirstResult[i].size.width;
            if(maxsize > SIZEUPPER) {continue;}
            if(maxsize > cellmaxsize) cellmaxsize = maxsize;
            Groups.push_back(RectfirstResult[i]);
        }
        else {
            Armorlists.push_back(Groups);
            //cout << cellmaxsize << endl;
            CellMaxs.push_back(cellmaxsize);
            cellmaxsize = 0;
            maxsize = 0;
            Groups.clear();
            Groups.push_back(RectfirstResult[i]);
            cellmaxsize = RectfirstResult[i].size.height * RectfirstResult[i].size.width;
        }
    }
    Armorlists.push_back(Groups);
    CellMaxs.push_back(cellmaxsize);
    
    size = Armorlists.size();
    //cout << CellMaxs.size() <<"  "<<Armorlists.size()<< endl;
    for(int i=0; i<size; i++){
        int Gsize = Armorlists[i].size();
        int GroupMax = CellMaxs[i];
        if(GroupMax > 5){
            for(int j=0;j<Gsize;j++){
                maxsize = Armorlists[i][j].size.height * Armorlists[i][j].size.width;
                if(maxsize == GroupMax){
                    RectResults.push_back(Armorlists[i][j]);
                }
            }
        }
    }
}


void ArmorDetect::GetArmors(){
    size_t size = RectResults.size();
    roiOld = roi;
    roi.clear();
    if(size < 2){
        return;
    }
    Point2f L1,L2;
    float k,angleabs = 0.0,angleL1,angleL2;
    float divscale,areaL1,areaL2;
    float ydis = 0;
    float maxangle,xdis,heightmax,hwdiv;
    Point2f _pt[4],pt[4];
    auto ptangle = [](const Point2f &p1,const Point2f &p2){
        return fabs(atan2(p2.y-p1.y,p2.x-p1.x)*180.0/CV_PI);
    };
    auto GetAreaofp3 = [](const Point2f &p1,const Point2f &p2,const Point2f &p3){
        Mat matrix = (Mat_<double>(3,3)<<p1.x,p1.y,1,p2.x,p2.y,1,p3.x,p3.y,1);
        return 0.5*determinant(matrix);
    };

    for(int i=0;i<size-1;i++){
        angleL1 = fabs(RectResults[i].angle);
        L1 = RectResults[i].center;
        areaL1 = RectResults[i].size.height * RectResults[i].size.width;
        RectResults[i].points(_pt);
         //pt
         // 0 2
         // 1 3
         // 
        if(angleL1 > 45.0){
        	pt[0] = _pt[3];
        	pt[1] = _pt[0];
        }
        else{
         	pt[0] = _pt[2];
         	pt[1] = _pt[3];
        }
        for(int j=i+1;j<size;j++){
            L2 = RectResults[j].center;
            if(L1.x != L2.x){
                k = GetK(L1,L2);
                if(L1.y > L2.y){
                    ydis = L1.y - L2.y;
                }
                else{
                    ydis = L2.y - L1.y;
                }
                areaL2 = RectResults[j].size.height * RectResults[j].size.width;
                if(areaL1 > areaL2){
                    divscale = areaL1 / areaL2;
                }
                else{
                    divscale = areaL2 / areaL1;
                }
                angleL2 = fabs(RectResults[j].angle);

                RectResults[j].points(_pt);
                if(angleL2 > 45.0){
                	pt[2] = _pt[2];
                	pt[3] = _pt[1];
                }
                else{
                	pt[2] = _pt[1];
                	pt[3] = _pt[0];
                }
                maxangle = MAX(ptangle(pt[0],pt[2]),ptangle(pt[1],pt[3]));
                if(angleL1 > 45.0 && angleL2 < 45.0){
                    angleabs = 90.0 - angleL1 + angleL2;
                }
                else if(angleL1 <= 45.0 && angleL2 >= 45.0){
                    angleabs = 90.0 - angleL2 + angleL1;
                }
                else{
                    if
                    	(angleL1 > angleL2) angleabs = angleL1 - angleL2;
                    else 
                    	angleabs = angleL2 - angleL1;
                }
                xdis = fabs(L1.x - L2.x);
                heightmax =MAX(MAX(RectResults[i].size.width,RectResults[j].size.width),MAX(RectResults[i].size.height,RectResults[j].size.height));
                hwdiv = xdis/heightmax;
                #if DEBUG
                /*
                	cout << "K = " << k << endl; 
                	cout << "divscale = " << divscale << endl; 
                	cout << "maxangle = " << maxangle << endl; 
                	cout << "hwdiv = " << hwdiv << endl; 
                	cout << "ydis = " << ydis << endl; 
                	cout << "angleabs = " << angleabs << endl; 
                */
                #endif
                if(fabs(k) < K && divscale < DIVSCALE && maxangle < MAXANGLE && hwdiv < HWDIV && ydis < YDISRATIO*heightmax){
                    if(angleabs < ANGLEABS){
                    	#if DEBUG       	
				        	cout << "K = " << k << endl; 
				        	cout << "divscale = " << divscale << endl; 
				        	cout << "maxangle = " << maxangle << endl; 
				        	cout << "hwdiv = " << hwdiv << endl; 
				        	cout << "ydis = " << ydis << endl; 
				        	cout << "angleabs = " << angleabs << endl; 	        
				        #endif
                        float armor_area = GetAreaofp3(pt[0],pt[1],pt[2]) + GetAreaofp3(pt[1],pt[2],pt[3]);
                        pair<vector<Point2f>,float> pushdata;
                        
                        for (int i = 0; i < 4; i ++){
                       		pushdata.first.push_back (pt[i]);
                       	}         
                        pushdata.second = armor_area;
                        roi.push_back(pushdata);
                        ArmorLostDelay = 0;
                    }
                }          
            }
        }
    }
    
    if(roi.size()==0){
        ArmorLostDelay++;
        if(ArmorLostDelay < 10){
            roi = roiOld;
        }
    }
}


double ArmorDetect::GetK(Point2f L1,Point2f L2){
    return (L1.y - L2.y) / (L1.x - L2.x);
}


void ArmorDetect::Clear(void){
	roi.clear();
	contours.clear();
	RectfirstResult.clear();
	Armorlists.clear();
	RectResults.clear();
	CellMaxs.clear();
}

