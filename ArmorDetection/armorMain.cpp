#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include "armorDetect.h"
#define DEBUG 1
#define VIDEO 1
using namespace cv;
using namespace std;

int main (int argvc, char ** argv){
#if VIDEO
	VideoCapture cap(1);
	if ( !cap.isOpened() )
    	{
        	cout << "Cannot open the web cam" << endl;
         	return -1;
    	}
	vector<pair<vector<Point2f>, float>> ROI;
	Mat inputFrame;
	ArmorDetect detector = ArmorDetect();
	while (true){
		double t =(double)getTickCount();
		bool fSuccess = cap.read(inputFrame);
		
		if (!fSuccess){
			cout << "Fail to read frame" << endl;
		}
		detector.process( inputFrame, ROI);
		#if DEBUG
			if (ROI.size() != 0)
			{
				for (int i = 0; i < 1; i++){
					line( inputFrame, ROI[i].first[0], ROI[i].first[1], Scalar (0, 255, 255), 3, 8 );
					line( inputFrame, ROI[i].first[1], ROI[i].first[3], Scalar (0, 255, 255), 3, 8 );
					line( inputFrame, ROI[i].first[3], ROI[i].first[2], Scalar (0, 255, 255), 3, 8 );
					line( inputFrame, ROI[i].first[2], ROI[i].first[0], Scalar (0, 255, 255), 3, 8 );
				}
			}
			
			namedWindow ("debug4", WINDOW_NORMAL);
	    		imshow ("debug4", inputFrame);
		#endif
		//waitKey(0);
		
		if (waitKey(30) == 27) 
	       	{
		    	cout << "esc key is pressed by user" << endl;
		    	break; 
	       	}
	       	double fps = 1/(((double)getTickCount() - t)/getTickFrequency());
	       	cout << "fps: " << fps << endl;
	       	
	}
#else
	Mat inputImage;
	vector<pair<vector<Point2f>, float>> ROI;
	inputImage = imread (argv[1], IMREAD_COLOR);
	ArmorDetect detector = ArmorDetect();
	detector.process ( inputImage, ROI );
	#if DEBUG
		for (int i = 0; i < ROI.size(); i++){
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
