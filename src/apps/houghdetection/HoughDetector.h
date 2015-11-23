/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#ifndef HOUGHDETECTOR_H_
#define HOUGHDETECTOR_H_


#include "opencv2/opencv.hpp"
#include <vector>

#include "AppContextJointClassRegr.h"
#include "SampleImgPatch.h"
#include "LabelJointClassRegr.h"
#include "SplitFunctionImgPatch.h"
#include "SplitEvaluatorJointClassRegr.h"
#include "LeafNodeStatisticsJointClassRegr.h"
#include "DataLoaderHoughObject.h"

#include "icgrf.h"


// typedefs for easier use later
typedef SplitFunctionImgPatch<uchar, float, AppContextJointClassRegr> TSplitFunctionImgPatch;


struct Vote
{
	Vote() { }
	Vote(Eigen::VectorXd offsetin, double win)
	{
		x = offsetin(0);
		y = offsetin(1);
		w = win;
	}
	
	int x;
	int y;
	double w;
};


class HoughDetector
{
public:

	// Constructor
    HoughDetector(RandomForest<SampleImgPatch, LabelJointClassRegr, TSplitFunctionImgPatch, SplitEvaluatorJointClassRegr<SampleImgPatch, AppContextJointClassRegr>, LeafNodeStatisticsJointClassRegr<AppContextJointClassRegr>, AppContextJointClassRegr>* rfin, AppContextJointClassRegr* apphpin);

    // detection
    void DetectList();
<<<<<<< HEAD
	void DetectPyramid(const cv::Mat img, std::vector<std::vector<cv::Mat> >& hough_maps, std::vector<std::vector<cv::MatND> >& hough_pose_maps, std::vector<std::vector<std::vector<std::vector<Vote> > > >& vBackprojections, cv::Rect ROI = cv::Rect(-1,-1,-1,-1), int index = -1);
=======
    void DetectPyramid(const cv::Mat img, std::vector<std::vector<cv::Mat> >& hough_maps, std::vector<std::vector<cv::MatND> >& hough_pose_maps, std::vector<std::vector<std::vector<std::vector<Vote> > > >& vBackprojections, cv::Rect ROI = cv::Rect(-1,-1,-1,-1), int index = -1);
>>>>>>> a852804173915d67e48d70c4aee0b141323b9b7b

private:

    void ReadTestimageList(std::string path_testimages, std::vector<std::string>& filenames);
<<<<<<< HEAD
    std::vector<cv::Size> AverageBboxsizeFromTrainingdata(std::vector<VectorXd>& trainPose);
    void DetectImage(const cv::Mat img, std::vector<cv::Mat>& imgDetect, std::vector<cv::MatND>& imgDetectPose, std::vector<std::vector<std::vector<Vote> > >& backprojections, cv::Rect ROI = cv::Rect(-1,-1,-1,-1), int index = -1);
    void DetectBoundingBoxes(cv::Mat img, vector<vector<cv::Mat> >& hough_maps, std::vector<std::vector<cv::MatND> >& hough_pose_maps, std::vector<std::vector<std::vector<std::vector<Vote> > > >& vBackprojections, std::vector<cv::Size> avg_bbox_size, std::vector<VectorXd> trainPose, int img_id);
	void DetectImagePose(const cv::Mat img, cv::Mat& hough_map, int detected_z, int cx, int cy, cv::Rect ROI = cv::Rect(-1,-1,-1,-1));
=======
    std::vector<cv::Size> AverageBboxsizeFromTrainingdata(std::vector<VectorXd>& trainPose, int& NT);
    void DetectImage(const cv::Mat img, std::vector<cv::Mat>& imgDetect, std::vector<cv::MatND>& imgDetectPose, std::vector<std::vector<std::vector<Vote> > >& backprojections, cv::Rect ROI = cv::Rect(-1,-1,-1,-1), int index = -1);

    void DetectBoundingBoxes(cv::Mat img, int NT, vector<vector<cv::Mat> >& hough_maps, std::vector<std::vector<cv::MatND> >& hough_pose_maps, std::vector<std::vector<std::vector<std::vector<Vote> > > >& vBackprojections, std::vector<cv::Size> avg_bbox_size, std::vector<VectorXd> trainPose, int img_id);

    void DetectImagePose(const cv::Mat img, cv::Mat& hough_map, int detected_z, int cx, int cy, cv::Rect ROI = cv::Rect(-1,-1,-1,-1));
    
>>>>>>> a852804173915d67e48d70c4aee0b141323b9b7b
    RandomForest<SampleImgPatch, LabelJointClassRegr, TSplitFunctionImgPatch, SplitEvaluatorJointClassRegr<SampleImgPatch, AppContextJointClassRegr>, LeafNodeStatisticsJointClassRegr<AppContextJointClassRegr>, AppContextJointClassRegr>* m_rf;
    
    AppContextJointClassRegr* m_apphp;
    
    int m_pwidth;
    int m_pheight;
};

#endif /* HOUGHDETECTOR_H_ */
