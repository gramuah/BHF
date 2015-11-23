/*
 * DataSetLoaderHoughObject.h
 * 
 * Author: Carolina Redondo Cabrera, Roberto Javier López-Sastre, Alejandro Véliz Fernández
 * Institution: GRAM, University of Alcalá, Spain
 * 
 */

#ifndef DATASETLOADERHOUGHOBJECT_H_
#define DATASETLOADERHOUGHOBJECT_H_

#include <vector>
#include <eigen3/Eigen/Core>
#include "opencv2/opencv.hpp"

#include "icgrf.h"

#include "AppContextJointClassRegr.h"
#include "SampleImgPatch.h"
#include "LabelJointClassRegr.h"
#include "FeatureGeneratorRGBImage.h"

using namespace std;
using namespace Eigen;


class DataLoaderHoughObject
{
public:

    // Constructors & Destructors
    explicit DataLoaderHoughObject(AppContextJointClassRegr* hp);
    ~DataLoaderHoughObject();

    // Loading dataset (for classification and regression tasks)
    DataSet<SampleImgPatch, LabelJointClassRegr> LoadTrainData();
    void GetTrainDataProperties(int& num_samples, int& num_classes, int& num_target_variables, int& num_feature_channels, int& num_z);

    // Normalize & Denormalize the data
    void NormalizeRegressionTargets(std::vector<VectorXd>& mean, std::vector<VectorXd>& std);
    void DenormalizeRegressionTargets(std::vector<VectorXd> mean, std::vector<VectorXd> std);

    // static image & feature methods
    static cv::Mat ReadImageData(std::string imgpath);
    static void ExtractFeatureChannelsObjectDetection(const cv::Mat& img, std::vector<cv::Mat>& vImg, AppContextJointClassRegr* appcontext);

    // stored image data
	static std::vector<std::vector<cv::Mat> > image_feature_data;

private:

    void LoadPosTrainFile(vector<string>& vFilenames, vector<cv::Rect>& vBBox, vector<VectorXd>& vTarget, vector<vector<cv::Point> >& vSegmasks, vector<int>& z_targets, vector<double>& pose_targets, vector<double>& ze_targets, int& num_target_dims, int& m_num_z);
	void LoadNegTrainFile(vector<string>& vFilenames, vector<cv::Rect>& vBBox);

	Eigen::MatrixXd GeneratePatchPositionsFromRegion(const cv::Mat& img, const std::vector<cv::Mat>& img_features, int num_patches, const cv::Rect* include = 0, const cv::Rect* exclude = 0);
	Eigen::MatrixXd GeneratePatchPositionsFromMask(const cv::Mat& img, const std::vector<cv::Mat>& img_features, int num_patches, vector<cv::Point> segmask_points);
	void SavePatchPositions(std::string savepath, std::vector<Eigen::MatrixXd> patch_positions);
	std::vector<Eigen::MatrixXd> LoadPatchPositions(std::string loadpath);

	void ExtractPatches(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, const cv::Mat& img, const std::vector<cv::Mat>& img_features, Eigen::MatrixXd patch_locations, int img_index, int label, int z, double azimuth, double ze, VectorXd vTarget = VectorXd(), VectorXd vSize = VectorXd());
	void UpdateSampleWeights(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset);

    // parameters
    AppContextJointClassRegr* m_hp;

    // a copy of the data set is also stored here. The dataset only contains pointers to the real data,
    // which is only stored once!
    DataSet<SampleImgPatch, LabelJointClassRegr> m_trainset;

    // data properties
    int m_num_samples;
    int m_num_classes;
    int m_num_target_variables;
    int m_num_z;
    int m_num_feature_channels;
    int m_num_pos;
    int m_num_neg;
};

#endif /* DATASETLOADERHOUGHOBJECT_H_ */
