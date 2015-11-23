/*
 * LabelJointClassRegr
 */

#ifndef LABELJOINTCLASSREGR_H_
#define LABELJOINTCLASSREGR_H_

#include <stdexcept>
#include <vector>
#include "opencv2/opencv.hpp"
#include "eigen3/Eigen/Core"


struct LabelJointClassRegr
{
	LabelJointClassRegr(double in_class_weight, int in_class_label, int z, double azimuth, double zenith, Eigen::VectorXd vCenter, Eigen::VectorXd vCenterPatch, Eigen::VectorXd imgSize,int img_index, double in_regr_weight, Eigen::VectorXd in_target, bool in_vote_allowed)
	{
		this->class_weight = in_class_weight;
		this->class_weight_gt = in_class_weight;
		this->gt_class_label = in_class_label;
		this->class_label = in_class_label;
		this->latent_label = z;
		this->latent_prediction = z;
		this->img_id = img_index;
		this->regr_center_gt = vCenter;
		this->regr_patch_center_gt = vCenterPatch;
		this->img_size = imgSize;
		this->regr_weight = in_regr_weight;
		this->regr_weight_gt = in_regr_weight;
		this->regr_target = in_target;
		this->regr_offset = in_target;
		this->regr_target_gt = in_target;
		this->azimuth = azimuth;
		this->zenith = zenith;
		this->vote_allowed = in_vote_allowed;
	}

	// weights, labels and targets
	double class_weight_gt;
	double class_weight;
	int gt_class_label;
	int class_label;
	int latent_label;
	int latent_prediction;
	int patch_id;
	int img_id;
	double azimuth;
	double zenith;
	Eigen::VectorXd regr_center_gt;
	Eigen::VectorXd img_size;
	Eigen::VectorXd regr_patch_center_gt;
	Eigen::VectorXd regr_patch_center_norm_gt;
	double regr_weight_gt;
	double regr_weight;
	Eigen::VectorXd regr_target_gt;
	Eigen::VectorXd regr_target;
	Eigen::VectorXd regr_offset;
	bool vote_allowed;
	std::vector<cv::Mat> hough_map_patch;


	void Save(std::ofstream& out);
	void Load(std::ifstream& in);
};

#endif /* LABELJOINTCLASSREGR_H_ */
