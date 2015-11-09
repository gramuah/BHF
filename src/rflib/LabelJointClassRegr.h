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
	// TODO: maybe we should force users of the lib to use the second constructur? otherwise weights are undefined!
	//LabelJointClassRegr() {}
	// TODO: do we really need the regr_weight???? I don't think so!

	/**
	 * Constructor for LabelJointClassRegr. There is now default constructor, only this one should be used,
	 * just to make sure that all member variables are initialized properly.
	 * Classification and regression information has to be given. However, some classes might not have any
	 * regression information (e.g., background class in HoughForests). This can be handled with the parameter
	 * vote_allowed which specifies if the sample associated with this label is allowed to vote for the class.
	 *
	 * @param[in] in_class_weight weight of the sample for the classification objective
	 * @param[in] in_class_label class label of the sample
	 * @param[in] in_regr_weight weight of the sample for the regression objective [NECESSARY????]
	 * @param[in] in_target regression target of the sample
	 * @param[in] in_vote_allowed specifies if the regression information is valid, i.e., if the sample is allowed
	 * to vote or not
	 */
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
	double class_weight; // working weight!
	int gt_class_label;
	int class_label; // working class label!
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
	double regr_weight; // working weight!
	Eigen::VectorXd regr_target_gt;
	Eigen::VectorXd regr_target; // working regression target
	Eigen::VectorXd regr_offset; // working regression target
	bool vote_allowed;
	std::vector<cv::Mat> hough_map_patch;


	void Save(std::ofstream& out);
	void Load(std::ifstream& in);
};

#endif /* LABELJOINTCLASSREGR_H_ */
