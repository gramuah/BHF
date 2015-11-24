/*
 * ADJointClassRegrForest.cpp
 *
 *  Created on: 31.05.2013
 *      Author: samuel
 */

#ifndef ADJOINTCLASSREGRFOREST_CPP_
#define ADJOINTCLASSREGRFOREST_CPP_

#include "ADJointClassRegrForest.h"


template<typename TAppContext>
ADJointClassRegrForest<TAppContext>::ADJointClassRegrForest(RFCoreParameters* hpin, TAppContext* appcontextin) :
	ADForest<SampleImgPatch, LabelJointClassRegr, SplitFunctionImgPatch<uchar, float, TAppContext>, SplitEvaluatorJointClassRegr<SampleImgPatch, TAppContext>, LeafNodeStatisticsJointClassRegr<TAppContext>, TAppContext>(hpin, appcontextin),
	ARForest<SampleImgPatch, LabelJointClassRegr, SplitFunctionImgPatch<uchar, float, TAppContext>, SplitEvaluatorJointClassRegr<SampleImgPatch, TAppContext>, LeafNodeStatisticsJointClassRegr<TAppContext>, TAppContext>(hpin, appcontextin),
	RandomForest<SampleImgPatch, LabelJointClassRegr, SplitFunctionImgPatch<uchar, float, TAppContext>, SplitEvaluatorJointClassRegr<SampleImgPatch, TAppContext>, LeafNodeStatisticsJointClassRegr<TAppContext>, TAppContext>(hpin, appcontextin)
{
}

template<typename TAppContext>
ADJointClassRegrForest<TAppContext>::~ADJointClassRegrForest()
{
    // trees are deleted in base classes
}


template<typename TAppContext>
void ADJointClassRegrForest<TAppContext>::Train(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, Eigen::MatrixXd& latent_variables, Eigen::VectorXd mean, Eigen::VectorXd std)
{
	// This is the standard random forest training procedure ...
    vector<DataSet<SampleImgPatch, LabelJointClassRegr> > inbag_dataset(this->m_hp->m_num_trees), outbag_dataset(this->m_hp->m_num_trees);
	this->BaggingForTrees(dataset, inbag_dataset, outbag_dataset);

	// Init the trees
	this->m_trees.resize(this->m_hp->m_num_trees);
	for (int t = 0; t < this->m_hp->m_num_trees; t++)
	{
		this->m_trees[t] = new JointClassRegrTree<uchar, float, LeafNodeStatisticsJointClassRegr<TAppContext>, TAppContext>(this->m_hp, this->m_appcontext);
		if (this->m_appcontext->do_classification_weight_updates || this->m_appcontext->do_regression_weight_updates)
			this->m_trees[t]->m_is_ADFTree = true;
		if (this->m_appcontext->do_regression_weight_updates)
			this->m_trees[t]->m_prediction_type_ADF = 1;
		this->m_trees[t]->Init(inbag_dataset[t]);
	}

	// Train the trees depth by depth
	for (unsigned int d = 0; d < this->m_hp->m_max_tree_depth; d++)
	{
		std::vector<LeafNodeStatisticsJointClassRegr<TAppContext> > predictions = this->TestAndAverage(dataset, d, mean, std);

		double current_accuracy = this->EvaluateClassification(dataset, predictions);
		double current_rmse = this->EvaluateRegression(dataset, predictions, mean, std, d);

		int num_nodes_left = 0;
		for (int t = 0; t < this->m_hp->m_num_trees; t++)
			num_nodes_left += this->m_trees[t]->GetTrainingQueueSize();
		if (!this->m_hp->m_quiet)
		{
			std::cout << "ADRHF: training depth " << d << " ";
			std::cout << "-> " << num_nodes_left << " nodes left ";
			std::cout << "-> cur.acc. = " << current_accuracy << ", ";
			std::cout << "cur.RMSE = " << current_rmse << std::endl;
		}

		// update the sample weights
		if (this->m_appcontext->do_classification_weight_updates)
			this->UpdateSampleTargetsClassification(dataset, predictions, this->m_appcontext->global_loss_classification);
		// calculate the pseudo targets!
		if (this->m_appcontext->do_regression_weight_updates)
			this->UpdateSampleTargetsRegression(dataset, predictions, latent_variables, mean, std, this->m_appcontext->global_loss_regression);

		// train the trees
		#pragma omp parallel for
		for (int t = 0; t < this->m_hp->m_num_trees; t++)
			this->m_trees[t]->Train(d);
	}
	if (this->m_hp->m_do_tree_refinement)
	{
		#pragma omp parallel for
		for (int t = 0; t < this->m_hp->m_num_trees; t++)
			this->m_trees[t]->UpdateLeafStatistics(outbag_dataset[t]);
	}
}


template<typename TAppContext>
double ADJointClassRegrForest<TAppContext>::EvaluateClassification(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, std::vector<LeafNodeStatisticsJointClassRegr<TAppContext> > predictions)
{
	int num_correct = 0;
	for (size_t s = 0; s < dataset.size(); s++)
	{
		int max_class_id = 0;
		double max_class_prob = predictions[s].m_class_histogram[0];
		for (size_t c = 1; c < predictions[s].m_class_histogram.size(); c++)
		{
			if (predictions[s].m_class_histogram[c] > max_class_prob)
			{
				max_class_prob = predictions[s].m_class_histogram[c];
				max_class_id = c;
			}
		}
		if (max_class_id == dataset[s]->m_label.class_label)
			num_correct++;
	}
	return (double)num_correct / (double)dataset.size();
}

template<typename TAppContext>
double ADJointClassRegrForest<TAppContext>::EvaluateRegression(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, std::vector<LeafNodeStatisticsJointClassRegr<TAppContext> > predictions, Eigen::VectorXd mean, Eigen::VectorXd std, int d)
{
	double squared_error = 0.0;
	int n_samples_evaluated = 0;
	
	for (size_t s = 0; s < dataset.size(); s++)
	{	
		if (!dataset[s]->m_label.vote_allowed)
			continue;

		Eigen::VectorXd gt_target = dataset[s]->m_label.regr_target_gt;
		Eigen::VectorXd gt_target_c = Eigen::VectorXd::Zero(2);
		gt_target_c(0) = dataset[s]->m_label.regr_center_gt(0);
		gt_target_c(1) = dataset[s]->m_label.regr_center_gt(1);
		int gt_classlabel = dataset[s]->m_label.gt_class_label;

		n_samples_evaluated++;

		// prediction
		Eigen::VectorXd rf_prediction = predictions[s].m_votes[gt_classlabel][0];
		Eigen::VectorXd rf_prediction_c = Eigen::VectorXd::Zero(2);
		rf_prediction_c(0) = predictions[s].m_hough_img_prediction[1](dataset[s]->m_label.latent_prediction-1 ,0);
		rf_prediction_c(1) = predictions[s].m_hough_img_prediction[1](dataset[s]->m_label.latent_prediction-1 ,1);
		Eigen::VectorXd rf_prediction_c_norm = Eigen::VectorXd::Zero(2);
		Eigen::VectorXd rf_prediction_off = Eigen::VectorXd::Zero(2);	
		rf_prediction_c_norm = rf_prediction_c - dataset[s]->m_label.regr_patch_center_gt;	
		rf_prediction_c_norm(0) -= mean(0);
		rf_prediction_c_norm(0) /= std(0);
		rf_prediction_c_norm(1) -= mean(1);
		rf_prediction_c_norm(1) /= std(1);
		
		//calculate RMSE;
		double pred_diff = 1.0 / (double)rf_prediction_c_norm.rows() * (rf_prediction_c_norm - gt_target).dot(rf_prediction_c_norm - gt_target);
		squared_error += pred_diff;
	}
	double rmse_value = sqrt(squared_error / (double)n_samples_evaluated);

	return rmse_value;
}



#endif /* ADJOINTCLASSREGRFOREST_CPP_ */
