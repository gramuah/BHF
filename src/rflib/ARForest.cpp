/*
 * ARForest.cpp
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 */

#ifndef ARFOREST_CPP_
#define ARFOREST_CPP_

#include "ARForest.h"



template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
ARForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::ARForest(RFCoreParameters* hpin, AppContext* appcontextin) :
	RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>(hpin, appcontextin)
{
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
ARForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::~ARForest()
{
    // Base destructor deletes all trees ...
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
ARForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Train(DataSet<Sample, Label>& dataset,  Eigen::MatrixXd& latent_variables, Eigen::VectorXd mean, Eigen::VectorXd std)
{
	// This is the alternating decision forest training procedure ...
    vector<DataSet<Sample, Label> > inbag_dataset(this->m_hp->m_num_trees), outbag_dataset(this->m_hp->m_num_trees);
	this->BaggingForTrees(dataset, inbag_dataset, outbag_dataset);

	// Init the trees
	this->m_trees.resize(this->m_hp->m_num_trees);
	for (unsigned int t = 0; t < this->m_trees.size(); t++)
	{
		this->m_trees[t] = new RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>(this->m_hp, this->m_appcontext);
		// Define tree as ADF-Tree and set the prediction type!
		this->m_trees[t]->m_is_ADFTree = true;
		this->m_trees[t]->m_prediction_type_ADF = 1; // 1 = regression -> in order to add parent predictions to children
		this->m_trees[t]->Init(inbag_dataset[t]);
	}

	// start the iterations
	for (unsigned int d = 0; d < this->m_hp->m_max_tree_depth; d++)
	{
		// Classify all samples with the current forest
		vector<LeafNodeStatistics> forest_predictions = this->TestAndAverage(dataset, d, mean, std);

		// calculate the pseudo-targets
		this->UpdateSampleTargetsRegression(dataset, forest_predictions, latent_variables, mean, std, this->m_hp->m_adf_loss_regression);

		// Status message
		int num_total_nodes_to_split = 0;
		for (int t = 0; t < this->m_trees.size(); t++)
			num_total_nodes_to_split += this->m_trees[t]->GetTrainingQueueSize();
		if (!this->m_hp->m_quiet)
			std::cout << "ARF: training depth " << d << " of the forest -> " << num_total_nodes_to_split << " nodes left for splitting" << std::endl;

		if (num_total_nodes_to_split == 0)
		{
			if (!this->m_hp->m_quiet)
				std::cout << "No nodes left for splitting ... stop growing trees" << std::endl;
			break;
		}

		// Train the trees
		#pragma omp parallel for
		for (int t = 0; t < this->m_hp->m_num_trees; t++)
			this->m_trees[t]->Train(d);
	}

	// tree refinement
	if (this->m_hp->m_do_tree_refinement)
	{
		// CAUTION: here, we have to first reset the targets to targets_gt !!!
		for (size_t t = 0; t < outbag_dataset.size(); t++)
			for (size_t s = 0; s < outbag_dataset[t].size(); s++)
				outbag_dataset[t][s]->m_label.regr_target = outbag_dataset[t][s]->m_label.regr_target_gt;

		#pragma omp parallel for
		for (size_t t = 0; t < this->m_trees.size(); t++)
			this->m_trees[t]->UpdateLeafStatistics(outbag_dataset[t]);
	}
}

template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
ARForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::UpdateSampleTargetsRegression(DataSet<Sample, Label> dataset, vector<LeafNodeStatistics> forest_predictions, Eigen::MatrixXd& latent_variables, Eigen::VectorXd mean, Eigen::VectorXd std, ADF_LOSS_REGRESSION::Enum wut)
{
	for (size_t s = 0; s < dataset.size(); s++)
	{
		// INFO: in the case of joint classification-regression, we also compute the residuals for
		// samples that are not allowed to vote! This has to be considered for evaluating the splitting
		// functions, etc.

		struct residual sample_residual = forest_predictions[s].CalculateADFTargetResidual(dataset[s]->m_label, forest_predictions[s].m_hough_img_prediction, mean, std, s, 1);
		
		dataset[s]->m_label.latent_prediction = sample_residual.bestz;//latent_residual;
		if (dataset[s]->m_label.vote_allowed == true){

			latent_variables(dataset[s]->m_label.patch_id, 0) = dataset[s]->m_label.patch_id;
			latent_variables(dataset[s]->m_label.patch_id, 1) = sample_residual.bestz;
		}

		// calcualte the pseudo targets
		dataset[s]->m_label.regr_target = Eigen::VectorXd::Zero(sample_residual.ret_vec.size());
		double norm = 0.0;
		switch (this->m_hp->m_adf_loss_regression)
		{
		case ADF_LOSS_REGRESSION::SQUARED_LOSS:
			for (size_t v = 0; v < sample_residual.ret_vec.size(); v++)
			{
				dataset[s]->m_label.regr_target(v) = -1.0 * sample_residual.ret_vec[v];
			}
			break;
		case ADF_LOSS_REGRESSION::ABSOLUTE_LOSS:
			// calculate the norm of the residual
			norm = 0.0;
			for (size_t v = 0; v < sample_residual.ret_vec.size(); v++)
				norm += sample_residual.ret_vec[v] * sample_residual.ret_vec[v];
			norm = sqrt(norm);
			// calculate the pseudo targets
			for (size_t v = 0; v < sample_residual.ret_vec.size(); v++)
			{
				if (sample_residual.ret_vec[v] < 0.0)
					dataset[s]->m_label.regr_target(v) = +1.0 * norm;
				else if (sample_residual.ret_vec[v] > 0.0)
					dataset[s]->m_label.regr_target(v) = -1.0 * norm;
				else
					dataset[s]->m_label.regr_target(v) = 0.0;
			}
			break;
		case ADF_LOSS_REGRESSION::HUBER_LOSS:
			// calculate the norm of the residual
			norm = 0.0;
			for (size_t v = 0; v < sample_residual.ret_vec.size(); v++)
				norm += sample_residual.ret_vec[v] * sample_residual.ret_vec[v];
			norm = sqrt(norm);
			for (size_t v = 0; v < sample_residual.ret_vec.size(); v++)
			{
				if (sample_residual.ret_vec[v] < -this->m_hp->m_Huberloss_delta)
					dataset[s]->m_label.regr_target(v) = +1.0 * norm;
				else if (sample_residual.ret_vec[v] > this->m_hp->m_Huberloss_delta)
					dataset[s]->m_label.regr_target(v) = -1.0 * norm;
				else
					dataset[s]->m_label.regr_target(v) = -1.0 * sample_residual.ret_vec[v];
			}
			break;
		default:
			throw std::runtime_error("ARForest: Computing pseudo targets doesn't know the loss function");
		}
	}
}




#endif /* ARFOREST_CPP_ */
