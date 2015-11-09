/*
 * ADForest.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */


#ifndef ADFOREST_CPP_
#define ADFOREST_CPP_

#include "ADForest.h"


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
ADForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::ADForest(RFCoreParameters* hpin, AppContext* appcontextin) :
	RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>(hpin, appcontextin)
{
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
ADForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::~ADForest()
{
    // Base destructor deletes all trees ...
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
ADForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Train(DataSet<Sample, Label>& dataset, Eigen::VectorXd mean, Eigen::VectorXd std)
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
		this->m_trees[t]->m_prediction_type_ADF = 0; // 0 = classification
		this->m_trees[t]->Init(inbag_dataset[t]);
	}

	// init the sample weight matrix
	this->m_sample_weight_progress = MatrixXd::Zero(dataset.size(), this->m_hp->m_max_tree_depth);

	// start the iterations
	for (unsigned int d = 0; d < this->m_hp->m_max_tree_depth; d++)
	{
		// Classify all samples with the current forest
		vector<LeafNodeStatistics> forest_predictions = this->TestAndAverage(dataset, d, mean, std);

		// ADForest is always for classification -> thus, this call explicitely for classification is ok
		// see ARForest.h for the regression version
		this->UpdateSampleTargetsClassification(dataset, forest_predictions, this->m_hp->m_adf_loss_classification);

		// store the current weight of the samples
		// CAUTION: also this is specific for the ML-classification task!!! should rather be generic ?!
		for (size_t s = 0; s < dataset.size(); s++)
			this->m_sample_weight_progress(s, d) = dataset[s]->m_label.class_weight;

		// Train the current depth of the forest
		int num_total_nodes_to_split = 0;
		for (int t = 0; t < this->m_trees.size(); t++)
			num_total_nodes_to_split += this->m_trees[t]->GetTrainingQueueSize();
		if (!this->m_hp->m_quiet)
			std::cout << "ADF: training depth " << d << " of the forest -> " << num_total_nodes_to_split << " nodes left for splitting" << std::endl;

		if (num_total_nodes_to_split == 0)
		{
			if (!this->m_hp->m_quiet)
				std::cout << "No nodes left for splitting ... stop growing trees" << std::endl;
			break;
		}

		// Train the next stage
		#pragma omp parallel for
		for (int t = 0; t < this->m_hp->m_num_trees; t++)
			this->m_trees[t]->Train(d);
	}

	// tree refinement
	if (this->m_hp->m_do_tree_refinement)
	{
		for (int t = 0; t < (int)this->m_trees.size(); t++)
			this->m_trees[t]->UpdateLeafStatistics(outbag_dataset[t]);
	}

	// store the sample weight progress
	// CAUTION: also this is specific for the ML-classification task!!! (AppContext)
	std::ofstream out(this->m_appcontext->path_sampleweight_progress.c_str(), ios::binary);
	out << this->m_sample_weight_progress << std::endl;
	out.flush();
	out.close();
}




template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
ADForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::UpdateSampleTargetsClassification(DataSet<Sample, Label>& dataset, vector<LeafNodeStatistics>& forest_predictions, ADF_LOSS_CLASSIFICATION::Enum wut)
{
	double sum_weights = 0.0;
	double a_const = 3.1415/2.0;
	for (size_t s = 0; s < dataset.size(); s++)
	{
		std::vector<double> sample_conf_vec = forest_predictions[s].CalculateADFTargetResidual_class(dataset[s]->m_label, 0);
		double sample_conf = sample_conf_vec[0];

		// We can scale the margin at this point to exploit a larger area of the loss function!
		// TODO: evaluate this parameter!!!
		double margin_scale_factor = 3.0;
		sample_conf *= margin_scale_factor;

		double new_weight;
		switch (this->m_hp->m_adf_loss_classification)
		{
		case ADF_LOSS_CLASSIFICATION::GRAD_HINGE:
			new_weight = (sample_conf < 1.0) ? 1.0 : 0.0;
			break;
		case ADF_LOSS_CLASSIFICATION::GRAD_LOGIT:
			//new_weight = 1.0 / (1.0 / exp(-2.0 * sample_conf));
			new_weight = exp(-sample_conf) / (1.0 + exp(-sample_conf));
			break;
		case ADF_LOSS_CLASSIFICATION::GRAD_EXP:
			new_weight = exp(-sample_conf);
			break;
		case ADF_LOSS_CLASSIFICATION::GRAD_SAVAGE:
			new_weight = (4.0 * exp(2.0*sample_conf) / pow(1.0+exp(2.0*sample_conf), 3.0));
			break;
		case ADF_LOSS_CLASSIFICATION::GRAD_TANGENT:
			new_weight = abs(-(4.0/a_const)*(1.0/(1.0+pow(sample_conf, 2.0)))*((2.0/a_const)*atan(sample_conf)-1.0));
			break;
		default:
			throw std::runtime_error("ADForest: Update classificiation weights doesn't know the loss function");
		}

		double old_weight = dataset[s]->m_label.class_weight;
		dataset[s]->m_label.class_weight = this->m_hp->m_shrinkage * new_weight + (1.0 - this->m_hp->m_shrinkage) * old_weight;

		// respect the global initial weighting given for a sample (e.g., the class balance)
		dataset[s]->m_label.class_weight *= dataset[s]->m_label.class_weight_gt;

		sum_weights += dataset[s]->m_label.class_weight;
	}
	// normalize the weights
	for (size_t s = 0; s < dataset.size(); s++)
		dataset[s]->m_label.class_weight /= sum_weights;
}


#endif /* ADFOREST_CPP_ */
