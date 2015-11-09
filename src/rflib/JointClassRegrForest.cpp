/*
 * JointClassRegrForest.cpp
 *
 *  Created on: 31.05.2013
 *      Author: samuel
 */

#ifndef JOINTCLASSREGRFOREST_CPP_
#define JOINTCLASSREGRFOREST_CPP_

#include "JointClassRegrForest.h"


template<typename TAppContext>
JointClassRegrForest<TAppContext>::JointClassRegrForest(RFCoreParameters* hpin, TAppContext* appcontextin) :
	RandomForest<SampleImgPatch, LabelJointClassRegr, SplitFunctionImgPatch<uchar, float, TAppContext>, SplitEvaluatorJointClassRegr<SampleImgPatch, TAppContext>, LeafNodeStatisticsJointClassRegr<TAppContext>, TAppContext>(hpin, appcontextin)
{
}

template<typename TAppContext>
JointClassRegrForest<TAppContext>::~JointClassRegrForest()
{
    // trees are deleted in base case
}

template<typename TAppContext>
void JointClassRegrForest<TAppContext>::Train(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, Eigen::VectorXd mean, Eigen::VectorXd std )
{
	// This is the standard random forest training procedure ...
    vector<DataSet<SampleImgPatch, LabelJointClassRegr> > inbag_dataset(this->m_hp->m_num_trees), outbag_dataset(this->m_hp->m_num_trees);
	this->BaggingForTrees(dataset, inbag_dataset, outbag_dataset);

	// Train the trees
	this->m_trees.resize(this->m_hp->m_num_trees);
	for (int t = 0; t < this->m_hp->m_num_trees; t++)
	{
		this->m_trees[t] = new JointClassRegrTree<uchar, float, LeafNodeStatisticsJointClassRegr<TAppContext>, TAppContext>(this->m_hp, this->m_appcontext);
		this->m_trees[t]->Init(inbag_dataset[t]);
	}

	for (unsigned int d = 0; d < this->m_hp->m_max_tree_depth; d++)
	{
		std::vector<LeafNodeStatisticsJointClassRegr<TAppContext> > predictions = this->TestAndAverage(dataset, d, mean, std);
		double current_accuracy = this->EvaluateClassification(dataset, predictions);

		int num_nodes_left = 0;
		for (int t = 0; t < this->m_hp->m_num_trees; t++)
			num_nodes_left += this->m_trees[t]->GetTrainingQueueSize();
		if (!this->m_hp->m_quiet)
		{
			std::cout << "HF: training depth " << d << " of the forest ";
			std::cout << "-> " << num_nodes_left << " nodes left for splitting ";
			std::cout << "-> current acc = " << current_accuracy << std::endl;
		}

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
double JointClassRegrForest<TAppContext>::EvaluateClassification(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, std::vector<LeafNodeStatisticsJointClassRegr<TAppContext> > predictions)
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


#endif /* JOINTCLASSREGRFOREST_CPP_ */
