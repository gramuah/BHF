/*
 * BoostedForest.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef BOOSTEDFOREST_CPP_
#define BOOSTEDFOREST_CPP_

#include "BoostedForest.h"



template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
BoostedForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::BoostedForest(RFCoreParameters* hpin, AppContext* appcontextin) :
	ADForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>(hpin, appcontextin),
	RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>(hpin, appcontextin)
{
	// should everything be done in the constructor of the RandomForest!
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
BoostedForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::~BoostedForest()
{
    // Free the trees ... done in base class
}


template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
void
BoostedForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::Train(DataSet<Sample, Label>& dataset)
{
	// Train the trees
	for (int t = 0; t < this->m_hp->m_num_trees; t++)
	{
		std::cout << "BT: train tree " << t+1 << " of " << this->m_hp->m_num_trees << std::endl;

		// init this tree
		this->m_trees.push_back(new RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>(this->m_hp, this->m_appcontext));
		this->m_trees[t]->Init(dataset);

		// calculate the prediction of the current state of the booster
		vector<LeafNodeStatistics> predictions = this->TestAndAverage(dataset);

		// update sample weights
		this->UpdateSampleTargetsClassification(dataset, predictions, this->m_hp->m_adf_loss_classification);

		// TODO: select the subset of samples that hold >90% of the weights!
		// for faster computation!
		// -> do we then also the make a tree refinement? with the out-of-bag examples???

		// train the trees
		for (unsigned int d = 0; d < this->m_hp->m_max_tree_depth; d++)
			this->m_trees[t]->Train(d);
	}
}




template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
std::vector<LeafNodeStatistics>
BoostedForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::TestAndAverage(DataSet<Sample, Label>& dataset)
{
	vector<vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > > leafnodes = this->Test(dataset);
	vector<LeafNodeStatistics> ret_stats(dataset.size(), this->m_appcontext);
	#pragma omp parallel for
	for (size_t s = 0; s < dataset.size(); s++)
	{
		vector<LeafNodeStatistics*> tmp_stats(leafnodes[s].size());
		for (size_t t = 0; t < leafnodes[s].size(); t++)
			tmp_stats[t] = leafnodes[s][t]->m_leafstats;

		ret_stats[s] = LeafNodeStatistics::Sum(tmp_stats, this->m_appcontext);
		//ret_stats[s] = LeafNodeStatistics::Average(tmp_stats, this->m_appcontext);
	}
	return ret_stats;
}

template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
LeafNodeStatistics
BoostedForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>::TestAndAverage(LabelledSample<Sample, Label>* sample)
{
	vector<Node<Sample, Label, SplitFunction, LeafNodeStatistics, AppContext>* > leafnodes;
	this->Test(sample, leafnodes);

	std::vector<LeafNodeStatistics*> tmp_stats(leafnodes.size());
	for (size_t t = 0; t < leafnodes.size(); t++)
		tmp_stats[t] = leafnodes[t]->m_leafstats;
	return LeafNodeStatistics::Sum(tmp_stats, this->m_appcontext);
	//return LeafNodeStatistics::Average(tmp_stats, this->m_appcontext);
}







#endif /* BOOSTEDFOREST_CPP_ */
