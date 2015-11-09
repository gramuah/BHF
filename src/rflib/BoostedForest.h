/*
 * BoostedForest.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 */

#ifndef BOOSTEDFOREST_H_
#define BOOSTEDFOREST_H_

#include <vector>
#include <fstream>
#include "omp.h"
#include <eigen3/Eigen/Core>

#include "ADForest.h"
#include "RandomTree.h"
#include "DataSet.h"
#include "RFCoreParameters.h"



template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
class BoostedForest : virtual public ADForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>
{
public:
    BoostedForest(RFCoreParameters* hpin, AppContext* appcontextin);
    virtual ~BoostedForest();

    void Train(DataSet<Sample, Label>& dataset);
    vector<LeafNodeStatistics> TestAndAverage(DataSet<Sample, Label>& dataset);
    LeafNodeStatistics TestAndAverage(LabelledSample<Sample, Label>* sample);

    // not necessary as long as we don't have learned weights per tree
    //virtual void Save(std::string savepath, int t_offset = 0);
    //virtual void Load(std::string loadpath, int t_offset = 0);

protected:

    // parameters
	//RFCoreParameters* m_hp;
	//AppContext* m_appcontext;

	//vector<RandomTree<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>* > m_trees;
	//vector<double> m_tree_weights; this is not necessary if only use a shrinkage factor!
};



#include "BoostedForest.cpp"

#endif /* BOOSTEDFOREST_H_ */

