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
};



#include "BoostedForest.cpp"

#endif /* BOOSTEDFOREST_H_ */

