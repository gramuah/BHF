/*
 * ADForest.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 */

#ifndef ADFOREST_H_
#define ADFOREST_H_

#include "RandomForest.h"




template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
class ADForest : virtual public RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>
{
public:
	explicit ADForest(RFCoreParameters* hpin, AppContext* appcontextin);
	virtual ~ADForest();
	virtual void Train(DataSet<Sample, Label>& dataset, Eigen::VectorXd mean, Eigen::VectorXd std);

protected:

	virtual void UpdateSampleTargetsClassification(DataSet<Sample, Label>& dataset, vector<LeafNodeStatistics>& forest_predictions, ADF_LOSS_CLASSIFICATION::Enum wut);

	// matrix storing the weights of each sample and for each depth (level of the forest)
	MatrixXd m_sample_weight_progress;
};



#include "ADForest.cpp"

#endif /* ADFOREST_H_ */
