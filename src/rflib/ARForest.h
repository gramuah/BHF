/*
 * ARForest.h
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 */

#ifndef ARFOREST_H_
#define ARFOREST_H_

#include "RandomForest.h"



template<typename Sample, typename Label, typename SplitFunction, typename SplitEvaluator, typename LeafNodeStatistics, typename AppContext>
class ARForest : virtual public RandomForest<Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics, AppContext>
{
public:
	explicit ARForest(RFCoreParameters* hpin, AppContext* appcontextin);
	virtual ~ARForest();
	void Train(DataSet<Sample, Label>& dataset, Eigen::MatrixXd& latent_variables, Eigen::VectorXd mean, Eigen::VectorXd std);

protected:

	void UpdateSampleTargetsRegression(DataSet<Sample, Label> dataset, vector<LeafNodeStatistics> forest_predictions, Eigen::MatrixXd& latent_variables, Eigen::VectorXd mean, Eigen::VectorXd std, ADF_LOSS_REGRESSION::Enum wut);
};



#include "ARForest.cpp"

#endif /* ARFOREST_H_ */
