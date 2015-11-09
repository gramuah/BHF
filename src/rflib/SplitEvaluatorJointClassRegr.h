/*
 * SplitEvaluatorJointClassRegr.h
 *
 *  Created on: 18.12.2012
 *      Author: samuel
 */

#ifndef SPLITEVALUATORJOINTCLASSREGR_H_
#define SPLITEVALUATORJOINTCLASSREGR_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU> // determinant!!!
#include <vector>
#include <math.h> // M_PI
#include "opencv2/opencv.hpp"

#include "LabelJointClassRegr.h"
#include "AppContext.h"

#include "icgrf.h"

using namespace std;
using namespace Eigen;


template<typename Sample, typename TAppContext>
class SplitEvaluatorJointClassRegr
{
public:

	// Constructors & destructors
	SplitEvaluatorJointClassRegr(TAppContext* appcontextin, int depth, DataSet<Sample, LabelJointClassRegr>& dataset);
    virtual ~SplitEvaluatorJointClassRegr();

    bool DoFurtherSplitting(DataSet<Sample, LabelJointClassRegr>& dataset, int depth);
    bool CalculateScoreAndThreshold(DataSet<Sample, LabelJointClassRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold);

protected:

    // Classification stuff
	bool CalculateEntropyAndThreshold(DataSet<Sample, LabelJointClassRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold, int use_gini);

	// regression stuff
	bool CalculateOffsetCompactnessAndThreshold(DataSet<Sample, LabelJointClassRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold);
	bool CalculatePoseCompactnessAndThreshold(DataSet<Sample, LabelJointClassRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold);
	bool CalculateOffsetCompactnessAndThresholdOnline(DataSet<Sample, LabelJointClassRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold);
	double EvaluateRegressionLoss(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> LSamples, vector<int> RSamples, double& LTotal, double& RTotal, int flag_pose);
	double EvaluateRegressionLoss_ReductionInVariance(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> LSamples, vector<int> RSamples, double& LTotal, double& RTotal);
	double EvaluateRegressionLoss_DiffEntropyGauss(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> LSamples, vector<int> RSamples, double& LTotal, double& RTotal);
	double EvaluateRegressionLoss_DiffEntropyGaussPose(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> LSamples, vector<int> RSamples, double& LTotal, double& RTotal);
	double EvaluateRegressionLoss_DiffEntropyGaussBlockPoseEstimation(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> LSamples, vector<int> RSamples, double& LTotal, double& RTotal);

	VectorXd CalculateMean(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> sample_ids, bool weighted, double& sum_weight);
	MatrixXd CalculateCovariance(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> sample_ids, bool weighted, double& total_weight);
	VectorXd CalculateMeanPose(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> sample_ids, bool weighted, double& sum_weight);
	MatrixXd CalculateCovariancePose(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> sample_ids, bool weighted, double& total_weight);
	double CalculateVariance(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> sample_ids, bool weighted, double& total_weight);

	// members
	TAppContext* m_appcontext;

    int m_eval_type;
    int m_eval_regr_type;

};



#include "SplitEvaluatorJointClassRegr.cpp"


#endif /* SPLITEVALUATORJOINTCLASSREGR_H_ */
