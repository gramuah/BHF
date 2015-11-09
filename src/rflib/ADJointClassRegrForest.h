/*
 * ADJointClassRegrForest.h
 *
 *  Created on: 31.05.2013
 *      Author: samuel
 */

#ifndef ADJOINTCLASSREGRFOREST_H_
#define ADJOINTCLASSREGRFOREST_H_

#include <vector>
#include <fstream>
#include "omp.h"

#include "ADForest.h"
#include "ARForest.h"

#include "JointClassRegrTree.h"

#include "SampleImgPatch.h"
#include "LabelJointClassRegr.h"
#include "SplitFunctionImgPatch.h"
#include "SplitEvaluatorJointClassRegr.h"
#include "LeafNodeStatisticsJointClassRegr.h"
#include "AppContext.h"




template<typename TAppContext>
class ADJointClassRegrForest :
	public ADForest<SampleImgPatch, LabelJointClassRegr, SplitFunctionImgPatch<uchar, float, TAppContext>, SplitEvaluatorJointClassRegr<SampleImgPatch, TAppContext>, LeafNodeStatisticsJointClassRegr<TAppContext>, TAppContext>,
	public ARForest<SampleImgPatch, LabelJointClassRegr, SplitFunctionImgPatch<uchar, float, TAppContext>, SplitEvaluatorJointClassRegr<SampleImgPatch, TAppContext>, LeafNodeStatisticsJointClassRegr<TAppContext>, TAppContext>
{
public:
    ADJointClassRegrForest(RFCoreParameters* hpin, TAppContext* appcontextin);
    virtual ~ADJointClassRegrForest();

    // override to have intermediate results
    virtual void Train(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, Eigen::MatrixXd& latent_variables, Eigen::VectorXd mean, Eigen::VectorXd std);

protected:
    double EvaluateClassification(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, std::vector<LeafNodeStatisticsJointClassRegr<TAppContext> > predictions);
    double EvaluateRegression(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, std::vector<LeafNodeStatisticsJointClassRegr<TAppContext> > predictions, Eigen::VectorXd mean, Eigen::VectorXd std, int d);
};


#include "ADJointClassRegrForest.cpp"


#endif /* JOINTCLASSREGRFOREST_H_ */

