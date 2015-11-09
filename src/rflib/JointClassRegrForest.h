/*
 * JointClassRegrForest.h
 *
 *  Created on: 31.05.2013
 *      Author: samuel
 */

#ifndef JOINTCLASSREGRFOREST_H_
#define JOINTCLASSREGRFOREST_H_

#include <vector>
#include <fstream>
#include "omp.h"

#include "RandomForest.h"
#include "JointClassRegrTree.h"

//typedef SplitFunctionImgPatch<uchar, float, TAppContext> TSplitFunctionImgPatch;

template<typename TAppContext>
class JointClassRegrForest : public RandomForest<SampleImgPatch, LabelJointClassRegr, SplitFunctionImgPatch<uchar, float, TAppContext>, SplitEvaluatorJointClassRegr<SampleImgPatch, TAppContext>, LeafNodeStatisticsJointClassRegr<TAppContext>, TAppContext>
{
public:
    JointClassRegrForest(RFCoreParameters* hpin, TAppContext* appcontextin);
    virtual ~JointClassRegrForest();

    // override to have intermediate results
    virtual void Train(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, Eigen::VectorXd mean, Eigen::VectorXd std);

protected:
    double EvaluateClassification(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, std::vector<LeafNodeStatisticsJointClassRegr<TAppContext> > predictions);
};


#include "JointClassRegrForest.cpp"


#endif /* JOINTCLASSREGRFOREST_H_ */

