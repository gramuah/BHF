/*
 * JointClassRegrTree.h
 *
 *  Created on: 31.05.2013
 *      Author: samuel
 */

#ifndef JOINTCLASSREGRTREE_H_
#define JOINTCLASSREGRTREE_H_


#include "RandomTree.h"

#include "SampleImgPatch.h"
#include "LabelJointClassRegr.h"
#include "SplitFunctionImgPatch.h"
#include "SplitEvaluatorJointClassRegr.h"
#include "LeafNodeStatisticsJointClassRegr.h"
#include "AppContext.h"

template<typename ImgBaseDataType, typename ImgBaseDataIntType, typename LeafNodeStatistics, typename TAppContext>
class JointClassRegrTree :
	public RandomTree<SampleImgPatch, LabelJointClassRegr, SplitFunctionImgPatch<ImgBaseDataType, ImgBaseDataIntType, TAppContext>, SplitEvaluatorJointClassRegr<SampleImgPatch, TAppContext>, LeafNodeStatistics, TAppContext>
{
public:

	// Constructors & Destructors
    JointClassRegrTree(RFCoreParameters* hpin, TAppContext* appcontextin);
    virtual ~JointClassRegrTree();


protected:
    void GetRandomSampleSubsetForSplitting(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset_full, DataSet<SampleImgPatch, LabelJointClassRegr>& dataset_subsample, int samples_per_class);
};


#include "JointClassRegrTree.cpp"


#endif /* JOINTCLASSREGRTREE_H_ */
