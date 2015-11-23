#ifndef JOINTCLASSREGRTREE_CPP_
#define JOINTCLASSREGRTREE_CPP_

#include "JointClassRegrTree.h"



template<typename ImgBaseDataType, typename ImgBaseDataIntType, typename LeafNodeStatistics, typename TAppContext>
JointClassRegrTree<ImgBaseDataType, ImgBaseDataIntType, LeafNodeStatistics, TAppContext>::JointClassRegrTree(RFCoreParameters* hpin, TAppContext* appcontextin) :
	RandomTree<SampleImgPatch, LabelJointClassRegr, SplitFunctionImgPatch<ImgBaseDataType, ImgBaseDataIntType, TAppContext>, SplitEvaluatorJointClassRegr<SampleImgPatch, TAppContext>, LeafNodeStatistics, TAppContext>(hpin, appcontextin)
{
}


template<typename ImgBaseDataType, typename ImgBaseDataIntType, typename LeafNodeStatistics, typename TAppContext>
JointClassRegrTree<ImgBaseDataType, ImgBaseDataIntType, LeafNodeStatistics, TAppContext>::~JointClassRegrTree()
{
	// nodes are deleted in the base class
}


template<typename ImgBaseDataType, typename ImgBaseDataIntType, typename LeafNodeStatistics, typename TAppContext>
void JointClassRegrTree<ImgBaseDataType, ImgBaseDataIntType, LeafNodeStatistics, TAppContext>::GetRandomSampleSubsetForSplitting(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset_full, DataSet<SampleImgPatch, LabelJointClassRegr>& dataset_subsample, int num_samples)
{
    int num_samples_used = min(num_samples, (int)dataset_full.size());
    double max_weight = dataset_full[0]->m_label.class_weight_gt;
    for (size_t i = 1; i < dataset_full.size(); i++)
    {
    	if (dataset_full[i]->m_label.class_weight_gt > max_weight)
    		max_weight = dataset_full[i]->m_label.class_weight_gt;
    }
    max_weight *= num_samples_used;
    dataset_subsample.Clear();
    vector<int> randinds = randPermSTL(dataset_full.size());
    vector<double> class_weights(this->m_appcontext->num_classes, 0.0);

    for (size_t i = 0; i < dataset_full.size(); i++)
    {
    	int c_sample_idx = randinds[i];
    	int c_sample_label = dataset_full[c_sample_idx]->m_label.class_label;
    	double c_sample_weight = dataset_full[c_sample_idx]->m_label.class_weight_gt;
    	if (class_weights[c_sample_label] >= max_weight)
    	{
    		continue;
    	}
    	else
    	{
    		dataset_subsample.AddLabelledSample(dataset_full[c_sample_idx]);
    		class_weights[c_sample_label] += c_sample_weight;
    	}
    }

}



#endif /* JOINTCLASSREGRTREE_CPP_ */

