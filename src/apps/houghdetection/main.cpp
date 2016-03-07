#include <iostream>
#include <string>
#include <fstream>
#include <ostream>
#include <eigen3/Eigen/Core>
#include <vector>
#include <stdexcept>

#include "AppContextJointClassRegr.h"
#include "DataLoaderHoughObject.h"
#include "SampleImgPatch.h"
#include "LabelJointClassRegr.h"
#include "SplitFunctionImgPatch.h"
#include "SplitEvaluatorJointClassRegr.h"
#include "LeafNodeStatisticsJointClassRegr.h"
#include "HoughDetector.h"

#include "JointClassRegrForest.h"
#include "ADJointClassRegrForest.h"

#include "icgrf.h"



/*
 * Overall TODO:
 * - ADF & ARF have to be implemented in this context here!!!
 * - LeafNodeStatistics: meanshift and hillclimb is not implemented yet,
 *   also the update statistics is not implemented!
 * - Ordinal Split function is not implemented yet!
 * - Allow patch sampling over the gt-bbox, only the center of the patch has to be within the bbox!
 * - Always use Eigen vectors, never use std vectors, we can better calculate with them!!!
 */


// typedefs for easier use later
typedef SplitFunctionImgPatch<uchar, float, AppContextJointClassRegr> TSplitFunctionImgPatch;
typedef RandomForest<SampleImgPatch, LabelJointClassRegr, TSplitFunctionImgPatch, SplitEvaluatorJointClassRegr<SampleImgPatch, AppContextJointClassRegr>, LeafNodeStatisticsJointClassRegr<AppContextJointClassRegr>, AppContextJointClassRegr> THoughForest;
typedef Node<SampleImgPatch, LabelJointClassRegr, TSplitFunctionImgPatch, LeafNodeStatisticsJointClassRegr<AppContextJointClassRegr>, AppContextJointClassRegr> HoughNode;



void train(AppContextJointClassRegr* apphp)
{
	// 0) some initial stuff (scaling)
	for (size_t i = 0; i < apphp->patch_size.size(); i++)
		apphp->patch_size[i] = (int)((double)apphp->patch_size[i] * apphp->general_scaling_factor);

	// 1) read the data
	if (!apphp->quiet)
		cout << "Load training data ..." << endl;
	
        DataLoaderHoughObject mydataloader(apphp);
	DataSet<SampleImgPatch, LabelJointClassRegr> dataset_train = mydataloader.LoadTrainData(0);

	int num_train_samples, num_classes, num_target_variables, num_feature_channels, num_z;
	mydataloader.GetTrainDataProperties(num_train_samples, num_classes, num_target_variables, num_feature_channels, num_z);
	// set some information for splitfunctions etc. in the app-context
	apphp->num_classes = num_classes;
	apphp->num_target_variables = num_target_variables;
	apphp->num_feature_channels = num_feature_channels;
	apphp->num_z = num_z;


	// normalize the regression targets of the training data
	std::vector<VectorXd> offset_means, offset_stds;
	mydataloader.NormalizeRegressionTargets(offset_means, offset_stds);

	if (!apphp->quiet)
	{
		std::cout << "Dataset information: " << std::endl;
		std::cout << num_train_samples << " patches, " << num_classes << " classes, ";
		std::cout << num_target_variables << "-d regression targets" << std::endl;
	}



	// 2) prepare the forest - fill the random forest core settings
	RFCoreParameters* rfparams = new RFCoreParameters();
	rfparams->m_debug_on = apphp->debug_on;
	rfparams->m_quiet = apphp->quiet;
	rfparams->m_max_tree_depth = apphp->max_tree_depth;
	rfparams->m_min_split_samples = apphp->min_split_samples;
	rfparams->m_num_node_tests = apphp->num_node_tests;
	rfparams->m_num_node_thresholds = apphp->num_node_thresholds;
	rfparams->m_num_random_samples_for_splitting = apphp->num_random_samples_for_splitting;
	rfparams->m_num_trees = apphp->num_trees;
	rfparams->m_bagging_method = (TREE_BAGGING_TYPE::Enum)apphp->bagging_type;
	rfparams->m_adf_loss_classification = apphp->global_loss_classification;
	rfparams->m_adf_loss_regression = apphp->global_loss_regression;
	cout << (*rfparams) << endl;


	// 3) train the forest
	THoughForest* rf;
	switch (apphp->method)
	{
	case RF_METHOD::HOUGHFOREST:
		rf = new JointClassRegrForest<AppContextJointClassRegr>(rfparams, apphp);
		break;
	case RF_METHOD::ADHOUGHFOREST:
		rf = new ADJointClassRegrForest<AppContextJointClassRegr>(rfparams, apphp);
		break;
	default:
		throw std::runtime_error("main.cpp: unknown rf-method defined!");
	}

	MatrixXd latent_variables = MatrixXd::Zero(dataset_train.size(), 2);
	if (!apphp->quiet)
		std::cout << "Training ... " << std::endl << std::flush;

	rf->Train(dataset_train, latent_variables, offset_means[1], offset_stds[1]);
	if (!apphp->quiet)
		cout << "done" << endl << flush;

	// 4.1) denormalize the regression targets of the training data & of the leafnode statistics
	mydataloader.DenormalizeRegressionTargets(offset_means, offset_stds);
	rf->DenormalizeTargetVariables(offset_means, offset_stds);

	// 4.2) and save it
	rf->Save(apphp->path_trees, latent_variables);

	// 4.3) delete the training samples
	dataset_train.DeleteAllSamples();
}


void houghdetect(AppContextJointClassRegr* apphp)
{
	// 0) some initial stuff (scaling)
	for (size_t i = 0; i < apphp->patch_size.size(); i++)
		apphp->patch_size[i] = (int)((double)apphp->patch_size[i] * apphp->general_scaling_factor);

	// Set the forest parameters
	RFCoreParameters* rfparams = new RFCoreParameters();
	rfparams->m_debug_on = apphp->debug_on;
	rfparams->m_quiet = apphp->quiet;
	rfparams->m_max_tree_depth = apphp->max_tree_depth;
	rfparams->m_min_split_samples = apphp->min_split_samples;
	rfparams->m_num_node_tests = apphp->num_node_tests;
	rfparams->m_num_node_thresholds = apphp->num_node_thresholds;
	rfparams->m_num_random_samples_for_splitting = apphp->num_random_samples_for_splitting;
	rfparams->m_num_trees = apphp->num_trees;
	rfparams->m_bagging_method = (TREE_BAGGING_TYPE::Enum)apphp->bagging_type;
	rfparams->m_adf_loss_classification = apphp->global_loss_classification;
	rfparams->m_adf_loss_regression = apphp->global_loss_regression;

    std::cout << "Obtaining data/config information... " << std::endl;
    DataLoaderHoughObject mydataloader(apphp);
    DataSet<SampleImgPatch, LabelJointClassRegr> dataset_train = mydataloader.LoadTrainData(1);
	int num_train_samples, num_classes, num_target_variables, num_feature_channels, num_z;
	mydataloader.GetTrainDataProperties(num_train_samples, num_classes, num_target_variables, num_feature_channels, num_z);
    cout << "num classes: " << num_classes << endl;
    cout << "num aspects: " << num_z << endl; 
	apphp->num_classes = num_classes;
	apphp->num_z = num_z;
	
	cout << " " << endl;
	std::cout << "Testing ..." << std::endl;
	// Loading the forest
	if (!apphp->quiet)
		std::cout << "Loading the forest" << std::endl;
	THoughForest* rf;
	switch (apphp->method)
	{
	case RF_METHOD::HOUGHFOREST:
		rf = new JointClassRegrForest<AppContextJointClassRegr>(rfparams, apphp);
		break;
	case RF_METHOD::ADHOUGHFOREST:
		rf = new ADJointClassRegrForest<AppContextJointClassRegr>(rfparams, apphp);
		break;
	default:
		throw std::runtime_error("main.cpp: unknown rf-method defined!");
	}
	rf->Load(apphp->path_trees);
	cout << "done" << endl;

	int status = mkdir(apphp->path_bboxes.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (status == -1)
	{
		std::cout << "The directory already exist!" << std::endl;
		// throw std::runtime_error("Could not create bboxespath");
	}

	// init the detector
	HoughDetector hd(rf, apphp);
	hd.DetectList();
}

void analyze_forest(AppContextJointClassRegr* apphp)
{
	// Set the forest parameters
	RFCoreParameters* rfparams = new RFCoreParameters();
	rfparams->m_debug_on = apphp->debug_on;
	rfparams->m_quiet = apphp->quiet;
	rfparams->m_max_tree_depth = apphp->max_tree_depth;
	rfparams->m_min_split_samples = apphp->min_split_samples;
	rfparams->m_num_node_tests = apphp->num_node_tests;
	rfparams->m_num_node_thresholds = apphp->num_node_thresholds;
	rfparams->m_num_random_samples_for_splitting = apphp->num_random_samples_for_splitting;
	rfparams->m_num_trees = apphp->num_trees;
	rfparams->m_bagging_method = (TREE_BAGGING_TYPE::Enum)apphp->bagging_type;
	rfparams->m_adf_loss_classification = apphp->global_loss_classification;
	rfparams->m_adf_loss_regression = apphp->global_loss_regression;

        DataLoaderHoughObject mydataloader(apphp);
	int num_train_samples, num_classes, num_target_variables, num_feature_channels, num_z;
	mydataloader.GetTrainDataProperties(num_train_samples, num_classes, num_target_variables, num_feature_channels, num_z);
	apphp->num_classes = num_classes;


	// Loading the forest
	if (!apphp->quiet)
		std::cout << "Loading the forest" << std::endl;
	THoughForest* rf;
	switch (apphp->method)
	{
	case RF_METHOD::HOUGHFOREST:
		rf = new JointClassRegrForest<AppContextJointClassRegr>(rfparams, apphp);
		break;
	case RF_METHOD::ADHOUGHFOREST:
		rf = new ADJointClassRegrForest<AppContextJointClassRegr>(rfparams, apphp);
		break;
	default:
		throw std::runtime_error("main.cpp: unknown rf-method defined!");
	}
	rf->Load(apphp->path_trees);



	// 5) analyse the trees
	std::vector<std::vector<Node<SampleImgPatch, LabelJointClassRegr, TSplitFunctionImgPatch, LeafNodeStatisticsJointClassRegr<AppContextJointClassRegr>, AppContextJointClassRegr>* > > all_leafs;
	all_leafs = rf->GetAllLeafNodes();
	for (size_t t = 0; t < all_leafs.size(); t++)
	{
		for (size_t i = 0; i < all_leafs[t].size(); i++)
		{
			cout << "Tree " << t << ", node " << i << "/" << all_leafs[t].size() << " in depth " << all_leafs[t][i]->m_depth << ":" << endl;
			all_leafs[t][i]->m_leafstats->Print();
		}
	}
}



int main(int argc, char* argv[])
{
	// 1) read input arguments
	std::string path_configfile;
	if (argc < 2)
	{
		std::cout << "Specify a config file" << std::endl;
		exit(-1);
	}
	else
	{
		path_configfile = argv[1];
	}


	// 2) read configuration
	AppContextJointClassRegr apphp;
	apphp.Read(path_configfile);

	if (!apphp.quiet)
		std::cout << "Parsed configuration file from " << path_configfile << std::endl;


	//try
	//{
		switch (apphp.mode)
		{
		case 0:
			train(&apphp);
			break;
		case 1:
			houghdetect(&apphp);
			break;
		case 2:
			train(&apphp);
			houghdetect(&apphp);
			break;
		case 3:
			analyze_forest(&apphp);
			break;
		default:
			throw std::runtime_error("main.cpp: wrong mode specified!");
			break;
		}
	//}
	//catch (std::exception& e)
	//{
	//	std::cout << "ERROR occured, exception thrown:" << std::endl;
	//	std::cout << e.what() << std::endl;
	//}

	if (!apphp.quiet)
		std::cout << "Program should be finished now ..." << std::endl << std::flush;
	return 0;
}
