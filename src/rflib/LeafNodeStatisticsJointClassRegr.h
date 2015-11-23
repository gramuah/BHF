/*
 * LeafNodeStatisticsJointClassRegr.h
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 */

#ifndef LEAFNODESTATISTICSJOINTCLASSREGR_H_
#define LEAFNODESTATISTICSJOINTCLASSREGR_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <vector>
#include <math.h>
#include <fstream>

#include "Interfaces.h"
#include "LabelledSample.h"
#include "SampleImgPatch.h"
#include "LabelJointClassRegr.h"
#include "AppContext.h"
#include "DataSet.h"

using namespace std;
using namespace Eigen;

struct residual{
	std::vector<double> ret_vec;
	int bestz;	
};

template<typename TAppContext>
class LeafNodeStatisticsJointClassRegr
{
public:

	// Constructors & destructors
	LeafNodeStatisticsJointClassRegr(TAppContext* appcontextin);
	virtual ~LeafNodeStatisticsJointClassRegr();

	// data methods
	virtual void Aggregate(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, int is_final_leaf);
	virtual void Aggregate(LeafNodeStatisticsJointClassRegr* leafstatsin);
	virtual void UpdateStatistics(LabelledSample<SampleImgPatch, LabelJointClassRegr>* labelled_sample);
	static LeafNodeStatisticsJointClassRegr<TAppContext> Average(std::vector<LeafNodeStatisticsJointClassRegr*> leafstats, LabelledSample<SampleImgPatch, LabelJointClassRegr>* sample, int d, Eigen::VectorXd mean, Eigen::VectorXd std, std::vector<cv::Mat> hough_map, TAppContext* apphp);
	static LeafNodeStatisticsJointClassRegr<TAppContext> Sum(std::vector<LeafNodeStatisticsJointClassRegr*> leafstats, TAppContext* apphp);
	virtual void DenormalizeTargetVariables(Eigen::VectorXd mean, Eigen::VectorXd std);
	virtual void DenormalizeTargetVariables(std::vector<Eigen::VectorXd> mean, std::vector<Eigen::VectorXd> std);

	// ADF related methods
	virtual void AddTarget(LeafNodeStatisticsJointClassRegr* leafnodestats);
	virtual struct residual CalculateADFTargetResidual(LabelJointClassRegr gt_label, vector <Eigen::MatrixXd> m_hough_img_prediction, Eigen::VectorXd mean, Eigen::VectorXd std, int s, int prediction_type);
	virtual std::vector<double> CalculateADFTargetResidual_class(LabelJointClassRegr gt_label, int prediction_type);

	// Analysis methods
	virtual void Print();

	// I/O methods
	virtual void Save(std::ofstream& out, Eigen::MatrixXd& latent_variables);
	virtual void Load(std::ifstream& in);


	// public memberes
	int m_num_samples;
	double m_total_samples_weight;
	
	//int m_num_pos_samples;
	vector<int> m_num_samples_class;
	vector<int> m_num_samples_latent;
	vector<double> m_class_histogram;
	
	vector<vector<Eigen::VectorXd> > m_votes;
	vector <Eigen::MatrixXd> m_prediction;
	vector <Eigen::MatrixXd> m_hough_img_prediction;
	vector<vector<Eigen::VectorXd> > m_offsets;
	vector<vector<Eigen::VectorXd> > m_regr_target;
	vector<vector<int> > m_latent_label;
	vector<vector<int> > m_latent_prediction;
	vector<vector<double> > m_azimuth;
	vector<vector<double> > m_zenith;
	vector<vector<double> > m_vote_weights;
	vector<int> patch_id;
	vector<int> img_id;
	vector<double> m_pseudoclass_histogram;
	vector<vector<Eigen::VectorXd> > m_prediction_centers;
	Eigen::VectorXd m_intermediate_prediction;


protected:

	void AggregateRegressionTargets(DataSet<SampleImgPatch, LabelJointClassRegr> dataset);
	void Aggregate_All(DataSet<SampleImgPatch, LabelJointClassRegr> dataset, int lblIdx, vector<int> sample_indices);
	void Aggregate_Mean(DataSet<SampleImgPatch, LabelJointClassRegr> dataset, int lblIdx, vector<int> sample_indices);
	void Aggregate_HillClimb(DataSet<SampleImgPatch, LabelJointClassRegr> dataset, int lblIdx, vector<int> sample_indices);
	void Aggregate_MeanShift(DataSet<SampleImgPatch, LabelJointClassRegr> dataset, int lblIdx, vector<int> sample_indices, int num_modes);

	// protected members
	TAppContext* m_appcontext;

};


#include "LeafNodeStatisticsJointClassRegr.cpp"



#endif /* LEAFNODESTATISTICSJOINTCLASSREGR_H_ */
