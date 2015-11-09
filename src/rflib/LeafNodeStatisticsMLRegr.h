/*
 * LeafNodeStatisticsMLRegr.h
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 */

#ifndef LEAFNODESTATISTICSMLREGR_H_
#define LEAFNODESTATISTICSMLREGR_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU> // determinant!!!
#include <vector>
#include <math.h> // M_PI
#include <fstream>

#include "Interfaces.h"
#include "LabelledSample.h"
#include "SampleML.h"
#include "LabelMLRegr.h"
#include "AppContextML.h"
#include "DataSet.h"

using namespace std;
using namespace Eigen;



class LeafNodeStatisticsMLRegr
{
public:

	// Constructors & destructors
	LeafNodeStatisticsMLRegr(AppContextML* appcontextin);
	virtual ~LeafNodeStatisticsMLRegr();

	// data methods
	virtual void Aggregate(DataSet<SampleML, LabelMLRegr>& dataset, int full, int is_final_leaf);
	virtual void Aggregate(LeafNodeStatisticsMLRegr* leafstatsin);
	virtual void UpdateStatistics(LabelledSample<SampleML, LabelMLRegr>* labelled_sample);
	static LeafNodeStatisticsMLRegr Average(std::vector<LeafNodeStatisticsMLRegr*> leafstats, AppContextML* apphp);
	virtual void DenormalizeTargetVariables(Eigen::VectorXd mean, Eigen::VectorXd std);

	// ADF-specific
	virtual void AddTarget(LeafNodeStatisticsMLRegr* leafnodestats);
	virtual std::vector<double> CalculateADFTargetResidual(LabelMLRegr gt_label, int prediction_type);

	// I/O methods
	virtual void Save(std::ofstream& out);
	virtual void Load(std::ifstream& in);


	// public memberes
	int m_num_samples;
	Eigen::VectorXd m_prediction;

protected:

	// protected members
    AppContextML* m_appcontext;

};










#endif /* LEAFNODESTATISTICSML_H_ */
