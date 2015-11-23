/*
 * SplitEvaluatorMLRegr.h
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 */

#ifndef SPLITEVALUATORMLREGR_H_
#define SPLITEVALUATORMLREGR_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <vector>
#include <math.h>
#include "AppContextML.h"
#include "LabelMLRegr.h"
#include "SampleML.h"

#include "icgrf.h"

using namespace std;
using namespace Eigen;



template<typename Sample>
class SplitEvaluatorMLRegr
{
public:

	// Constructors & destructors
    SplitEvaluatorMLRegr(AppContextML* appcontextin, int depth, DataSet<Sample, LabelMLRegr>& dataset);
    virtual ~SplitEvaluatorMLRegr();

    bool DoFurtherSplitting(DataSet<Sample, LabelMLRegr>& dataset, int depth);
    bool CalculateScoreAndThreshold(DataSet<Sample, LabelMLRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold);

protected:

    // Classification stuff
    bool CalculateMVNPluginAndThreshold(DataSet<Sample, LabelMLRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double,double>& score_and_threshold);

	// members
    AppContextML* m_appcontext;

};

#include "SplitEvaluatorMLRegr.cpp"






#endif /* SPLITFUNCTIONSML_H_ */
