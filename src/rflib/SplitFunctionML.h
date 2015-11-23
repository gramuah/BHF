/*
 * SplitFunctionML.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 */

#ifndef SPLITFUNCTIONML_H_
#define SPLITFUNCTIONML_H_

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/LU>
#include <vector>
#include <math.h>
#include "AppContextML.h"
#include "LabelMLClass.h"
#include "SampleML.h"

#include "icgrf.h"

using namespace std;
using namespace Eigen;



class SplitFunctionML
{
public:

	// Constructors & destructors
    explicit SplitFunctionML(AppContextML* appcontextin);
    virtual ~SplitFunctionML();

    void SetRandomValues();
    void SetThreshold(double inth);
    void SetSplit(SplitFunctionML* spfin);
    int Split(SampleML& sample) const;
    double CalculateResponse(SampleML& sample) const;

    // I/O method
    void Save(std::ofstream& out);
    void Load(std::ifstream& in);

    vector<int> m_feature_indices;
    vector<double> m_feature_weights;
    double m_th;

protected:

    // members
    AppContextML* m_appcontext;

};


#endif /* SPLITFUNCTIONML_H_ */
