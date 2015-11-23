/*
 * DataSample.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef LABELLEDSAMPLE_H_
#define LABELLEDSAMPLE_H_

#include <fstream>


template<typename Sample, typename Label>
class LabelledSample
{
public:

    explicit LabelledSample(Sample insample, Label inlabel, double inweight, int inglobalindex) :
    		m_sample(insample), m_label(inlabel), m_weight(inweight), m_global_index(inglobalindex)
    {
    }

    ~LabelledSample() { }

    void Save(std::ofstream& out);
    void Load(std::ifstream& in);

    // =======================================================
    // members
    Sample m_sample;
    Label m_label;
    double m_weight;
    int m_global_index;
};


#include "LabelledSample.cpp"

#endif /* LABELLEDSAMPLE_H_ */
