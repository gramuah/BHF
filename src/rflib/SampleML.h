/*
 * SampleML.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef SAMPLEML_H_
#define SAMPLEML_H_

#include <vector>
#include <fstream>
#include <stdexcept>

struct SampleML
{
	SampleML() {}

	std::vector<double> features;

	void Save(std::ofstream& out)
	{
		throw std::logic_error("SampleML.Save(): Not implemented yet");
	}
	void Load(std::ifstream& in)
	{
		throw std::logic_error("SampleML.Load(): Not implemented yet");
	}
};

#endif /* SAMPLEML_H_ */
