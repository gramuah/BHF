/*
 * SampleImgPatch.h
 */

#ifndef SAMPLEIMGPATCH_H_
#define SAMPLEIMGPATCH_H_

#include <vector>
#include <fstream>
#include <stdexcept>
#include "opencv2/opencv.hpp"
#include <vector>

struct SampleImgPatch
{
	SampleImgPatch() {};

	size_t imgid;
	std::vector<cv::Mat> features;
	cv::Mat normalization_feature_mask; // this is for the case of Haar-like features (if a normalization has to be perfomed not over the whole area, e.g., for depth data!)
	int x;
	int y;

	void Save(std::ofstream& out)
	{
		throw std::logic_error("SampleML.Save() not implemented yet!");
	}
	void Load(std::ifstream& in)
	{
		throw std::logic_error("SampleML.Load() not implemented yet!");
	}
};

#endif /* SAMPLEML_H_ */
