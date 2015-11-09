/*
 * SplitFunctionML.cpp
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#include "SplitFunctionML.h"



SplitFunctionML::SplitFunctionML(AppContextML* appcontextin) : m_appcontext(appcontextin)
{
}

SplitFunctionML::~SplitFunctionML()
{
}


void SplitFunctionML::SetRandomValues()
{
	if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::FEATURE_THRESHOLD)
	{
		m_feature_indices.resize(1);
		m_feature_indices[0] = randInteger(0, m_appcontext->num_feature_channels-1);
	}
	else if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::OBLIQUE)
	{
		m_feature_indices.resize(m_appcontext->splitfunction_oblique_k);
		m_feature_weights.resize(m_appcontext->splitfunction_oblique_k+1);
		for (int i = 0; i < m_feature_indices.size(); i++)
		{
			m_feature_indices[i] = randInteger(0, m_appcontext->num_feature_channels-1);
			m_feature_weights[i] = randDouble();
		}
		m_feature_weights[m_feature_weights.size()-1] = randDouble(); // bias term
	}
	else if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::ORDINAL)
	{
		// INFO: for the ordinal splits, we also have a parameter K. For ease of less coding,
		// we simply use the k parameter from the oblique splits ;)
		m_feature_indices.resize(m_appcontext->splitfunction_oblique_k);
		for (int i = 0; i < m_feature_indices.size(); i++)
			m_feature_indices[i] = randInteger(0, m_appcontext->num_feature_channels-1);
	}
	else
	{
		throw std::runtime_error("SplitFunctionML::ERROR: undefined split-function-type!");
	}
}

void SplitFunctionML::SetThreshold(double inth)
{
	this->m_th = inth;
}

void SplitFunctionML::SetSplit(SplitFunctionML* spfin)
{
	m_feature_indices = spfin->m_feature_indices;
	m_feature_weights = spfin->m_feature_weights;
	m_th = spfin->m_th;
}


int SplitFunctionML::Split(SampleML& sample) const
{
	if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::FEATURE_THRESHOLD ||
			m_appcontext->split_function_type == SPLITFUNCTION_TYPE::OBLIQUE)
	{
		if (this->CalculateResponse(sample) < m_th)
			return 0;
		else
			return 1;
	}
	else if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::ORDINAL)
	{
		// set the response to the first index
		int response_ind = 0;
		double response_val = sample.features[m_feature_indices[0]];
		// find the maximum index
		for (int i = 1; i < m_feature_indices.size(); i++)
		{
			if (sample.features[m_feature_indices[i]] > response_val)
			{
				response_val = sample.features[m_feature_indices[i]];
				response_ind = i;
			}
		}
		if (response_ind == (int)m_th)
			return 0;
		else
			return 1;
	}
	else
	{
		throw std::runtime_error("SplitFunctionML::ERROR: undefined split-function-type!");
		return 0;
	}
}


double SplitFunctionML::CalculateResponse(SampleML& sample) const
{
	double response = 0.0;
	if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::FEATURE_THRESHOLD)
	{
		response = sample.features[m_feature_indices[0]];
	}
	else if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::OBLIQUE)
	{
		//response = 0.0;
		response = m_feature_weights[m_feature_weights.size()-1]; // bias term
		for (int i = 0; i < m_feature_indices.size(); i++)
			response += sample.features[m_feature_indices[i]] * m_feature_weights[i];
	}
	else if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::ORDINAL)
	{
		response = 0.0;
		double response_val = sample.features[m_feature_indices[0]];
		for (int i = 1; i < m_feature_indices.size(); i++)
		{
			if (sample.features[m_feature_indices[i]] > response_val)
			{
				response_val = sample.features[m_feature_indices[i]];
				response = static_cast<double>(i);
			}
		}
	}
	else
	{
		throw std::runtime_error("SplitFunctionML::ERROR: undefined split-function-type!");
	}
	return response;
}


void SplitFunctionML::Save(std::ofstream& out)
{
	// store this
	out << m_feature_indices.size() << " ";
	for (int i = 0; i < m_feature_indices.size(); i++)
		out << m_feature_indices[i] << " ";
	out << endl;
	out << m_feature_weights.size() << " ";
	for (int i = 0; i < m_feature_weights.size(); i++)
		out << m_feature_weights[i] << " ";
	out << endl;
	out << m_th << endl;
}


void SplitFunctionML::Load(std::ifstream& in)
{
	int num_split_features;
	in >> num_split_features;
	m_feature_indices.resize(num_split_features);
	for (int i = 0; i < m_feature_indices.size(); i++)
		in >> m_feature_indices[i];
	int num_split_weights;
	in >> num_split_weights; // should of course be the same ...
	m_feature_weights.resize(num_split_weights);
	for (int i = 0; i < m_feature_weights.size(); i++)
		in >> m_feature_weights[i];
	in >> m_th;
}

