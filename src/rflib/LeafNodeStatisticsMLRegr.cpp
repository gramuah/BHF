/*
 * LeafNodeStatisticsMLRegr.cpp
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 */

#include "LeafNodeStatisticsMLRegr.h"



LeafNodeStatisticsMLRegr::LeafNodeStatisticsMLRegr(AppContextML* appcontextin) : m_appcontext(appcontextin)
{
	this->m_prediction = Eigen::VectorXd::Zero(m_appcontext->num_target_variables);
}

LeafNodeStatisticsMLRegr::~LeafNodeStatisticsMLRegr() { };


void LeafNodeStatisticsMLRegr::Aggregate(DataSet<SampleML, LabelMLRegr>& dataset, int full, int is_final_leaf)
{
	if (!is_final_leaf)
	{
		// reset the target prediction
		this->m_prediction = VectorXd::Zero(m_appcontext->num_target_variables);

		// fill the prediction
		this->m_num_samples = (int)dataset.size();
		for (int s = 0; s < this->m_num_samples; s++)
			this->m_prediction += dataset[s]->m_label.regr_target;

		// normalize it
		m_prediction /= (double)m_num_samples;
	}

}


void LeafNodeStatisticsMLRegr::Aggregate(LeafNodeStatisticsMLRegr* leafstatsin)
{
	if (this->m_num_samples == 0)
		this->m_prediction = Eigen::VectorXd::Zero(this->m_appcontext->num_target_variables);

	// add both de-normalzed predictions
	this->m_prediction = this->m_prediction * this->m_num_samples + leafstatsin->m_prediction * leafstatsin->m_num_samples;

	// increase the number of sample in the leaf statistics
	this->m_num_samples += leafstatsin->m_num_samples;

	// normalize the new statistics
	this->m_prediction /= (double)this->m_num_samples;
}


void LeafNodeStatisticsMLRegr::UpdateStatistics(LabelledSample<SampleML, LabelMLRegr>* labelled_sample)
{
	if (m_num_samples == 0)
	{
		this->m_prediction = labelled_sample->m_label.regr_target;
		m_num_samples = 1;
		return;
	}

	// denormalize the prediction target
	m_prediction *= (double)m_num_samples;

	// add the sample to the corresponding bin
	m_prediction += labelled_sample->m_label.regr_target;

	// increase the total number of samples
	m_num_samples++;

	// normalize the histogram again
	m_prediction /= (double)m_num_samples;
}

LeafNodeStatisticsMLRegr LeafNodeStatisticsMLRegr::Average(std::vector<LeafNodeStatisticsMLRegr*> leafstats, AppContextML* apphp)
{
	// add the predictions from the leafnode stats ...
	LeafNodeStatisticsMLRegr ret_stats(apphp);
	for (size_t i = 0; i < leafstats.size(); i++)
		ret_stats.m_prediction += leafstats[i]->m_prediction;

	// ... noramlize it (mean) ...
	ret_stats.m_prediction /= (double)leafstats.size();

	// ... and return the result.
	return ret_stats;
}

void LeafNodeStatisticsMLRegr::DenormalizeTargetVariables(Eigen::VectorXd mean, Eigen::VectorXd std)
{
	throw std::logic_error("ERROR: this method is not implemented yet (LeafNodeStatisticsMLRegr.cpp:133");
}

void LeafNodeStatisticsMLRegr::AddTarget(LeafNodeStatisticsMLRegr* leafnodestats_src)
{
	this->m_prediction += leafnodestats_src->m_prediction;
}


std::vector<double> LeafNodeStatisticsMLRegr::CalculateADFTargetResidual(LabelMLRegr gt_label, int prediction_type)
{
	// prediction type can be ignored here, as it is clear that we are in a regression task
	std::vector<double> ret_vec(this->m_prediction.size());

	for (size_t v = 0; v < this->m_prediction.rows(); v++)
		ret_vec[v] = this->m_prediction(v) - gt_label.regr_target_gt(v);

	// return the residual vector
	return ret_vec;
}


void LeafNodeStatisticsMLRegr::Save(std::ofstream& out)
{
	out << m_num_samples << " ";
	out << m_prediction.rows() << " ";
	for (int c = 0; c < m_prediction.rows(); c++)
		out << m_prediction(c) << " ";
	out << endl;
}


void LeafNodeStatisticsMLRegr::Load(std::ifstream& in)
{
	int target_dim;
	in >> m_num_samples >> target_dim;
	m_prediction = Eigen::VectorXd::Zero(target_dim);
	for (int c = 0; c < m_prediction.rows(); c++)
		in >> m_prediction(c);
}


