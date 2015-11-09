/*
 * SplitEvaluatorMLRegr.cpp
 *
 * Author: Samuel Schulter, Christian Leistner, Paul Wohlhart, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 *
 */

#ifndef SPLITEVALUATORMLREGR_CPP_
#define SPLITEVALUATORMLREGR_CPP_

#include "SplitEvaluatorMLRegr.h"


template<typename Sample>
SplitEvaluatorMLRegr<Sample>::SplitEvaluatorMLRegr(AppContextML* appcontextin, int depth, DataSet<Sample, LabelMLRegr>& dataset) : m_appcontext(appcontextin)
{
}


template<typename Sample>
SplitEvaluatorMLRegr<Sample>::~SplitEvaluatorMLRegr()
{
}


template<typename Sample>
bool SplitEvaluatorMLRegr<Sample>::DoFurtherSplitting(DataSet<Sample, LabelMLRegr>& dataset, int depth)
{
	if (depth >= (this->m_appcontext->max_tree_depth-1) || (int)dataset.size() < this->m_appcontext->min_split_samples)
		return false;
	return true;
}


template<typename Sample>
bool SplitEvaluatorMLRegr<Sample>::CalculateScoreAndThreshold(DataSet<Sample, LabelMLRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold)
{
	if (m_appcontext->split_function_type == SPLITFUNCTION_TYPE::ORDINAL)
		throw std::logic_error("SplitEvaluatorMLRegr: ordinal splits not implemented for regression evaluator!");

	switch (m_appcontext->splitevaluation_type_regression)
	{
	case SPLITEVALUATION_TYPE_REGRESSION::REDUCTION_IN_VARIANCE:
		throw std::logic_error("SplitEvaluatorMLRegr::Red-In-Var not yet implemented");
		break;
	case SPLITEVALUATION_TYPE_REGRESSION::DIFF_ENTROPY_GAUSS:
		return CalculateMVNPluginAndThreshold(dataset, responses, score_and_threshold);
		break;
	case 20: // Diagonal Approx. of MVN-Plugin
		// TODO: this could be implemented, use the method from above and make a switch:
		// 		 it's just an element-wise multiplication with the identity matrix!
		throw std::logic_error("SplitEvaluatorMLRegr::Not implemented regression loss type");
		return false;
		break;
	case 21: // MultiVariateNormal-UniformMinVarianceUnbiasedEstimate, see [Nowozin, ICML'12]
		throw std::logic_error("SplitEvaluatorMLRegr::Not implemented regression loss type");
		return false;
		break;
	case 22: // 1-NearestNeighbor loss, see [Nowozin, ICML'12]
		throw std::logic_error("SplitEvaluatorMLRegr::Not implemented regression loss type");
		return false;
		break;
	default:
		throw std::logic_error("SplitEvaluatorMLRegr::splitfunction_regression_loss_type not defined");
		return false;
		break;
	}
}






// PROTECTED/HELPER METHODS
template<typename Sample>
bool SplitEvaluatorMLRegr<Sample>::CalculateMVNPluginAndThreshold(DataSet<Sample, LabelMLRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double,double>& score_and_threshold)
{
	// In: samples, sorted responses, out: optimality-measure + threshold

	// Initialize the variables and counters
	double InfoGain, LEntropy, REntropy, bestThreshold = 0.0, BestInfoGain = 1e16;
	double LTotal = 0.0, RTotal = 0.0, LSqNormTotal = 0.0, RSqNormTotal = 0.0;
	VectorXd RMean = VectorXd::Zero(m_appcontext->num_target_variables);
	VectorXd LMean = VectorXd::Zero(m_appcontext->num_target_variables);
	VectorXd RSum = VectorXd::Zero(m_appcontext->num_target_variables);
	VectorXd LSum = VectorXd::Zero(m_appcontext->num_target_variables);
	MatrixXd LCov = MatrixXd::Zero(m_appcontext->num_target_variables, m_appcontext->num_target_variables);
	MatrixXd RCov = MatrixXd::Zero(m_appcontext->num_target_variables, m_appcontext->num_target_variables);
	vector<int> RSamples, LSamples;
	bool found = false;

	// Calculate random thresholds and sort them
	double min_response = responses[0].first;
	double max_response = responses[responses.size()-1].first;
	double d = (max_response - min_response);
	vector<double> random_thresholds(m_appcontext->num_node_thresholds, 0.0);
	for (int i = 0; i < random_thresholds.size(); i++)
		random_thresholds[i] = (randDouble() * d) + min_response;
	sort(random_thresholds.begin(), random_thresholds.end());

	// First, put everything in the right node
	RSamples.resize(responses.size());
	for (int r = 0; r < responses.size(); r++)
	{
		double csw = dataset[responses[r].second]->m_weight;
		Eigen::VectorXd cst = dataset[responses[r].second]->m_label.regr_target;
		RSum += csw * cst;
		RTotal += csw;
		RSamples[r] = responses[r].second;
	}
	RMean = RSum / RTotal;

	// Now, iterate all responses and calculate Gini indices at the cutoff points (thresholds)
	int th_idx = 0;
	bool stop_search = false;
	for (int r = 0; r < responses.size(); r++)
	{
		// if the current sample is smaller than the current threshold put it to the left side
		if (responses[r].first <= random_thresholds[th_idx])
		{
			// move the current response from the right node to the left node
			double csw = dataset[responses[r].second]->m_weight;
			Eigen::VectorXd cst = dataset[responses[r].second]->m_label.regr_target;
			RSum -= csw * cst;
			RTotal -= csw;
			if (RTotal < 0.0) // should never happen
				RTotal = 0.0;
			LSum += csw * cst;
			LTotal += csw;
			LSamples.push_back(RSamples[0]);
			RSamples.erase(RSamples.begin());
		}
		else
		{
			if (LTotal > 0.0 && RTotal > 0.0)
			{
				// now, we have to check the split quality, this would be a valid split

				// RIGHT: Weighted mean
				RMean = RSum / RTotal;
				// weighted co-variance [http://en.wikipedia.org/wiki/Sample_mean_and_sample_covariance#Weighted_samples]
				RCov = MatrixXd::Zero(m_appcontext->num_target_variables, m_appcontext->num_target_variables);
				RSqNormTotal = 0.0;
				for (int s = 0; s < RSamples.size(); s++)
				{
					Eigen::VectorXd cst = dataset[RSamples[s]]->m_label.regr_target;
					RCov += dataset[RSamples[s]]->m_weight * ((cst - RMean) * (cst - RMean).transpose());
					RSqNormTotal += pow(dataset[RSamples[s]]->m_weight/RTotal, 2.0);
				}
				RCov /= RTotal; // this normalization is important: sum_i w_i = 1 should hold for the weighted covariance
				if (RSqNormTotal < 1.0) // this happens if only one sample is available!
					RCov /= (1.0 - RSqNormTotal);
				double RCovDet = RCov.determinant();
				if (RCovDet <= 0.0) // happens if 2 samples only available -> one eigval=0 -> minimal quantization errors -> -5e-11
					RCovDet = 1e-10;
				//REntropy = (double)m_num_target_variables/2.0 - (double)m_num_target_variables/2.0 * log(2.0 * M_PI) + 0.5 * log(RCovDet);
				REntropy = log(RCovDet);
				if (REntropy <= 0.0)
					REntropy = 0.0;

				// LEFT: Weighted mean
				LMean = LSum / LTotal;
				// weighted co-variance
				LCov = MatrixXd::Zero(m_appcontext->num_target_variables, m_appcontext->num_target_variables);
				LSqNormTotal = 0.0;
				for (int s = 0; s < LSamples.size(); s++)
				{
					Eigen::VectorXd cst = dataset[LSamples[s]]->m_label.regr_target;
					LCov += dataset[LSamples[s]]->m_weight * ((cst - LMean) * (cst - LMean).transpose());
					LSqNormTotal += pow(dataset[LSamples[s]]->m_weight/LTotal, 2.0);
				}
				if (LSamples.size() == 0)
				{
					cout << LCov << endl;
					cout << LSqNormTotal << endl;
				}
				LCov /= LTotal;
				if (LSqNormTotal < 1.0) // this happens if only one sample is available!
					LCov /= (1.0 - LSqNormTotal);
				double LCovDet = LCov.determinant();
				if (LCovDet <= 0.0) // happens if 2 samples only available -> one eigval=0 -> minimal quantization errors -> -5e-11
					LCovDet = 1e-10;
				//LEntropy = (double)m_num_target_variables/2.0 - (double)m_num_target_variables/2.0 * log(2.0 * M_PI) + 0.5 * log(LCovDet);
				LEntropy = log(LCovDet);
				if (LEntropy <= 0.0)
					LEntropy = 0.0;

				// combine left and right entropy measures (weighted!!!)
				InfoGain = (LTotal*LEntropy + RTotal*REntropy) / (LTotal + RTotal);

				if (this->m_appcontext->debug_on)
					cout << "Eval: " << InfoGain << ", LTotal=" << LTotal << ", RTotal=" << RTotal << "(" << LEntropy << ", " << REntropy << ")" << endl;

				if (InfoGain < BestInfoGain)
				{
					BestInfoGain = InfoGain;
					bestThreshold = random_thresholds[th_idx];
					found = true;
				}
			}

			// next, we have to find the next random threshold that is larger than the current response
			// -> there might be several threshold within the gap between the last response and this one.
			while (responses[r].first > random_thresholds[th_idx])
			{
				if (th_idx < (random_thresholds.size()-1))
				{
					th_idx++;
					// CAUTION::: THIS HAS TO BE INCLUDED !!!!!!!!!!!??????
					r--; // THIS IS IMPORTANT, WE HAVE TO CHECK THE CURRENT SAMPLE AGAIN!!!
				}
				else
				{
					stop_search = true;
					break; // all thresholds tested
				}
			}
			// now, we can go on with the next response ...
		}
		if (stop_search)
			break;
	}

	score_and_threshold.first = BestInfoGain;
	score_and_threshold.second = bestThreshold;
	return found;
}


#endif /* SPLITEVALUATORMLREGR_CPP_ */
