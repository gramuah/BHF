
#ifndef SPLITEVALUATORJOINTCLASSREGR_CPP_
#define SPLITEVALUATORJOINTCLASSREGR_CPP_


#include "SplitEvaluatorJointClassRegr.h"



template<typename Sample, typename TAppContext>
SplitEvaluatorJointClassRegr<Sample, TAppContext>::SplitEvaluatorJointClassRegr(TAppContext* appcontextin, int depth, DataSet<Sample, LabelJointClassRegr>& dataset) : m_appcontext(appcontextin)
{
	// 0 = classification, 1 = regression
	this->m_eval_type = randInteger(0, 1);
	this->m_eval_regr_type = 0;//randInteger(0, 1);
	// NEW interpretation of this parameter:
	// 0 means: only classification
	if (m_appcontext->depth_regression_only == 0)
	{
		this->m_eval_type = 0;
	}
	else
	{
		bool is_positive_value = false;
		if (m_appcontext->depth_regression_only > 0)
			is_positive_value = true;

		if (is_positive_value)
		{
			if (depth >= (m_appcontext->depth_regression_only-1))
				this->m_eval_type = 1;
		}
		else
		{
			if (depth < ((-m_appcontext->depth_regression_only)-1))
				this->m_eval_type = 1;
		}

		// check if only samples from a single voting-class are left! -> then we make regression
		if (this->m_eval_type == 0)
		{
			bool is_pure = true;
			int start_class = dataset[0]->m_label.class_label;
			bool is_voteallowed = dataset[0]->m_label.vote_allowed;
			for (size_t s = 1; s < dataset.size(); s++)
			{
				if (dataset[s]->m_label.class_label != start_class)
				{
					is_pure = false;
					break;
				}
			}
			if (is_pure && is_voteallowed) // only a single class left that also stores offset vectors -> make regression node
			{
				this->m_eval_type = 1;
			}
		}
	}

	if (this->m_appcontext->debug_on)
		std::cout << "Splittype: " << this->m_eval_type << std::endl;
}


template<typename Sample, typename TAppContext>
SplitEvaluatorJointClassRegr<Sample, TAppContext>::~SplitEvaluatorJointClassRegr()
{
}


template<typename Sample, typename TAppContext>
bool
SplitEvaluatorJointClassRegr<Sample, TAppContext>::DoFurtherSplitting(DataSet<Sample, LabelJointClassRegr>& dataset, int depth)
{
	if (depth >= (this->m_appcontext->max_tree_depth-1) || (int)dataset.size() < this->m_appcontext->min_split_samples)
		return false;

	// Test pureness of the node (maybe this should be softened?!?)
	int startLabel = dataset[0]->m_label.class_label;
	for (size_t s = 0; s < dataset.size(); s++)
		if (dataset[s]->m_label.class_label != startLabel)
			return true;

	// If the data is pure according to class labels, there is one case we DO NOT stop splitting:
	// if this class label also has voting elements, then we do further splitting (regression!)
	if (dataset[0]->m_label.vote_allowed == true)
		return true;

	return false;
}


template<typename Sample, typename TAppContext>
bool
SplitEvaluatorJointClassRegr<Sample, TAppContext>::CalculateScoreAndThreshold(DataSet<Sample, LabelJointClassRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold)
{
	if (m_eval_type == 0) // classification
	{
		int use_gini = 0;
		return this->CalculateEntropyAndThreshold(dataset, responses, score_and_threshold, use_gini);
	}
	else if (m_eval_type == 1) // regression
	{
		if (m_appcontext->splitevaluation_type_regression == SPLITEVALUATION_TYPE_REGRESSION::REDUCTION_IN_VARIANCE){
			return this->CalculateOffsetCompactnessAndThresholdOnline(dataset, responses, score_and_threshold);
		}else{
			if (m_eval_regr_type == 0){
				return this->CalculateOffsetCompactnessAndThreshold(dataset, responses, score_and_threshold);
			}else{		
				return this->CalculatePoseCompactnessAndThreshold(dataset, responses, score_and_threshold);
			}
		}
	}
	else
	{
		throw std::runtime_error("SplitEvaluator (calc score and th): eval type unknown!");
		return false;
	}
}







// Private / Helper methods
template<typename Sample, typename TAppContext>
bool
SplitEvaluatorJointClassRegr<Sample, TAppContext>::CalculateEntropyAndThreshold(DataSet<Sample, LabelJointClassRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold, int use_gini)
{
	// Multi-class enabled

	// Initialize the counters
	double DGini, LGini, RGini, LTotal = 0.0, RTotal = 0.0, bestThreshold = 0.0, bestDGini = 1e16;
	vector<double> LCount(m_appcontext->num_classes, 0.0), RCount(m_appcontext->num_classes, 0.0);
	bool found = false;

	// Calculate random thresholds and sort them
	double min_response = responses[0].first;
	double max_response = responses[responses.size()-1].first;

	double d = (max_response - min_response);
	vector<double> random_thresholds(m_appcontext->num_node_thresholds, 0.0);
	for (int i = 0; i < random_thresholds.size(); i++)
	{
		random_thresholds[i] = (randDouble() * d) + min_response;
	}
	sort(random_thresholds.begin(), random_thresholds.end());

	// First, put everything in the right node
	for (int r = 0; r < responses.size(); r++)
	{
		int labelIdx = dataset[responses[r].second]->m_label.class_label;
		double sample_w = dataset[responses[r].second]->m_label.class_weight;
		RCount[labelIdx] += sample_w;
		RTotal += sample_w;
	}

	// Now, iterate all responses and calculate Gini indices at the cutoff points (thresholds)
	int th_idx = 0;
	bool stop_search = false;
	for (int r = 0; r < responses.size(); r++)
	{
		// if the current sample is smaller than the current threshold put it to the left side
		if (responses[r].first <= random_thresholds[th_idx])
		{
			double cur_sample_weight = dataset[responses[r].second]->m_label.class_weight;
			RTotal -= cur_sample_weight;
			if (RTotal < 0.0)
				RTotal = 0.0;
			LTotal += cur_sample_weight;
			int labelIdx = dataset[responses[r].second]->m_label.class_label;
			RCount[labelIdx] -= cur_sample_weight;
			if (RCount[labelIdx] < 0.0)
				RCount[labelIdx] = 0.0;
			LCount[labelIdx] += cur_sample_weight;
		}
		else
		{
			// ok, now we found the first sample having higher response than the current threshold

			// now, we have to check the Gini index, this would be a valid split
			LGini = 0.0, RGini = 0.0;
			if (use_gini)
			{
				for (int c = 0; c < LCount.size(); c++)
				{
					double pL = LCount[c]/LTotal, pR = RCount[c]/RTotal;
					if (LCount[c] >= 1e-10) // FUCK YOU rounding errors
						LGini += pL * (1.0 - pL);
					if (RCount[c] >= 1e-10)
						RGini += pR * (1.0 - pR);
				}
			}
			else
			{
				for (int c = 0; c < LCount.size(); c++)
				{
					double pL = LCount[c]/LTotal, pR = RCount[c]/RTotal;
					if (LCount[c] >= 1e-10) // FUCK YOU rounding errors
						LGini -= pL * log(pL);
					if (RCount[c] >= 1e-10)
						RGini -= pR * log(pR);
				}
			}
			DGini = (LTotal*LGini + RTotal*RGini)/(LTotal + RTotal);

			if (DGini < bestDGini && LTotal > 0.0 && RTotal > 0.0)
			{
				bestDGini = DGini;
				bestThreshold = random_thresholds[th_idx];
				found = true;
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

	score_and_threshold.first = bestDGini;
	score_and_threshold.second = bestThreshold;
	return found;
}



template<typename Sample, typename TAppContext>
bool
SplitEvaluatorJointClassRegr<Sample, TAppContext>::CalculateOffsetCompactnessAndThreshold(DataSet<Sample, LabelJointClassRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold)
{
	// In: 		samples, sorted responses
	// Out: 	offset_measure + threshold
	// Multi-Class enabled

	// Initialize the counters
	double curr_variance, LTotal = 0.0, RTotal = 0.0, bestThreshold = 0.0, best_variance = 1e16;
	vector<vector<int> > LSamples(m_appcontext->num_classes), RSamples(m_appcontext->num_classes);
	bool found = false;

	// Calculate random thresholds and sort them
	// TODO: actually, we should here use the min max values of only the positive data!!!!
	double min_response = responses[0].first;
	double max_response = responses[responses.size()-1].first;
	double d = (max_response - min_response);
	vector<double> random_thresholds(m_appcontext->num_node_thresholds, 0.0);
	for (int i = 0; i < random_thresholds.size(); i++)
	{
		random_thresholds[i] = (randDouble() * d) + min_response;
	}
	sort(random_thresholds.begin(), random_thresholds.end());


	// First, put everything in the right node
	for (int r = 0; r < responses.size(); r++)
	{
		// if this sample is negative, skip it !
		if (!dataset[responses[r].second]->m_label.vote_allowed)
			continue;

		int lblIdx = dataset[responses[r].second]->m_label.class_label;
		RSamples[lblIdx].push_back(responses[r].second);
	}

	// Now, iterate all responses and calculate Gini indices at the cutoff points (thresholds)
	int th_idx = 0;
	bool stop_search = false;
	for (int r = 0; r < responses.size(); r++)
	{
		// if this sample is negative, skip it!
		if (!dataset[responses[r].second]->m_label.vote_allowed)
			continue;

		// if the current sample is smaller than the current threshold put it to the left side
		int lblIdx = dataset[responses[r].second]->m_label.class_label;
		if (responses[r].first <= random_thresholds[th_idx])
		{
			// Remove the current sample from the right side and put it on the left side ...
			LSamples[lblIdx].push_back(RSamples[lblIdx][0]);
			RSamples[lblIdx].erase(RSamples[lblIdx].begin());
		}
		else
		{
			// ok, now we found the first sample having higher response than the current threshold
			curr_variance = 0.0;
			double LTotal_class = 0.0, RTotal_class = 0.0;
			LTotal = 0.0, RTotal = 0.0;
			for (size_t c = 0; c < m_appcontext->num_classes; c++)
			{
				 // if we don't have any samples for this class, skip it
				if (LSamples[c].size() == 0 && RSamples[c].size() == 0)
					continue;

				curr_variance += EvaluateRegressionLoss(dataset, LSamples[c], RSamples[c], LTotal_class, RTotal_class, 0);
				LTotal += LTotal_class;
				RTotal += RTotal_class;
			}
			curr_variance /= static_cast<double>(m_appcontext->num_classes);
			if (curr_variance < best_variance && LTotal > 0.0 && RTotal > 0.0)
			{
				best_variance = curr_variance;
				bestThreshold = random_thresholds[th_idx];
				found = true;
			}

			// next, we have to find the next random threshold that is larger than the current response
			// -> there might be several threshold within the gap between the last response and this one.
			while (responses[r].first > random_thresholds[th_idx])
			{
				if (th_idx < (random_thresholds.size()-1))
				{
					th_idx++;
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

	score_and_threshold.first = best_variance;
	score_and_threshold.second = bestThreshold;
	return found;
}


template<typename Sample, typename TAppContext>
bool
SplitEvaluatorJointClassRegr<Sample, TAppContext>::CalculatePoseCompactnessAndThreshold(DataSet<Sample, LabelJointClassRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold)
{
	// In: 		samples, sorted responses
	// Out: 	offset_measure + threshold
	// Multi-Class enabled

	// Initialize the counters
	double curr_variance, LTotal = 0.0, RTotal = 0.0, bestThreshold = 0.0, best_variance = 1e16;
	vector<vector<int> > LSamples(m_appcontext->num_classes), RSamples(m_appcontext->num_classes);
	bool found = false;

	// Calculate random thresholds and sort them
	// TODO: actually, we should here use the min max values of only the positive data!!!!
	double min_response = responses[0].first;
	double max_response = responses[responses.size()-1].first;
	double d = (max_response - min_response);
	vector<double> random_thresholds(m_appcontext->num_node_thresholds, 0.0);
	for (int i = 0; i < random_thresholds.size(); i++)
	{
		random_thresholds[i] = (randDouble() * d) + min_response;
	}
	sort(random_thresholds.begin(), random_thresholds.end());


	// First, put everything in the right node
	for (int r = 0; r < responses.size(); r++)
	{
		// if this sample is negative, skip it !
		if (!dataset[responses[r].second]->m_label.vote_allowed)
			continue;

		int lblIdx = dataset[responses[r].second]->m_label.class_label;
		RSamples[lblIdx].push_back(responses[r].second);
	}

	// Now, iterate all responses and calculate Gini indices at the cutoff points (thresholds)
	int th_idx = 0;
	bool stop_search = false;
	for (int r = 0; r < responses.size(); r++)
	{
		// if this sample is negative, skip it!
		if (!dataset[responses[r].second]->m_label.vote_allowed)
			continue;

		// if the current sample is smaller than the current threshold put it to the left side
		int lblIdx = dataset[responses[r].second]->m_label.class_label;
		if (responses[r].first <= random_thresholds[th_idx])
		{
			// Remove the current sample from the right side and put it on the left side ...
			LSamples[lblIdx].push_back(RSamples[lblIdx][0]);
			RSamples[lblIdx].erase(RSamples[lblIdx].begin());
		}
		else
		{
			// ok, now we found the first sample having higher response than the current threshold
			curr_variance = 0.0;
			double LTotal_class = 0.0, RTotal_class = 0.0;
			LTotal = 0.0, RTotal = 0.0;
			for (size_t c = 0; c < m_appcontext->num_classes; c++)
			{
				 // if we don't have any samples for this class, skip it
				if (LSamples[c].size() == 0 && RSamples[c].size() == 0)
					continue;

				curr_variance += EvaluateRegressionLoss(dataset, LSamples[c], RSamples[c], LTotal_class, RTotal_class, 1);
				LTotal += LTotal_class;
				RTotal += RTotal_class;
			}
			curr_variance /= static_cast<double>(m_appcontext->num_classes);
			if (curr_variance < best_variance && LTotal > 0.0 && RTotal > 0.0)
			{
				best_variance = curr_variance;
				bestThreshold = random_thresholds[th_idx];
				found = true;
			}

			// next, we have to find the next random threshold that is larger than the current response
			// -> there might be several threshold within the gap between the last response and this one.
			while (responses[r].first > random_thresholds[th_idx])
			{
				if (th_idx < (random_thresholds.size()-1))
				{
					th_idx++;
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

	score_and_threshold.first = best_variance;
	score_and_threshold.second = bestThreshold;
	return found;
}



template<typename Sample, typename TAppContext>
bool
SplitEvaluatorJointClassRegr<Sample, TAppContext>::CalculateOffsetCompactnessAndThresholdOnline(DataSet<Sample, LabelJointClassRegr>& dataset, std::vector<std::pair<double, int> > responses, std::pair<double, double>& score_and_threshold)
{
	// INFO: this is only valid for the Reduction-In-Variance!
	// In: samples, sorted responses, out:offset_measure+threshold

	// Initialize the counters
	double curr_variance, LTotal = 0.0, RTotal = 0.0, bestThreshold = 0.0, best_variance = 1e16;
	vector<VectorXd> LMean_class(this->m_appcontext->num_classes, Eigen::VectorXd::Zero(this->m_appcontext->num_target_variables));
	vector<VectorXd> RMean_class(this->m_appcontext->num_classes, Eigen::VectorXd::Zero(this->m_appcontext->num_target_variables));
	vector<double> LVarSq(this->m_appcontext->num_classes, 0.0), RVarSq(this->m_appcontext->num_classes, 0.0);
	vector<double> LTotal_class(this->m_appcontext->num_classes), RTotal_class(this->m_appcontext->num_classes);
	bool found = false;

	// Calculate random thresholds and sort them
	double min_response = responses[0].first;
	double max_response = responses[responses.size()-1].first;
	double d = (max_response - min_response);
	vector<double> random_thresholds(m_appcontext->num_node_thresholds, 0.0);
	for (int i = 0; i < random_thresholds.size(); i++)
	{
		random_thresholds[i] = (randDouble() * d) + min_response;
	}
	sort(random_thresholds.begin(), random_thresholds.end());


	// First, put everything in the right node
	for (int r = 0; r < responses.size(); r++)
	{
		int lblIdx = dataset[responses[r].second]->m_label.class_label;

		// if this sample is negative, skip it !
		if (dataset[responses[r].second]->m_label.vote_allowed == false)
			continue;
		//RSamples[lblIdx].push_back(responses[r].second);

		// 4 online
		double csw = dataset[responses[r].second]->m_label.regr_weight;
		Eigen::VectorXd cst = dataset[responses[r].second]->m_label.regr_target;

		double temp = RTotal_class[lblIdx] + csw;
		//VectorXd delta = cst - RMean;
		VectorXd delta = cst - RMean_class[lblIdx];
		VectorXd R = delta * csw / temp;
		//RMean += R;
		RMean_class[lblIdx] += R;
		RVarSq[lblIdx] = RVarSq[lblIdx] + RTotal_class[lblIdx] * delta.dot(delta) * csw / temp;
		RTotal_class[lblIdx] = temp;
		//RVar = RVarSq/RTotal;
	}

	// Now, iterate all responses and calculate Gini indices at the cutoff points (thresholds)
	int th_idx = 0;
	bool stop_search = false;
	for (int r = 0; r < responses.size(); r++)
	{
		int lblIdx = dataset[responses[r].second]->m_label.class_label;

		// if this sample is negative, skip it !
		if (dataset[responses[r].second]->m_label.vote_allowed == false)
			continue;

		// if the current sample is smaller than the current threshold put it to the left side
		if (responses[r].first <= random_thresholds[th_idx])
		{
			// Remove the current sample from the right side and put it on the left side ...
			//LSamples[lblIdx].push_back(RSamples[lblIdx][0]);
			//RSamples[lblIdx].erase(RSamples[lblIdx].begin());

			// 4 online
			double csw = dataset[responses[r].second]->m_label.regr_weight;
			Eigen::VectorXd cst = dataset[responses[r].second]->m_label.regr_target;

			double temp = RTotal_class[lblIdx] - csw;
			//VectorXd delta = cst - RMean;
			VectorXd delta = cst - RMean_class[lblIdx];
			VectorXd R = delta * csw / temp;
			//RMean -= R;
			RMean_class[lblIdx] -= R;
			RVarSq[lblIdx] = RVarSq[lblIdx] - RTotal_class[lblIdx] * delta.dot(delta) * csw / temp;
			RTotal_class[lblIdx] = temp;
			//RVar = RVarSq/RTotal;

			temp = LTotal_class[lblIdx] + csw;
			//delta = cst - LMean;
			delta = cst - LMean_class[lblIdx];
			R = delta * csw / temp;
			//LMean += R;
			LMean_class[lblIdx] += R;
			LVarSq[lblIdx] = LVarSq[lblIdx] + LTotal_class[lblIdx] * delta.dot(delta) * csw / temp;
			LTotal_class[lblIdx] = temp;
			//LVar = LVarSq/LTotal;
		}
		else
		{
			// ok, now we found the first sample having higher response than the current threshold
			//double curr_variance_old = EvaluateRegressionLoss(dataset, LSamples, RSamples, LTotal, RTotal);
			//curr_variance = LTotal / (LTotal+RTotal) * LVar + RTotal / (LTotal+RTotal) * RVar;
			// as we see from this formula: LTotal/xx * LVarUnnorm/LTotal + ...
			// we can drop the LTotal normalization for the LVar & RVar
			curr_variance = 0.0;
			LTotal = 0.0, RTotal = 0.0;
			for (int c = 0; c < m_appcontext->num_classes; c++)
			{
				double var_class = (LVarSq[c] + RVarSq[c]) / (LTotal_class[c] + RTotal_class[c]);
				if (LTotal_class[c] == 0.0)
					var_class = RVarSq[c] / RTotal_class[c];
				if (RTotal_class[c] == 0.0)
					var_class = LVarSq[c] / LTotal_class[c];
				if (LTotal_class[c] == 0.0 && RTotal_class[c] == 0.0)
					var_class = 0.0;
				curr_variance += var_class;
				LTotal += LTotal_class[c];
				RTotal += RTotal_class[c];
			}
			curr_variance /= (double)m_appcontext->num_classes;

			if (curr_variance < best_variance && LTotal > 0.0 && RTotal > 0.0)
			{
				best_variance = curr_variance;
				bestThreshold = random_thresholds[th_idx];
				found = true;
			}

			// next, we have to find the next random threshold that is larger than the current response
			// -> there might be several threshold within the gap between the last response and this one.
			while (responses[r].first > random_thresholds[th_idx])
			{
				if (th_idx < (random_thresholds.size()-1))
				{
					th_idx++;
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

	score_and_threshold.first = best_variance;
	score_and_threshold.second = bestThreshold;
	return found;
}


template<typename Sample, typename TAppContext>
double
SplitEvaluatorJointClassRegr<Sample, TAppContext>::EvaluateRegressionLoss(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> LSamples, vector<int> RSamples, double& LTotal, double& RTotal, int flag_pose)
{
    switch (m_appcontext->splitevaluation_type_regression)
    {
    case SPLITEVALUATION_TYPE_REGRESSION::REDUCTION_IN_VARIANCE:
        return EvaluateRegressionLoss_ReductionInVariance(dataset, LSamples, RSamples, LTotal, RTotal);
        break;
    case SPLITEVALUATION_TYPE_REGRESSION::DIFF_ENTROPY_GAUSS:
	if(flag_pose == 0){
        	return EvaluateRegressionLoss_DiffEntropyGauss(dataset, LSamples, RSamples, LTotal, RTotal);
	}else{
		return EvaluateRegressionLoss_DiffEntropyGaussPose(dataset, LSamples, RSamples, LTotal, RTotal);
	}
        break;
    case SPLITEVALUATION_TYPE_REGRESSION::DIFF_ENTROPY_GAUSS_BLOCK_POSE:
        return EvaluateRegressionLoss_DiffEntropyGaussBlockPoseEstimation(dataset, LSamples, RSamples, LTotal, RTotal);
        break;
    default:
    	throw std::runtime_error("SplitEvaluator (evaluate regression loss): unknown splitfunciton-regression type!");
    	return 0.0;
        break;
    }
}


template<typename Sample, typename TAppContext>
double
SplitEvaluatorJointClassRegr<Sample, TAppContext>::EvaluateRegressionLoss_ReductionInVariance(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> LSamples, vector<int> RSamples, double& LTotal, double& RTotal)
{
	// Calculate variances in left and right child nodes
	double LVar = 0.0, RVar = 0.0;
    LVar = CalculateVariance(dataset, LSamples, true, LTotal);
    RVar = CalculateVariance(dataset, RSamples, true, RTotal);

    // 3) Calculate loss
    double total_var = LTotal / (LTotal+RTotal) * LVar + RTotal / (LTotal+RTotal) * RVar;
    return total_var;
}


template<typename Sample, typename TAppContext>
double
SplitEvaluatorJointClassRegr<Sample, TAppContext>::EvaluateRegressionLoss_DiffEntropyGauss(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> LSamples, vector<int> RSamples, double& LTotal, double& RTotal)
{
	// 1) Calculate co-variance matrix of the offset vectors for both sets!
	MatrixXd LCov = CalculateCovariance(dataset, LSamples, true, LTotal);
	MatrixXd RCov = CalculateCovariance(dataset, RSamples, true, RTotal);

	//double LEntropyEstimate = 2.0/2.0 - 2.0/2.0 * log(2.0*PI) + 1.0/2.0*log(LCov.determinant());
	double LCovDet = LCov.determinant();
	if (LCovDet <= 0.0)
		LCovDet = 1e-10;
	double RCovDet = RCov.determinant();
	if (RCovDet <= 0.0)
		RCovDet = 1e-10;
	double LEntropyEstimate = log(LCovDet);
	double REntropyEstimate = log(RCovDet);

	// 2) Calculate information gain
	double infogain = (LTotal * LEntropyEstimate + RTotal * REntropyEstimate) / (LTotal+RTotal);
	return infogain;
}

template<typename Sample, typename TAppContext>
double
SplitEvaluatorJointClassRegr<Sample, TAppContext>::EvaluateRegressionLoss_DiffEntropyGaussPose(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> LSamples, vector<int> RSamples, double& LTotal, double& RTotal)
{
	// 1) Calculate co-variance matrix of the offset vectors for both sets!
	MatrixXd LCov = CalculateCovariancePose(dataset, LSamples, true, LTotal);
	MatrixXd RCov = CalculateCovariancePose(dataset, RSamples, true, RTotal);

	//double LEntropyEstimate = 2.0/2.0 - 2.0/2.0 * log(2.0*PI) + 1.0/2.0*log(LCov.determinant());
	double LCovDet = LCov.determinant();
	if (LCovDet <= 0.0)
		LCovDet = 1e-10;
	double RCovDet = RCov.determinant();
	if (RCovDet <= 0.0)
		RCovDet = 1e-10;
	double LEntropyEstimate = log(LCovDet);
	double REntropyEstimate = log(RCovDet);

	// 2) Calculate information gain
	double infogain = (LTotal * LEntropyEstimate + RTotal * REntropyEstimate) / (LTotal+RTotal);
	return infogain;
}


template<typename Sample, typename TAppContext>
double
SplitEvaluatorJointClassRegr<Sample, TAppContext>::EvaluateRegressionLoss_DiffEntropyGaussBlockPoseEstimation(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> LSamples, vector<int> RSamples, double& LTotal, double& RTotal)
{
	// PROBLEM: This is not a generic method, it assumes that num_target_variables = 6 !!!

	// 1) Calculate the co-variance matrices
	MatrixXd LCov = CalculateCovariance(dataset, LSamples, true, LTotal);
	MatrixXd RCov = CalculateCovariance(dataset, RSamples, true, RTotal);

	// 2) Set the 3-block off-diagonals to zero
	MatrixXd LCovPosition = MatrixXd::Zero(3, 3);
	MatrixXd RCovPosition = MatrixXd::Zero(3, 3);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			LCovPosition(i, j) = LCov(i, j);
			RCovPosition(i, j) = RCov(i, j);
		}
	}
	MatrixXd LCovAngle = MatrixXd::Zero(3, 3);
	MatrixXd RCovAngle = MatrixXd::Zero(3, 3);
	for (int i = 3; i < 6; i++)
	{
		for (int j = 3; j < 6; j++)
		{
			LCovAngle(i-3, j-3) = LCov(i, j);
			RCovAngle(i-3, j-3) = RCov(i, j);
		}
	}


	// 3) Calculate the entropy measures
	double LCovDet = LCovPosition.determinant() + LCovAngle.determinant();
	if (LCovDet <= 0.0)
		LCovDet = 1e-10;
	double RCovDet = RCovPosition.determinant() + RCovAngle.determinant();
	if (RCovDet <= 0.0)
		RCovDet = 1e-10;
	double LEntropyEstimate = log(LCovDet);
	double REntropyEstimate = log(RCovDet);

	// 4) Calculate information gain
	double infogain = (LTotal * LEntropyEstimate + RTotal * REntropyEstimate) / (LTotal+RTotal);
	return infogain;
}



// Helper statistic functions
template<typename Sample, typename TAppContext>
VectorXd
SplitEvaluatorJointClassRegr<Sample, TAppContext>::CalculateMean(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> sample_ids, bool weighted, double& sum_weight)
{
	VectorXd mode = VectorXd::Zero(m_appcontext->num_target_variables);
	sum_weight = 0.0;
	for (unsigned int i = 0; i < sample_ids.size(); i++)
	{
		double csw = dataset[sample_ids[i]]->m_label.regr_weight;
		Eigen::VectorXd cst = dataset[sample_ids[i]]->m_label.regr_target;
		mode += csw * cst;
		sum_weight += csw;
	}
	if (sum_weight > 0.0)
		mode /= sum_weight;
	return mode;
}

// Helper statistic functions
template<typename Sample, typename TAppContext>
VectorXd
SplitEvaluatorJointClassRegr<Sample, TAppContext>::CalculateMeanPose(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> sample_ids, bool weighted, double& sum_weight)
{
	double angle = 0;
	double x = 0.0;
	double y = 0.0;
  	double cPI = 3.14159265358979323846;
	double meanA=0;

	VectorXd mode = VectorXd::Zero(1);
	sum_weight = 0.0;
	for (unsigned int i = 0; i < sample_ids.size(); i++)
	{
		double csw = dataset[sample_ids[i]]->m_label.regr_weight;
		angle = double(dataset[sample_ids[i]]->m_label.azimuth)*cPI/180.0;
		x += cos(angle);
    		y += sin(angle);
		
		sum_weight += csw;
	}
	x /= float(sum_weight);
	y /= float(sum_weight);
	meanA = atan2 (y,x)*180/cPI;
  	meanA = (meanA < 0)? (360+meanA) : meanA;
	
	if (sum_weight > 0.0)
		mode(0) = meanA;
	return mode;
}

template<typename Sample, typename TAppContext>
MatrixXd
SplitEvaluatorJointClassRegr<Sample, TAppContext>::CalculateCovariance(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> sample_ids, bool weighted, double& total_weight)
{
	VectorXd Mean = CalculateMean(dataset, sample_ids, weighted, total_weight);

	MatrixXd cov = MatrixXd::Zero(m_appcontext->num_target_variables, m_appcontext->num_target_variables);
	VectorXd cst = VectorXd::Zero(m_appcontext->num_target_variables);
	double SumSqWeight = 0.0;
	for (int i = 0; i < sample_ids.size(); i++)
	{
		double csw = dataset[sample_ids[i]]->m_label.regr_weight;
		cst = dataset[sample_ids[i]]->m_label.regr_target;
		cov += csw * ((cst-Mean) * (cst-Mean).transpose());
		SumSqWeight += pow(csw/total_weight, 2);
	}
	if (total_weight > 0.0)
	{
		cov /= total_weight; // normalize, such that sum_i w_i = 1 holds !!!
		if (SumSqWeight < 1.0) // this happens if only one sample is available!!!
			cov /= (1.0 - SumSqWeight);
	}

	// return a regularized cov matrix
	return cov + 0.05*cov(0, 0) * MatrixXd::Identity(m_appcontext->num_target_variables, m_appcontext->num_target_variables);
}

template<typename Sample, typename TAppContext>
MatrixXd
SplitEvaluatorJointClassRegr<Sample, TAppContext>::CalculateCovariancePose(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> sample_ids, bool weighted, double& total_weight)
{
	VectorXd Mean = CalculateMeanPose(dataset, sample_ids, weighted, total_weight);

	MatrixXd cov = MatrixXd::Zero(1, 1);
	VectorXd cst = VectorXd::Zero(1);
	double SumSqWeight = 0.0;
	for (int i = 0; i < sample_ids.size(); i++)
	{
		double csw = dataset[sample_ids[i]]->m_label.regr_weight;
		cst(0) = dataset[sample_ids[i]]->m_label.azimuth;
		double tmp1 = abs(cst(0)-Mean(0));
    		double tmp2 = 360 - tmp1;
   		double tempNorm = min(tmp1,tmp2);
		cov(0,0) += csw * (tempNorm * tempNorm);
		SumSqWeight += pow(csw/total_weight, 2);
	}
	if (total_weight > 0.0)
	{
		cov /= total_weight; // normalize, such that sum_i w_i = 1 holds !!!
		if (SumSqWeight < 1.0) // this happens if only one sample is available!!!
			cov /= (1.0 - SumSqWeight);
	}

	// return a regularized cov matrix
	return cov + 0.05*cov(0, 0) * MatrixXd::Identity(1, 1);
}

template<typename Sample, typename TAppContext>
double
SplitEvaluatorJointClassRegr<Sample, TAppContext>::CalculateVariance(DataSet<Sample, LabelJointClassRegr>& dataset, vector<int> sample_ids, bool weighted, double& total_weight)
{
	// 1) Calculate Mean
	VectorXd mean = CalculateMean(dataset, sample_ids, weighted, total_weight);

	// 2) Calculate Variance
	VectorXd cst = VectorXd::Zero(m_appcontext->num_target_variables);
	double squared_dist = 0.0, var = 0.0;
	for (unsigned int i = 0; i < sample_ids.size(); i++)
	{
		// and calculate the squared distance to the mode
		cst = dataset[sample_ids[i]]->m_label.regr_target;
		double csw = dataset[sample_ids[i]]->m_label.regr_weight;
		squared_dist = (cst - mean).dot(cst - mean);
		var += csw * squared_dist;
	}
	if (total_weight > 0.0)
		var /= total_weight;
	return var;
}



#endif /* SPLITEVALUATORJOINTCLASSREGR_CPP_ */
