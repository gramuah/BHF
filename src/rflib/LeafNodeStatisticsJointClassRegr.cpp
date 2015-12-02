
#ifndef LEAFNODESTATISTICSJOINTCLASSREGR_CPP_
#define LEAFNODESTATISTICSJOINTCLASSREGR_CPP_


#include "LeafNodeStatisticsJointClassRegr.h"


template<typename TAppContext>
LeafNodeStatisticsJointClassRegr<TAppContext>::LeafNodeStatisticsJointClassRegr(TAppContext* appcontextin) : m_appcontext(appcontextin)
{
	this->m_num_samples = 0;
	this->m_num_samples_class.resize(m_appcontext->num_classes, 0.0);
	this->m_num_samples_latent.resize(m_appcontext->num_z, 0);
	this->m_class_histogram.clear();
	this->m_class_histogram.resize(m_appcontext->num_classes, 0.0);
	this->m_votes.resize(m_appcontext->num_classes);
	this->m_vote_weights.resize(m_appcontext->num_classes);
}

template<typename TAppContext>
LeafNodeStatisticsJointClassRegr<TAppContext>::~LeafNodeStatisticsJointClassRegr() { }


template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::Aggregate(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, int is_final_leaf)
{
	bool do_class_balancing = true;

	// classification statistics
	this->m_class_histogram.clear();
	this->m_class_histogram.resize(m_appcontext->num_classes, 0.0);
	this->m_num_samples = (int)dataset.size();
	this->m_num_samples_class.clear();
	this->m_num_samples_latent.clear();
	this->m_num_samples_class.resize(m_appcontext->num_classes, 0);
	this->m_num_samples_latent.resize(m_appcontext->num_z, 0);
	if (do_class_balancing)
		this->m_total_samples_weight = 0;
	for (size_t s = 0; s < dataset.size(); s++)
	{
		this->m_num_samples_class[dataset[s]->m_label.class_label]++;
		if (dataset[s]->m_label.class_label != 0){	
			this->m_num_samples_latent[dataset[s]->m_label.latent_prediction - 1]++;}

		if (do_class_balancing) // including class balancing
			m_class_histogram[dataset[s]->m_label.class_label] += dataset[s]->m_label.class_weight_gt;
		else // no class balancing
			m_class_histogram[dataset[s]->m_label.class_label] += (1.0 / (double)this->m_num_samples);

		if (do_class_balancing)
			this->m_total_samples_weight += dataset[s]->m_label.class_weight_gt;
	}
	if (do_class_balancing) // for class-balanced leafnodes
		for (size_t c = 0; c < m_class_histogram.size(); c++)
			m_class_histogram[c] /= this->m_total_samples_weight;



	// regression targets
	// INFO:
	// 1) full is obsolete, it will be neglected. We always calcualte the full predictions
	//    in all intermediate leafnodes!!!!
	// 2) Thus, if it is a final leaf node, do not re-calculate the leaf nodes statistics!!!
	//    Reason: as we always have calculated the predictions for all intermediate leaf nodes,
	//         there is no need for doing this recalculation, as it is already done, we just
	//         have to switch the state from Intermediate to Final.
	//         There is also a second, more important reason: If we would recalculate the
	//         prediction at this point, ARFs won't work anymore, because we would calculate
	//         the predictions with the pseudo targets and do not add the parent anymore!
	// TODO: actually we don't need to re-calculate the histograms above!
	// INFO: BUT we have to call this method because derived classes could do something only for
	//       the final leaf nodes (e.g., compute the variance of the voting regression targets,
	//       see LeafNodeStatisticsHeadpose.cpp)
	if (!is_final_leaf)
		this->AggregateRegressionTargets(dataset);
}


template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::Aggregate(LeafNodeStatisticsJointClassRegr* leafstatsin)
{
	bool do_class_balancing = true;

	// update class histogram
	for (size_t c = 0; c < this->m_class_histogram.size(); c++)
	{
		if (do_class_balancing)
		{
			// WITH class balancing
			// de-normalize this leafs class-histogram
			this->m_class_histogram[c] = this->m_class_histogram[c] * this->m_total_samples_weight;
			// add the de-normalized new class-histogram
			this->m_class_histogram[c] += leafstatsin->m_class_histogram[c] * (double)leafstatsin->m_total_samples_weight;
			// normalize the the histogram
			this->m_class_histogram[c] /= (this->m_total_samples_weight + leafstatsin->m_total_samples_weight);

		}
		else
		{
			// WITHOUT class balancing
			// de-normalize this leafs class-histogram
			this->m_class_histogram[c] = this->m_class_histogram[c] * (double)this->m_num_samples;
			// add the de-normalized new class-histogram
			this->m_class_histogram[c] += leafstatsin->m_class_histogram[c] * (double)leafstatsin->m_num_samples;
			// normalize the the histogram
			this->m_class_histogram[c] /= (double)(this->m_num_samples + leafstatsin->m_num_samples);
		}

	}
	// update the number of samples in the new leafstats
	this->m_num_samples += leafstatsin->m_num_samples;

	if (do_class_balancing)// for class-balanced leafnodes
		this->m_total_samples_weight += leafstatsin->m_total_samples_weight;

	// update votes
	for (size_t c = 0; c < this->m_votes.size(); c++) // loop over classes
	{
		int num_votes_old;
		size_t num_additional_votes;
		switch (m_appcontext->leafnode_regression_type)
		{
		case LEAFNODE_REGRESSION_TYPE::ALL:
			num_votes_old = this->m_votes[c].size();
			num_additional_votes = leafstatsin->m_votes[c].size();
			this->m_votes[c].resize(num_votes_old + num_additional_votes);
			this->m_vote_weights[c].resize(num_votes_old + num_additional_votes);
			for (size_t i = 0; i < leafstatsin->m_votes[c].size(); i++)
			{
				this->m_votes[c][num_votes_old + i] = leafstatsin->m_votes[c][i];
				this->m_vote_weights[c][num_votes_old + i] = leafstatsin->m_vote_weights[c][i];
			}
			break;
		case LEAFNODE_REGRESSION_TYPE::HILLCLIMB:
		case LEAFNODE_REGRESSION_TYPE::MEAN:
			throw std::runtime_error("Aggregate cannot handle hillclimb or mean yet");
			break;
		default:
			throw std::runtime_error("Aggregate cannot handle this leafnode regression type");
			break;
		}
	}
}


template<typename TAppContext>
LeafNodeStatisticsJointClassRegr<TAppContext> LeafNodeStatisticsJointClassRegr<TAppContext>::Average(std::vector<LeafNodeStatisticsJointClassRegr*> leafstats, LabelledSample<SampleImgPatch, LabelJointClassRegr>* sample, int d, Eigen::VectorXd mean, Eigen::VectorXd std, std::vector<cv::Mat> hough_map,TAppContext* apphp)
{


	// class histogram
	LeafNodeStatisticsJointClassRegr<TAppContext> ret_stats(apphp); // already initialized (class-hist is initialized to a zero vector)
	for (size_t i = 0; i < leafstats.size(); i++)
	{
		for (size_t c = 0; c < leafstats[i]->m_class_histogram.size(); c++)
			ret_stats.m_class_histogram[c] += leafstats[i]->m_class_histogram[c];
	}
	for (size_t c = 0; c < ret_stats.m_class_histogram.size(); c++)
		ret_stats.m_class_histogram[c] /= (double)leafstats.size();

	// voting elements (only necessary for ARFs!)
	//if ((apphp->method == RF_METHOD::ADF || apphp->method == RF_METHOD::ADHOUGHFOREST)
	//		&& apphp->do_regression_weight_updates)
	if (apphp->leafnode_regression_type == LEAFNODE_REGRESSION_TYPE::MEAN)
	{
		ret_stats.m_votes.resize(apphp->num_classes);
		ret_stats.m_prediction_centers.resize(apphp->num_classes);
		ret_stats.m_prediction.resize(apphp->num_classes);
		ret_stats.m_hough_img_prediction.resize(apphp->num_classes);
		ret_stats.m_vote_weights.resize(apphp->num_classes);
		for (size_t c = 0; c < apphp->num_classes; c++)
		{
			int accumOffsets=0;
			bool found_voting_vector = false;
			ret_stats.m_prediction[c] = MatrixXd::Zero(apphp->num_z,apphp->num_target_variables);
			ret_stats.m_hough_img_prediction[c] = MatrixXd::Zero(apphp->num_z,apphp->num_target_variables);
			Eigen::VectorXd tmp_vote = Eigen::VectorXd::Zero(apphp->num_target_variables); // only for hillclimb and mean
			double tmp_sum = 0.0;
			ret_stats.m_prediction_centers[c].resize(apphp->num_z);
			for (size_t i = 0; i < leafstats.size(); i++) // loop over trees!
			{
				if (c > 0 && sample->m_label.class_label > 0){// only foreground
					float w = 1.0 / float(leafstats.size()*leafstats[i]->m_offsets[1].size());
					accumOffsets += (int)leafstats[i]->m_offsets[1].size();
					for(size_t of=0; of < (int)leafstats[i]->m_offsets[1].size(); of++){
						
						int vote_x =0;
						int vote_y =0;

						//calculate the offsets from the residuals
						if (d == 0){// root nodes
							double regr_target_aux_x = leafstats[i]->m_regr_target[1][of](0);
							double regr_target_aux_y = leafstats[i]->m_regr_target[1][of](1);
						
							double aux_x = regr_target_aux_x * std(0);
							double aux_y = regr_target_aux_y * std(1);
							aux_x += mean(0);
							aux_y += mean(1);
							vote_x = int(sample->m_label.regr_patch_center_gt(0) + aux_x);
							vote_y = int(sample->m_label.regr_patch_center_gt(1) + aux_y);
						}else{
							//denormalization estimated residual				
							double regr_target_aux_x = leafstats[i]->m_regr_target[1][of](0);
							double regr_target_aux_y = leafstats[i]->m_regr_target[1][of](1);
						
							double aux_x = regr_target_aux_x * std(0);
							double aux_y = regr_target_aux_y * std(1);
							aux_x += mean(0);
							aux_y += mean(1);
							
							//estimated offset
							int off_x = int(leafstats[i]->m_offsets[1][of](0) - aux_x);
							int off_y = int(leafstats[i]->m_offsets[1][of](1) -aux_y);						

							vote_x = int(sample->m_label.regr_patch_center_gt(0) + off_x);
							vote_y = int(sample->m_label.regr_patch_center_gt(1) + off_y);
						}
						if (vote_y >= 0 && vote_y < hough_map[0].rows && vote_x >= 0 && vote_x < hough_map[0].cols)
						{
							int aux = leafstats[i]->m_latent_prediction[1][of]-1;
							hough_map[aux].at<float>(vote_y, vote_x) += (float)(w * leafstats[i]->m_vote_weights[1][0]);
							
						}
						
					}
				}
				
				int max_num_votes;
				int current_size;
				switch (apphp->leafnode_regression_type)
				{
				case LEAFNODE_REGRESSION_TYPE::ALL:
					throw std::logic_error("ARF + leafnode_regression_type == ALL makes no sense as a single prediction is required. Could be extended to collect all votes and then make e.g. a meanshift. But the code has to be adapted!!!");
					// The following code was not tested yet!
					// And a post processing is also missing, after collecting all the votes over the trees,
					// for ARF, a single predicion is required, -> meanshift, mean, hillclimb afterwards!
					max_num_votes = min(int(300 / leafstats.size()), (int)leafstats[i]->m_votes[c].size());
					current_size = ret_stats.m_votes[c].size();
					ret_stats.m_votes[c].resize(current_size + max_num_votes);
					ret_stats.m_vote_weights[c].resize(current_size + max_num_votes);
					for (size_t v = 0; v < max_num_votes; v++)
					{
						ret_stats.m_votes[c][current_size + v] = leafstats[i]->m_votes[c][v];
						ret_stats.m_vote_weights[c][current_size + v] = leafstats[i]->m_vote_weights[c][v];
					}
					break;
				// here, we simply average them ...
				case LEAFNODE_REGRESSION_TYPE::HILLCLIMB:
				case LEAFNODE_REGRESSION_TYPE::MEAN:
					if (leafstats[i]->m_votes[c].size() > 0)
					{
						//tmp_vote += leafstats[i]->m_votes[c][0] * leafstats[i]->m_vote_weights[c][0];
						// INFO: we should not mix the classification and regression tasks here
						//tmp_vote += leafstats[i]->m_votes[c][0];
						if (c == 1)
							tmp_vote += leafstats[i]->m_intermediate_prediction;
						else
							tmp_vote += leafstats[i]->m_votes[c][0];

						for(int zz=0; zz < apphp->num_z; zz++){
							for(int v=0; v < apphp->num_target_variables; v++){
								ret_stats.m_prediction[c](zz,v) += leafstats[i]->m_prediction[c](zz,v);
								
							}
						}
						
						tmp_sum += leafstats[i]->m_vote_weights[c][0];
						found_voting_vector = true;
					}
					break;
				case LEAFNODE_REGRESSION_TYPE::MEANSHIFT:
					break;
				default:
					throw std::runtime_error("LeafNodeStatistics (average): this leafnode-regression-type is not implemented!");
				}
			}
			if(c > 0 && sample->m_label.class_label > 0){
				// Find current max value + location in hough space
		        	cv::Point max_loc_tmp;
		        	cv::Point min_loc_tmp;
		        	double min_val_tmp;
		        	double max_val_tmp;
		        
		        	for (size_t zz = 0; zz < hough_map.size(); zz++)
        			{

					hough_map[zz] += 0.1*sample->m_label.hough_map_patch[zz];//addTarget (propagate the hough space) 
 					sample->m_label.hough_map_patch[zz] = hough_map[zz];//update the hough space
					Eigen::VectorXd centers = Eigen::VectorXd::Zero(2);
        					
					cv::minMaxLoc(hough_map[zz], &min_val_tmp, &max_val_tmp, &min_loc_tmp, &max_loc_tmp);
					centers(0) = (double)max_loc_tmp.x;
					centers(1) = (double)max_loc_tmp.y;
								
					//center prediction
					ret_stats.m_hough_img_prediction[c](zz,0) = (double)centers(0);
					ret_stats.m_hough_img_prediction[c](zz,1) = (double)centers(1);


            	
        			}
			}


			// normalization for target votes
			if (apphp->leafnode_regression_type == LEAFNODE_REGRESSION_TYPE::HILLCLIMB || apphp->leafnode_regression_type == LEAFNODE_REGRESSION_TYPE::MEAN)
			{
				if (found_voting_vector)
				{
					ret_stats.m_votes[c].resize(1);
					ret_stats.m_votes[c][0] = tmp_vote / (double)leafstats.size();
					ret_stats.m_vote_weights[c].resize(1, tmp_sum / (double)leafstats.size());
					for(int zz=0; zz < apphp->num_z; zz++){
						for(int v=0; v < apphp->num_target_variables; v++){
							ret_stats.m_prediction[c](zz,v) /= (double)leafstats.size();
						
						}
					}
				}
			}
		}
	}

	return ret_stats;
}


template<typename TAppContext>
LeafNodeStatisticsJointClassRegr<TAppContext> LeafNodeStatisticsJointClassRegr<TAppContext>::Sum(std::vector<LeafNodeStatisticsJointClassRegr*> leafstats, TAppContext* apphp)
{
	// add up the class histograms
	LeafNodeStatisticsJointClassRegr<TAppContext> ret_stats(apphp); // already initialized
	for (size_t i = 0; i < leafstats.size(); i++)
	{
		for (size_t c = 0; c < leafstats[i]->m_class_histogram.size(); c++)
			ret_stats.m_class_histogram[c] += apphp->shrinkage * leafstats[i]->m_class_histogram[c];
	}

	// voting elements (only necessary for ARFs!)
	Eigen::VectorXd tmp_vote = Eigen::VectorXd::Zero(apphp->num_target_variables);
	ret_stats.m_votes.resize(apphp->num_classes);
	ret_stats.m_vote_weights.resize(apphp->num_classes);
	for (size_t c = 0; c < apphp->num_classes; c++)
	{
		for (size_t i = 0; i < leafstats.size(); i++) // loop over trees!
		{
			int max_num_votes;
			switch (apphp->leafnode_regression_type)
			{
			case LEAFNODE_REGRESSION_TYPE::ALL:
			case LEAFNODE_REGRESSION_TYPE::MEANSHIFT:
				throw std::logic_error("LeafNodeStatistics (sum): only works for hillclimb and mean!");
				break;
			// here, we simply average them ...
			case LEAFNODE_REGRESSION_TYPE::HILLCLIMB:
			case LEAFNODE_REGRESSION_TYPE::MEAN:
				if (leafstats[i]->m_votes.size() > 0)
				{
					tmp_vote += leafstats[i]->m_votes[c][0] * leafstats[i]->m_vote_weights[c][0];
				}
				break;
			default:
				throw std::runtime_error("LeafNodeStatistics (sum): this leafnode-regression-type is not implemented!");
			}
		}

		ret_stats.m_votes[c].resize(1);
		ret_stats.m_votes[c][0] = tmp_vote;
		ret_stats.m_vote_weights[c].resize(1, 1.0);
	}

	return ret_stats;
}



template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::UpdateStatistics(LabelledSample<SampleImgPatch, LabelJointClassRegr>* labelled_sample)
{
	bool do_class_balancing = true;

	// label of the sample
	int labelIdx = labelled_sample->m_label.class_label;

	// if no samples have been routed to this leaf, "initialize" it ...
	// this should never happen, except for the online case!
	if (this->m_num_samples == 0)
	{
		// use this single sample for the histogram
		for (int c = 0; c < this->m_class_histogram.size(); c++)
			this->m_class_histogram[c] = 0.0;

		if (do_class_balancing)
			this->m_class_histogram[labelIdx] = labelled_sample->m_label.class_weight_gt;
		else
			this->m_class_histogram[labelIdx] = 1.0;

		// increase the total number of samples
		m_num_samples++;
		// increase the class-specific counter
		m_num_samples_class[labelIdx]++;
		// increase the total weight
		m_total_samples_weight = labelled_sample->m_label.class_weight_gt;


		// update the votes
		if (labelled_sample->m_label.vote_allowed == true)
		{
			this->m_votes.resize(m_appcontext->num_classes);
			this->m_vote_weights.resize(m_appcontext->num_classes);

			this->m_votes[labelIdx].push_back(labelled_sample->m_label.regr_target);
			this->m_vote_weights[labelIdx].resize(1, 1.0);
		}

		// and RETURN!
		return;
	}


	// Add the statics of a new sample to the histogram ...
	// denormalize the histogram
	for (size_t l = 0; l < m_class_histogram.size(); l++)
	{
		if (do_class_balancing)
			m_class_histogram[l] *= this->m_total_samples_weight;
		else
			m_class_histogram[l] *= (double)m_num_samples;
	}

	// add the sample to the corresponding bin
	if (do_class_balancing)
		m_class_histogram[labelIdx] += labelled_sample->m_label.class_weight_gt;
	else
		m_class_histogram[labelIdx] += 1.0;

	// increase the total number of samples
	m_num_samples++;
	// increase the class-specific counter
	m_num_samples_class[labelIdx]++;
	// increase the total weight
	m_total_samples_weight += labelled_sample->m_label.class_weight_gt;

	// normalize the histogram again
	for (size_t l = 0; l < m_class_histogram.size(); l++)
	{
		if (do_class_balancing)
			m_class_histogram[l] /= this->m_total_samples_weight;
		else
			m_class_histogram[l] /= (double)m_num_samples;
	}



	// update the regression targets
	if (labelled_sample->m_label.vote_allowed == true)
	{
		//std::vector<double> new_regr_target;
		Eigen::VectorXd new_regr_target;
		switch (this->m_appcontext->leafnode_regression_type)
		{
		case LEAFNODE_REGRESSION_TYPE::ALL:
			// simply add the regression target to the list
			//new_regr_target.resize(labelled_sample->m_label.regr_target.rows(), 0);
			//for (size_t j = 0; j < new_regr_target.size(); j++)
			//	new_regr_target[j] = labelled_sample->m_label.regr_target(j);
			//this->m_votes.push_back(new_regr_target);
			this->m_votes[labelIdx].push_back(labelled_sample->m_label.regr_target);
			// voting weights (the sum of the offset weights is the foreground probability)
			//this->m_vote_weights.clear();
			//this->m_vote_weights.resize(this->m_votes.size(), this->m_class_histogram[1] / (double)this->m_votes.size());
			this->m_vote_weights[labelIdx].clear();
			this->m_vote_weights[labelIdx].resize(this->m_votes[labelIdx].size(), this->m_class_histogram[labelIdx] / (double)this->m_votes[labelIdx].size());
			break;
		case LEAFNODE_REGRESSION_TYPE::MEAN:
			// denormalize the mean
			if (this->m_num_samples_class[labelIdx] == 1) // i.e., this is the first positive samples falling in that node
			{
				this->m_votes[labelIdx].push_back(Eigen::VectorXd::Zero(labelled_sample->m_label.regr_target.rows()));
				this->m_vote_weights[labelIdx].resize(1, this->m_class_histogram[labelIdx]);
			}
			this->m_votes[labelIdx][0] *= (double)this->m_num_samples_class[labelIdx] - 1.0;
			this->m_votes[labelIdx][0] += labelled_sample->m_label.regr_target;
			this->m_votes[labelIdx][0] /= (double)this->m_num_samples_class[labelIdx];
			// voting weight is the class probability
			this->m_vote_weights[labelIdx][0] = this->m_class_histogram[labelIdx];
			break;
		case LEAFNODE_REGRESSION_TYPE::MEANSHIFT:
		case LEAFNODE_REGRESSION_TYPE::HILLCLIMB:
			throw std::logic_error("LeafNodeStatistics (sum): only works for hillclimb and mean!");
			break;
		default:
			throw std::runtime_error("LeafNodeStatistics (sum): this leafnode-regression-type is not implemented!");
		}
	}
}


template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::DenormalizeTargetVariables(Eigen::VectorXd mean, Eigen::VectorXd std)
{
	// TODO: this stuff is useless here ... we don't have class-specific mean or std values!!!!
	// Find a better solution. We should only use the method with the std::vectors!
	for (size_t c = 0; c < this->m_appcontext->num_classes; c++)
	{
		for (size_t i = 0; i < this->m_votes[c].size(); i++)
		{
			for (size_t v = 0; v < this->m_votes[c][i].size(); v++)
			{
				this->m_votes[c][i](v) = (this->m_votes[c][i](v) * std(v));
				this->m_votes[c][i](v) += mean(v);
			}
		}
	}
}

template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::DenormalizeTargetVariables(std::vector<Eigen::VectorXd> mean, std::vector<Eigen::VectorXd> std)
{
	for (size_t c = 0; c < this->m_appcontext->num_classes; c++)
	{
		for (size_t i = 0; i < this->m_votes[c].size(); i++)
		{
			this->m_votes[c][i] = (this->m_votes[c][i].array() * std[c].array()).matrix();
			this->m_votes[c][i] += mean[c];
		}

		for(int zz=0; zz<m_appcontext->num_z; zz++){
			for (size_t v = 0; v < this->m_appcontext->num_target_variables; v++)
			{
				this->m_prediction[c](zz,v) = (this->m_prediction[c](zz,v) * std[c](v));
				this->m_prediction[c](zz,v) += mean[c](v);
			}
		}
	}
}


template<typename TAppContext>
std::vector<double> LeafNodeStatisticsJointClassRegr<TAppContext>::CalculateADFTargetResidual_class(LabelJointClassRegr gt_label, int prediction_type)
{
	// CAUTION: for ARF, here we simply consider only the first vote ... this one should already be
	// meanshifted or be the mean of all votes?!!

	std::vector<double> ret_vec;
	if (prediction_type == 0) // CLASSIFICATION
	{
		// v1
		double max_conf_not_gt = -1.0;
		for (size_t c = 0; c < this->m_class_histogram.size(); c++)
		{
			if (c != gt_label.gt_class_label)
				if (this->m_class_histogram[c] > max_conf_not_gt)
					max_conf_not_gt = this->m_class_histogram[c];
		}
		ret_vec.resize(1, this->m_class_histogram[gt_label.gt_class_label] - max_conf_not_gt);

		// v0
		//ret_vec.resize(1, this->m_class_histogram[gt_label.gt_class_label]);
	}
	else if (prediction_type == 1) // REGRESSION
	{
		// init residuals to a zero vector
		ret_vec.resize(gt_label.regr_target.rows(), 0.0);

		// Skip samples that are not allowed to vote (e.g., background class in HoughForests)
		// We simply return a zero vector, which will be added to the inital zero vector, so
		// everything should be ok.
		if (!gt_label.vote_allowed)
			return ret_vec;

		// return the difference between the prediction & ground-truth
		for (size_t v = 0; v < ret_vec.size(); v++)
			ret_vec[v] = this->m_votes[gt_label.class_label][0](v) - gt_label.regr_target_gt(v); // correct order is important: prediction - ground-truth!
	}
	else
		throw std::logic_error("LeafNodeStats (calc ADF residuals): only class or regr prediction type available!");

	return ret_vec;
}

template<typename TAppContext>
struct residual LeafNodeStatisticsJointClassRegr<TAppContext>::CalculateADFTargetResidual(LabelJointClassRegr gt_label, vector <Eigen::MatrixXd> m_hough_center_prediction, Eigen::VectorXd mean, Eigen::VectorXd std, int s, int prediction_type)
{
	// CAUTION: for ARF, here we simply consider only the first vote ... this one should already be
	// meanshifted or be the mean of all votes?!!
	struct residual res;
	std::vector<double> ret_vec_aux;
	ret_vec_aux.resize(gt_label.regr_target.rows(), 0.0);
	std::vector<double> ret_vec_aux_best;
	ret_vec_aux_best.resize(gt_label.regr_target.rows(), 0.0);
	std::vector<double> ret_vec_aux_best_center;
	ret_vec_aux_best_center.resize(gt_label.regr_target.rows(), 0.0);
	std::vector<double> ret_vec_aux_center;
	ret_vec_aux_center.resize(gt_label.regr_target.rows(), 0.0);
	std::vector<double> ret_vec_estimated_regr_target;
	ret_vec_estimated_regr_target.resize(gt_label.regr_target.rows(), 0.0);
	std::vector<double> ret_vec_aux_best_center_norm;
	ret_vec_aux_best_center_norm.resize(gt_label.regr_target.rows(), 0.0);
	double bestNorm = 1e16;
	res.bestz = 0;
	
	if (prediction_type == 0) // CLASSIFICATION
	{
		// v1
		double max_conf_not_gt = -1.0;
		for (size_t c = 0; c < this->m_class_histogram.size(); c++)
		{
			if (c != gt_label.gt_class_label)
				if (this->m_class_histogram[c] > max_conf_not_gt)
					max_conf_not_gt = this->m_class_histogram[c];
		}
		res.ret_vec.resize(1, this->m_class_histogram[gt_label.gt_class_label] - max_conf_not_gt);
		res.bestz = gt_label.latent_prediction;
		// v0
		//ret_vec.resize(1, this->m_class_histogram[gt_label.gt_class_label]);
	}
	else if (prediction_type == 1) // REGRESSION
	{
		// init residuals to a zero vector
		res.ret_vec.resize(gt_label.regr_target.rows(), 0.0);

		// Skip samples that are not allowed to vote (e.g., background class in HoughForests)
		// We simply return a zero vector, which will be added to the inital zero vector, so
		// everything should be ok.
		if (!gt_label.vote_allowed)
			return res;

		// return the difference between the prediction & ground-truth
		

		for (size_t v = 0; v < gt_label.regr_target.rows(); v++){
			ret_vec_aux[v] = m_hough_center_prediction[gt_label.class_label](gt_label.latent_label-1, v) - gt_label.regr_center_gt(v);
			ret_vec_aux_center[v] = m_hough_center_prediction[gt_label.class_label](gt_label.latent_label-1, v);
		}
		ret_vec_aux_best = ret_vec_aux;
		ret_vec_aux_best_center = ret_vec_aux_center;
		res.bestz = gt_label.latent_label;
		double tempNorm = norm(ret_vec_aux, cv::NORM_L2);
		if(tempNorm == 0){
			res.ret_vec[0] = 0.0;
			res.ret_vec[1] = 0.0; 
		}else{
			for (size_t v = 0; v < gt_label.regr_target.rows(); v++){
				ret_vec_estimated_regr_target[v] = ret_vec_aux_best_center[v] - gt_label.regr_patch_center_gt(v);
				ret_vec_estimated_regr_target[v] -= mean(v);
				ret_vec_estimated_regr_target[v] /= std(v);
				res.ret_vec[v] = ret_vec_estimated_regr_target[v] - gt_label.regr_target_gt(v);
			}
		}


	}
	else
		throw std::logic_error("LeafNodeStats (calc ADF residuals): only class or regr prediction type available!");

	return res;
}


template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::AddTarget(LeafNodeStatisticsJointClassRegr* leafnodestats_src)
{
	// If we have a "only-negative" node, just return
	// INFO: this is a very special case for the joint classification-regression task! In standard
	// ARFs, we would propagate the target to the left and right child nodes. However, if we also
	// have a classification task, then it can happen that the splitting function separates classes
	// perfectly and one class is a non-voting class and the other one is a voting class.
	// Thus, if this is the child node with all non-voting samples, then we do not add the target and return!
	bool found_vote = false;
	for (size_t c = 0; c < this->m_votes.size(); c++)
	{
		if (this->m_votes[c].size() > 0)
		{
			found_vote = true;
			break;
		}
	}

	if (!found_vote)
		return;

	// INFO: for regression, i.e., ARFs, we only consider the "first vote", as we
	// expect that there is only a single vote (mean or meanshift)!
	// ==> See also the CalculateADFTargetResidual() method!
	for (size_t c = 0; c < leafnodestats_src->m_votes.size(); c++) // iterate the classes from the parent node
	{
		// could happen that the parent node doesn't have any votes for a certain class anymore,
		// so we don't need to accumulate them!
		if (leafnodestats_src->m_votes[c].size() == 0)
			continue;

		// it can also happen that the new child node has no votes for a certain class anymore
		// due to splitting, so we have to init a new vector and a voting weight!
		// INFO: Again, this is a special case and I really don't know how to handle this correctly ;)
		// How shall we set the voting weights? I my opinion it has to be set to 0, because we do not
		// have any samples of that class in this node! So, in accordance with the above special case,
		// we should simply not even add the vote here!
		// => as stated above, we really don't even consider this vote if there is no sample from that
		// class in this node
		if (this->m_votes[c].size() > 0)
		{
	
			this->m_votes[c][0] += 0.1*leafnodestats_src->m_votes[c][0]; 
			this->m_prediction[c] += 0.1*leafnodestats_src->m_prediction[c];
		}
	}
}



template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::Print()
{
	cout << this->m_num_samples << " samples; class-specific: ";
	for (size_t c = 0; c < this->m_num_samples_class.size(); c++)
		cout << this->m_num_samples_class[c] << " ";
	cout << endl;
	for (size_t c = 0; c < this->m_class_histogram.size(); c++)
		cout << this->m_class_histogram[c] << " ";
	cout << endl;
	cout << "Votes :";
	for (size_t c = 0; c < this->m_votes.size(); c++)
	{
		cout << c << " (" << this->m_votes[c].size() << "): ";
		for (size_t i = 0; i < this->m_votes[c].size(); i++)
		{
			cout << "(";
			for (size_t d = 0; d < this->m_votes[c][i].rows(); d++)
				cout << this->m_votes[c][i](d) << " ";
			cout << "; " << this->m_vote_weights[c][i];
			cout << ")";
		}
	}
	cout << endl;
}



template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::Save(std::ofstream& out, Eigen::MatrixXd& latent_variables)
{
	// sample statistics
	out << m_num_samples << " " << m_num_samples_class.size() << " ";
	for (size_t c = 0; c < m_num_samples_class.size(); c++)
		out << m_num_samples_class[c] << " ";

	// class histogram
	out << this->m_class_histogram.size() << " ";
	for (size_t c = 0; c < this->m_class_histogram.size(); c++)
		out << m_class_histogram[c] << " ";

	// voting vectors
	out << this->m_votes.size() << " "; // num classes!
	for (size_t c = 0; c < m_votes.size(); c++)
	{
		out << this->m_votes[c].size() << " ";
		if (this->m_votes[c].size() > 0){
			out << this->m_votes[c][0].size() << " ";

			for (size_t v = 0; v < this->m_votes[c].size(); v++)
				for (size_t d = 0; d < this->m_votes[c][v].rows(); d++)
					out << this->m_votes[c][v](d) << " ";

			out << m_appcontext->num_z << " " << m_appcontext->num_target_variables << " ";
			for (int zz=0; zz < m_appcontext->num_z; zz++){
		   		for (int v = 0; v < m_appcontext->num_target_variables; v++){

					if (isnan(this->m_prediction[c](zz,v))){
						out << 0 << " ";
					}else{
						out << this->m_prediction[c](zz,v) << " ";
					}		
				}

		   	}


			out << this->m_offsets[c].size() << endl;
			for (size_t v = 0; v < this->m_offsets[c].size(); v++){
				for (size_t d = 0; d < this->m_offsets[c][v].rows(); d++)
					out << this->m_offsets[c][v](d) << " ";
				
				out << this->m_azimuth[c][v] << " " << this->m_zenith[c][v] << endl;
				int z=0;
				for (int s=0; s < int(latent_variables.size()/2); s++){
					if(this->patch_id[v] == latent_variables(s,0)){
						z = latent_variables(s,1);
						break;
					}
				}
				out << this->m_latent_label[c][v] << " " << z << " " << this->img_id[v] << " ";
			}	

			out << this->m_vote_weights[c].size() << " ";
			for (size_t v = 0; v < this->m_vote_weights[c].size(); v++)
				out << this->m_vote_weights[c][v] << " ";
		}
	}

	// pseudo-class histogram
	out << this->m_pseudoclass_histogram.size() << " ";
	for (size_t c = 0; c < this->m_pseudoclass_histogram.size(); c++)
		out << m_pseudoclass_histogram[c] << " ";

	out << endl;
}


template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::Load(std::ifstream& in)
{

	// sample statistics
	int num_classes, num_z, num_targets;
	double dummy1, dummy2;
	in >> m_num_samples >> num_classes;
	this->m_num_samples_class.resize(num_classes);
	for (size_t c = 0; c < m_num_samples_class.size(); c++)
		in >> m_num_samples_class[c];

	// class histogram
	in >> num_classes;
	m_class_histogram.resize(num_classes);
	for (size_t c = 0; c < m_class_histogram.size(); c++)
		in >> m_class_histogram[c];


	// voting vectors
	in >> num_classes;
	m_offsets.resize(num_classes);
	m_azimuth.resize(num_classes);
	m_zenith.resize(num_classes);
	m_latent_label.resize(num_classes);
	m_latent_prediction.resize(num_classes);
	m_prediction.resize(num_classes);
	for (size_t c = 0; c < num_classes; c++)
	{
		int num_votes, vote_dim;
		in >> num_votes;
		m_votes[c].resize(num_votes);
		if (num_votes > 0){
			in >> vote_dim;
			
			for (size_t v = 0; v < m_votes[c].size(); v++)
			{
				
				m_votes[c][v] = Eigen::VectorXd::Zero(vote_dim);
				for (size_t d = 0; d < m_votes[c][v].rows(); d++)
					in >> m_votes[c][v](d);
			}
		
			in >> num_z >> num_targets;
			m_prediction[c] = Eigen::MatrixXd::Zero(num_z, num_targets);

			for (int zz=0; zz < num_z; zz++){
				for (int v = 0; v < num_targets; v++){
					in >> m_prediction[c](zz,v);
		   		}
			}

			int num_offsets;
			if (c != 0){
				in >> num_offsets;
				m_offsets[c].resize(num_offsets);
				img_id.resize(num_offsets);
				m_azimuth[c].resize(num_offsets);
				m_zenith[c].resize(num_offsets);
				m_latent_label[c].resize(num_offsets);
				m_latent_prediction[c].resize(num_offsets);
				for (size_t v = 0; v < num_offsets; v++)
				{
					m_offsets[c][v] = Eigen::VectorXd::Zero(vote_dim);
					for (size_t d = 0; d < m_offsets[c][v].rows(); d++)
						in >> m_offsets[c][v](d);

					in >> m_azimuth[c][v];	
					in >> m_zenith[c][v];				
					in >> m_latent_label[c][v];
					in >> m_latent_prediction[c][v];
					in >> img_id[v];
				}
			
				in >> num_votes;
				m_vote_weights[c].resize(num_votes);
				for (size_t v = 0; v < m_vote_weights[c].size(); v++){
					in >> m_vote_weights[c][v];
				}

			}
		}

	}

	// pseudo-class histogram
	int num_pseudoclasses;
	in >> num_pseudoclasses;
	m_pseudoclass_histogram.resize(num_pseudoclasses);
	for (size_t c = 0; c < m_pseudoclass_histogram.size(); c++)
		in >> m_pseudoclass_histogram[c];


}




// Private / Helper methods
template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::AggregateRegressionTargets(DataSet<SampleImgPatch, LabelJointClassRegr> dataset)
{
	// clear all votes
	this->m_votes.clear();
	this->m_vote_weights.clear();
	this->m_votes.resize(this->m_appcontext->num_classes);
	this->m_offsets.resize(this->m_appcontext->num_classes);
	this->m_azimuth.resize(this->m_appcontext->num_classes);
	this->m_zenith.resize(this->m_appcontext->num_classes);
	this->m_regr_target.resize(this->m_appcontext->num_classes);
	this->m_latent_label.resize(this->m_appcontext->num_classes);
	this->m_latent_prediction.resize(this->m_appcontext->num_classes);
	this->m_vote_weights.resize(this->m_appcontext->num_classes);
	this->m_prediction.resize(this->m_appcontext->num_classes);
	

	// iterate the classes
	for (size_t c = 0; c < this->m_votes.size(); c++)
	{

		this->m_prediction[c] = MatrixXd::Zero(m_appcontext->num_z, m_appcontext->num_target_variables);
		for (size_t zz = 0; zz < m_appcontext->num_z; zz++)
		{
			vector<int> sample_indices_z;
			int num_small_size_z = 0; 
			num_small_size_z = min(1000, this->m_num_samples_latent[zz]);
			sample_indices_z.resize(num_small_size_z);
			int scnt_z = 0;			
			
			for (size_t ii = 0; ii < dataset.size(); ii++)
			{

				// only include data that is allowed to vote and belongs to the current class of consideration!
				if (dataset[ii]->m_label.vote_allowed == true && dataset[ii]->m_label.class_label == c && dataset[ii]->m_label.latent_prediction == zz+1){

					sample_indices_z[scnt_z++] = ii;
				}
				if (scnt_z == num_small_size_z)
					break;
			}
			if (sample_indices_z.size() > 0) // caution, we could also get an empty list, because only "negative" data is available (could also happen because we randomly selected 1000 samples above!)
			{
				// compute the mean regression target
				//this->m_prediction[c] = MatrixXd::Zero(m_appcontext->num_z,m_appcontext->num_target_variables);
				for (size_t k = 0; k < sample_indices_z.size(); k++)
				{
					for(size_t v = 0; v < m_appcontext->num_target_variables; v++){
						this->m_prediction[c](zz, v) += dataset[sample_indices_z[k]]->m_label.regr_target(v);
					}
				}
	
				for(size_t v = 0; v < m_appcontext->num_target_variables; v++){
					this->m_prediction[c](zz,v) /= (double)sample_indices_z.size();
				}

			}

		}
	


		// find the samples contributing to the mode estimation
		vector<int> sample_indices;
		// As in the old implementation, we limit the number of votes to 1000
		int num_small_size = min(1000, this->m_num_samples_class[c]);
		// find a random subset of less samples ... (randInteger can be slow! -> so use the first ones)
		int scnt = 0;
		sample_indices.resize(num_small_size);
		int count_fg = 0;
		for (size_t i = 0; i < dataset.size(); i++)
		{
			// only include data that is allowed to vote and belongs to the current class of consideration!
			if (dataset[i]->m_label.vote_allowed == true && dataset[i]->m_label.class_label == c){
				count_fg ++;}

		}
		this->m_offsets[c].resize(count_fg);
		this->m_azimuth[c].resize(count_fg);
		this->m_zenith[c].resize(count_fg);
		this->m_regr_target[c].resize(count_fg);
		this->m_latent_label[c].resize(count_fg);
		this->img_id.resize(count_fg);
		this->m_latent_prediction[c].resize(count_fg);
		this->patch_id.resize(count_fg);
	
		int ind_aux = 0;
		for (size_t i = 0; i < dataset.size(); i++)
		{
			// only include data that is allowed to vote and belongs to the current class of consideration!
			if (dataset[i]->m_label.vote_allowed == true && dataset[i]->m_label.class_label == c){

				this->m_offsets[c][ind_aux] = Eigen::VectorXd::Zero(dataset[0]->m_label.regr_target.rows());
				this->m_regr_target[c][ind_aux] = Eigen::VectorXd::Zero(dataset[0]->m_label.regr_target.rows());
				this->m_offsets[c][ind_aux] = dataset[i]->m_label.regr_offset;
				this->m_regr_target[c][ind_aux] = dataset[i]->m_label.regr_target;
				this->m_azimuth[c][ind_aux] = dataset[i]->m_label.azimuth;
				this->m_zenith[c][ind_aux] = dataset[i]->m_label.zenith;
				this->m_latent_label[c][ind_aux] = dataset[i]->m_label.latent_label;
				this->img_id[ind_aux] = dataset[i]->m_label.img_id;
				this->m_latent_prediction[c][ind_aux] = dataset[i]->m_label.latent_prediction;
				this->patch_id[ind_aux] = dataset[i]->m_label.patch_id;
				ind_aux ++;

			}
		}
		for (size_t i = 0; i < dataset.size(); i++)
		{
			// only include data that is allowed to vote and belongs to the current class of consideration!
			if (dataset[i]->m_label.vote_allowed == true && dataset[i]->m_label.class_label == c){
				sample_indices[scnt++] = i;
			}
			if (scnt == num_small_size)
				break;
		}
		// resize the sample_indices to the real number of identified samples that are allowed to vote
		// and from the correct class
		if (scnt < sample_indices.size())
			sample_indices.resize(scnt);

		// aggregate the votes
		switch (m_appcontext->leafnode_regression_type)
		{
		case LEAFNODE_REGRESSION_TYPE::ALL:
			this->Aggregate_All(dataset, c, sample_indices);
			break;
		case LEAFNODE_REGRESSION_TYPE::MEAN:
			this->Aggregate_Mean(dataset, c, sample_indices);
			break;
		case LEAFNODE_REGRESSION_TYPE::HILLCLIMB:
			this->Aggregate_HillClimb(dataset, c, sample_indices);
			break;
		case LEAFNODE_REGRESSION_TYPE::MEANSHIFT:
			this->Aggregate_MeanShift(dataset, c, sample_indices, m_appcontext->mean_shift_votes_k);
			break;
		default:
			throw std::runtime_error("LeafNodeStatistics (aggregate regression targets): unknown regression type");
		}
	}

	// CAUTION: this is only a debugging solution for the binary case of HoughForests for object detection!
	if (this->m_votes[1].size() > 0)
		this->m_intermediate_prediction = this->m_votes[1][0];

}



template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::Aggregate_All(DataSet<SampleImgPatch, LabelJointClassRegr> dataset, int lblIdx, vector<int> sample_indices)
{
	// collect all votes
	this->m_votes[lblIdx].resize(sample_indices.size());
	for (size_t i = 0; i < sample_indices.size(); i++)
	{
		this->m_votes[lblIdx][i] = dataset[sample_indices[i]]->m_label.regr_target;
	}

	// voting weights (the sum of the offset weights is the foreground probability)
	this->m_vote_weights[lblIdx].resize(sample_indices.size(), this->m_class_histogram[lblIdx] / (double)sample_indices.size());
}



template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::Aggregate_Mean(DataSet<SampleImgPatch, LabelJointClassRegr> dataset, int lblIdx, vector<int> sample_indices)
{

	if (sample_indices.size() > 0) // caution, we could also get an empty list, because only "negative" data is available (could also happen because we randomly selected 1000 samples above!)
	{
		// compute the mean regression target
		this->m_votes[lblIdx].resize(1, Eigen::VectorXd::Zero(dataset[sample_indices[0]]->m_label.regr_target.rows()));
		for (size_t i = 0; i < sample_indices.size(); i++)
		{
			this->m_votes[lblIdx][0] += dataset[sample_indices[i]]->m_label.regr_target;
		}
		this->m_votes[lblIdx][0] /= (double)sample_indices.size();

		// voting weights, in this case it is the class probability (the sum of the offset weights is the foreground probability)
		this->m_vote_weights[lblIdx].resize(1, this->m_class_histogram[lblIdx]);
	}
}



template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::Aggregate_HillClimb(DataSet<SampleImgPatch, LabelJointClassRegr> dataset, int lblIdx, vector<int> sample_indices)
{
	throw std::logic_error("LeafNodeStatistics (Aggreagte_HillClimb): not implemented yet");
}


template<typename TAppContext>
void LeafNodeStatisticsJointClassRegr<TAppContext>::Aggregate_MeanShift(DataSet<SampleImgPatch, LabelJointClassRegr> dataset, int lblIdx, vector<int> sample_indices, int num_modes)
{
	throw std::logic_error("LeafNodeStatistics (Aggregate_MeanShift): not implemented yet");
}










#endif /* LEAFNODESTATISTICSJOINTCLASSREGR_CPP_ */



	
