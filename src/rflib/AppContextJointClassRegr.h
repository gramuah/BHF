#ifndef APPCONTEXTJOINTCLASSREGR_H_
#define APPCONTEXTJOINTCLASSREGR_H_

#include <ostream>
#include <string>
#include <vector>
#include <libconfig.h++>
#include "icgrf.h"

#include "AppContext.h"

using std::cout;
using std::endl;


class AppContextJointClassRegr : public AppContext
{
public:
	AppContextJointClassRegr() {}
	virtual ~AppContextJointClassRegr() {}

protected:
	// implements the abstract base method!
	inline void ValidateHyperparameters()
	{
		if (!AppContext::ValidateCompleteGeneralPart())
		{
			cout << "General settings missing!" << endl;
			exit(-1);
		}

		if (!AppContext::ValidateStandardForestSettings())
		{
			cout << "Standard Forest settings missing" << endl;
			exit(-1);
		}

		if (this->leafnode_regression_type == LEAFNODE_REGRESSION_TYPE::NOTSET)
		{
			cout << "specify a leafnode-regression type!" << endl;
			exit(-1);
		}

		if (!AppContext::ValidateImageSplitFunction())
		{
			cout << "you have to specify a split function suitable for images!" << endl;
			exit(-1);
		}

		if (this->method != RF_METHOD::HOUGHFOREST && this->method != RF_METHOD::BOOSTEDHOUGHFOREST &&
			this->method != RF_METHOD::ADHOUGHFOREST)
		{
			cout << "Use one of these methods: StdHF (5), BoostedHF (6), ADHF (7): " << this->method << endl;
			exit(-1);
		}

		if (!this->ValidateHoughDetectionSpecificStuff())
		{
			cout << "Some Hough forest specific parameters are missing in the config file!" << endl;
			exit(-1);
		}

		if (!this->ValidateHeadPoseEstimationSpecificStuff())
		{
			cout << "Some Head-pose estimation parameters are missing in the config file!" << endl;
			exit(-1);
		}

		if (!this->ValidateHoughDetectionDataStuff())
		{
			cout << "Some Hough-specific data parameters are missing!" << endl;
			exit(-1);
		}

		if (this->method == RF_METHOD::ADHOUGHFOREST || this->method == RF_METHOD::BOOSTEDHOUGHFOREST)
		{
			if (!this->ValidateADFParameters())
			{
				cout << "some ADF parameters are missing!" << endl;
				exit(-1);
			}
			if (!this->ValidateARFParameters())
			{
				cout << "some ARF parameters are missing!" << endl;
				exit(-1);
			}
		}



		if (this->mean_shift_votes_k <= 0)
		{
			cout << "ERROR: the k for the mean shift votes has to be > 0" << endl;
			exit(-1);
		}

		if ((this->method == RF_METHOD::ADHOUGHFOREST || this->method == RF_METHOD::BOOSTEDHOUGHFOREST) &&
				this->do_classification_weight_updates == 0 && this->do_regression_weight_updates == 0)
		{
			cout << "WARNING: both classification and regression weight updates are turned off -> this is a standard HF" << endl;
		}

		if (this->backproj_bbox_cumsum_th == 0.0)
		{
			cout << "WARNING: the parameter <backproj_bbox_cumsum_th> should not be set to exactly zero! it gets set to 0.001" << endl;
			this->backproj_bbox_cumsum_th = 0.001;
		}

		if (!return_bboxes && !print_hough_maps)
		{
			cout << "WARNING: neither hough maps nor bboxes will be returned ... turning on <return bboxes>" << endl;
			this->return_bboxes = 1;
		}

		if (!return_bboxes && backproj_bbox_estimation)
		{
			cout << "WARNING: return bboxes is off but backprojections are on ... turning on <return bboxes>" << endl;
			this->return_bboxes = 1;
		}

		if (this->store_dataset && this->load_dataset)
		{
			cout << "WARNING: Storing & Loading fixed data set is active: turning OFF store_dataset" << endl;
			this->store_dataset = 0;
		}
	}


	inline bool ValidateHoughDetectionSpecificStuff()
	{
		// forest training stuff
		if (this->mean_shift_votes_k == -1)
		{
			std::cout << "mean shift votes k" << std::endl;
			return false;
		}

		if (this->depth_regression_only == -1)
		{
			std::cout << "depth regression only" << std::endl;
			return false;
		}

		// prediction stuff
		if (this->use_meanshift_voting == -1)
		{
			std::cout << "use meanshift voting" << std::endl;
			return false;
		}

		if (this->voting_grid_distance == -1)
		{
			std::cout << "voting grid distance" << std::endl;
			return false;
		}

		if (this->houghmaps_outputscale == -1)
		{
			std::cout << "houghmaps outputscale" << std::endl;
			return false;
		}

		if (this->print_detection_images == -1)
		{
			std::cout << "print detection images" << std::endl;
			return false;
		}

		if (this->patch_size.size() == 0)
		{
			std::cout << "patch size" << std::endl;
			return false;
		}

		if (this->return_bboxes == -1)
		{
			std::cout << "return bboxes" << std::endl;
			return false;
		}

		if (this->avg_bbox_scaling.size() == 0)
		{
			std::cout << "avg bbox scaling" << std::endl;
			return false;
		}

		if (this->use_min_max_filters == -1)
		{
			std::cout << "use min max filters" << std::endl;
			return false;
		}

		if (this->backproj_bbox_estimation == -1)
		{
			std::cout << "backproj bbox estimation" << std::endl;
			return false;
		}

		if (this->backproj_bbox_kernel_w == -1)
		{
			std::cout << "backproj bbox kernel w" << std::endl;
			return false;
		}

		if (this->backproj_bbox_cumsum_th < 0.0)
		{
			std::cout << "backproj bbox cumsum th" << std::endl;
			return false;
		}

		if (this->nms_type == NMS_TYPE::NOTSET)
		{
			std::cout << "nms type" << std::endl;
			return false;
		}

		if (this->hough_gauss_kernel == -1)
		{
			std::cout << "hough gauss kernel" << std::endl;
			return false;
		}

		if (this->max_detections_per_image == -1)
		{
			std::cout << "max detections per image" << std::endl;
			return false;
		}

		if (this->min_detection_peak_height < 0.0)
		{
			std::cout << "min detection peak height" << std::endl;
			return false;
		}

		if (this->use_scale_interpolation == -1)
		{
			std::cout << "use scale interpolation" << std::endl;
			return false;
		}

		if (this->test_scales.size() == 0)
		{
			std::cout << "test scales" << std::endl;
			return false;
		}

		return true;
	}

	inline bool ValidateHeadPoseEstimationSpecificStuff()
	{
		if (this->poseestim_maxvar_vote < 0.0)
			return false;

		if (this->poseestim_min_fgprob_vote < 0.0)
			return false;

		if (this->path_poseestimates.empty())
			return false;

		return true;
	}

	inline bool ValidateHoughDetectionDataStuff()
	{
		if (this->path_houghimages.empty())
			return false;

		if (this->path_bboxes.empty())
			return false;

		if (this->path_detectionimages.empty())
			return false;


		if (this->store_dataset == -1)
			return false;

		if (this->load_dataset == -1)
			return false;

		if (this->path_fixedDataset.empty())
			return false;

		if (this->path_posImages.empty())
			return false;

		if (this->path_posAnnofile.empty())
			return false;

		if (this->numPosPatchesPerImage == -1)
			return false;

		if (this->use_segmasks == -1)
			return false;

		if (this->path_negImages.empty())
			return false;

		if (this->path_negAnnofile.empty())
			return false;

		if (this->numNegPatchesPerImage == -1)
			return false;

		if (this->path_testImages.empty())
			return false;
		
		if (this->path_testFilenames.empty())
			return false;

		return true;
	}

};


#endif /* APPCONTEXTJOINTCLASSREGR_H_ */
