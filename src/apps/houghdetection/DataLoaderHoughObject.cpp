/*
 * DataSetLoaderHoughObject.cpp
 * 
 * Author: Carolina Redondo Cabrera, Roberto Javier López-Sastre, Alejandro Véliz Fernández
 * Institution: GRAM, University of Alcalá, Spain
 * 
 */

#include "DataLoaderHoughObject.h"

std::vector<std::vector<cv::Mat> > DataLoaderHoughObject::image_feature_data;


DataLoaderHoughObject::DataLoaderHoughObject(AppContextJointClassRegr* hp) : m_hp(hp) { }

DataLoaderHoughObject::~DataLoaderHoughObject() { }


DataSet<SampleImgPatch, LabelJointClassRegr> DataLoaderHoughObject::LoadTrainData()
{
	// 1) load pre-defined image data
	vector<MatrixXd> patch_locations;
	if (m_hp->load_dataset)
	{
		patch_locations = this->LoadPatchPositions(m_hp->path_fixedDataset);
		if (!m_hp->quiet)
			std::cout << "Loaded predefined random patch locations!" << std::endl;
	}

	// 2) load positive image data
	vector<string> vFilenames;
	vector<cv::Rect> vBBox;
	vector<VectorXd> vOffsets;
	vector<vector<cv::Point> > vSegmasks;
	vector<int> z_targets;
	vector<double> pose_targets;
	vector<double> ze_targets;
	Eigen::VectorXd vSize = Eigen::VectorXd::Zero(2);

	this->LoadPosTrainFile(vFilenames, vBBox, vOffsets, vSegmasks, z_targets, pose_targets, ze_targets, this->m_num_target_variables, this->m_num_z);
	if (!m_hp->quiet)
		std::cout << vFilenames.size() << " positive images available for cropping patches" << std::endl;
	if (!m_hp->quiet)
		std::cout << "Progress: " << std::flush;
	for (size_t i = 0; i < vFilenames.size(); i++)
	{

		if (!m_hp->quiet && ((int)i % (int)round((double)vFilenames.size()/10.0)) == 0)
			std::cout << round((double)i/(double)vFilenames.size()*100) << "% " << std::flush;

		// read image
		cv::Mat img_raw = DataLoaderHoughObject::ReadImageData(this->m_hp->path_posImages + "/" + vFilenames[i]);
		// scaling
		cv::Mat img;
		cv::Size new_size = cv::Size((int)((double)img_raw.cols*m_hp->general_scaling_factor), (int)((double)img_raw.rows*m_hp->general_scaling_factor));
		resize(img_raw, img, new_size, 0, 0, cv::INTER_LINEAR);

		// extract features
		std::vector<cv::Mat> img_features;
		DataLoaderHoughObject::ExtractFeatureChannelsObjectDetection(img, img_features, m_hp);

		// generate random patch positions
		if (!m_hp->load_dataset)
			patch_locations.push_back(this->GeneratePatchPositionsFromRegion(img, img_features, m_hp->numPosPatchesPerImage, &vBBox[i], 0));

		// extract image patches
		vSize(0) = vBBox[i].width;
    	vSize(1) = vBBox[i].height;
            
		this->ExtractPatches(m_trainset, img, img_features, patch_locations[i], i, 1, z_targets[i], pose_targets[i], ze_targets[i], vOffsets[i], vSize);
	}

	if (!m_hp->quiet)
		std::cout << std::endl;

	// 3) read negative data
	int n_pos_imgs = vFilenames.size();
	vFilenames.clear();
	vBBox.clear();
	this->LoadNegTrainFile(vFilenames, vBBox);

	if (!m_hp->quiet)
		std::cout << vFilenames.size() << " negative images available for cropping patches" << std::endl;
	if (!m_hp->quiet)
		std::cout << "Progress: ";

	for (size_t i = 0; i < vFilenames.size(); i++)
	{
		if (!m_hp->quiet && ((int)i % (int)round((double)vFilenames.size()/10.0)) == 0)
			std::cout << round((double)i/(double)vFilenames.size()*100) << "% " << std::flush;

		// read image
		cv::Mat img_raw = DataLoaderHoughObject::ReadImageData(this->m_hp->path_negImages + "/" + vFilenames[i]);
		// scaling
		cv::Mat img;
		cv::Size new_size = cv::Size((int)((double)img_raw.cols*m_hp->general_scaling_factor), (int)((double)img_raw.rows*m_hp->general_scaling_factor));
		resize(img_raw, img, new_size, 0, 0, cv::INTER_LINEAR);

		// calculate image features
		std::vector<cv::Mat> img_features;
		DataLoaderHoughObject::ExtractFeatureChannelsObjectDetection(img, img_features, m_hp);

		// extract patches
		if (!m_hp->load_dataset)
			patch_locations.push_back(this->GeneratePatchPositionsFromRegion(img, img_features, m_hp->numNegPatchesPerImage, &vBBox[i], 0));

		this->ExtractPatches(m_trainset, img, img_features, patch_locations[n_pos_imgs + i], n_pos_imgs+i, 0, 0, -1, -1);
	}
	if (!m_hp->quiet)
		std::cout << std::endl;

	// 4) store patch locations (for both pos and neg data!)
	if (m_hp->store_dataset)
	{
		this->SavePatchPositions(m_hp->path_fixedDataset, patch_locations);
		if (!m_hp->quiet)
			std::cout << "Stored random patch locations!" << std::endl;
	}

	this->m_num_samples = m_trainset.size();
	this->m_num_classes = 2;
	this->m_num_target_variables = 2;
	this->m_num_feature_channels = DataLoaderHoughObject::image_feature_data[0].size();
<<<<<<< HEAD
	
	// return filled dataset
=======

	// 5) update the sample weights
	this->UpdateSampleWeights(m_trainset);
	// 6) return filled dataset
>>>>>>> a852804173915d67e48d70c4aee0b141323b9b7b
	return m_trainset;
}


void DataLoaderHoughObject::GetTrainDataProperties(int& num_samples, int& num_classes, int& num_target_variables, int& num_feature_channels, int& num_z)
{
	num_samples = this->m_num_samples;
	num_classes = this->m_num_classes;
	num_target_variables = this->m_num_target_variables;
	num_feature_channels = this->m_num_feature_channels;
	num_z = this->m_num_z;
}


cv::Mat DataLoaderHoughObject::ReadImageData(std::string imgpath)
{

	cv::Mat img = cv::imread(imgpath);
	if (!img.data)
		throw std::runtime_error("Error reading image:" + imgpath);
	return img;
}


void DataLoaderHoughObject::ExtractFeatureChannelsObjectDetection(const cv::Mat& img, vector<cv::Mat>& vImg, AppContextJointClassRegr* appcontext)
{
	// 32 feature channels
	// 7+9 channels: L, a, b, |I_x|, |I_y|, |I_xx|, |I_yy|, HOGlike features with 9 bins (weighted orientations 5x5 neighborhood)

	// if we use min+max filters:
	//   16+16 channels: minfilter + maxfilter on 5x5 neighborhood
	// if not:
	//   16 channels

	bool use_integral_image = false;
	if (appcontext->split_function_type == SPLITFUNCTION_TYPE::HAAR_LIKE &&
			appcontext->use_min_max_filters == 0) // only if haar-like features and min-max filters are NOT used!
		use_integral_image = true;

	cv::Mat img_gray;
	cv::cvtColor(img, img_gray, CV_RGB2GRAY);

	FeatureGeneratorRGBImage fg;
	fg.ExtractChannel(FC_LAB, use_integral_image, img, vImg);
	fg.ExtractChannel(FC_SOBEL, use_integral_image, img_gray, vImg);
	fg.ExtractChannel(FC_GRAD2, use_integral_image, img_gray, vImg);
	fg.ExtractChannel(FC_HOG, use_integral_image, img_gray, vImg);


	// apply a Gauss filter to all channels
	for (size_t i = 0; i < vImg.size(); i++)
	{
		cv::GaussianBlur(vImg[i], vImg[i], cv::Size(3, 3), 0, 0);
	}


	if (appcontext->use_min_max_filters)
	{
		// if use min-max filters are used, then the integral image is calcuated after the min-max filters
		if (appcontext->split_function_type == SPLITFUNCTION_TYPE::HAAR_LIKE)
			use_integral_image = true;

		// number of feature channels before min-max filtering
		size_t num_channels_prior = vImg.size();

		// do the min max filtering
		for (size_t c = 0; c < num_channels_prior; c++)
			fg.ExtractChannel(FC_MIN_MAX, use_integral_image, vImg[c], vImg);

		// erase the unfiltered channels
		for (size_t c = 0; c < num_channels_prior; c++)
			vImg.erase(vImg.begin());
	}
}

void DataLoaderHoughObject::NormalizeRegressionTargets(std::vector<VectorXd>& mean, std::vector<VectorXd>& std)
{
	// calculate the mean per votable class
	mean.resize(this->m_num_classes, VectorXd::Zero(this->m_num_target_variables));
	std::vector<int> num_samples(this->m_num_classes, 0);
	for (size_t i = 0; i < this->m_trainset.size(); i++)
	{
		int lblIdx = this->m_trainset[i]->m_label.class_label;
		if (this->m_trainset[i]->m_label.vote_allowed)
		{
			mean[lblIdx] += this->m_trainset[i]->m_label.regr_target;
			num_samples[lblIdx]++;
		}
	}
	for (size_t c = 0; c < mean.size(); c++)
	{
		if (num_samples[c] > 0)
			mean[c] /= (double)num_samples[c];
	}

	// calculate std (and also subtract the mean alongside!)
	std.resize(this->m_num_classes, VectorXd::Zero(this->m_num_target_variables));
	for (size_t i = 0; i < this->m_trainset.size(); i++)
	{
		int lblIdx = this->m_trainset[i]->m_label.class_label;
		if (this->m_trainset[i]->m_label.vote_allowed)
		{
			// subtract the mean
			this->m_trainset[i]->m_label.regr_target -= mean[lblIdx];

			//Voting process without normalization
			this->m_trainset[i]->m_label.regr_patch_center_norm_gt = this->m_trainset[i]->m_label.regr_patch_center_gt - mean[lblIdx];
			
			// add the squared differences, TODO: libEigen3 definitely offers a more easy way to compute this!
			std[lblIdx] += (this->m_trainset[i]->m_label.regr_target.array() * this->m_trainset[i]->m_label.regr_target.array()).matrix();
<<<<<<< HEAD
=======
			
>>>>>>> a852804173915d67e48d70c4aee0b141323b9b7b
		}
	}
	// normalize the squared diffs
	for (size_t c = 0; c < std.size(); c++)
	{
		std[c] /= (static_cast<double>(num_samples[c]) - 1.0);
		for (size_t t = 0; t < std[c].rows(); t++) // TODO: also this should be easier with libEigen3
			std[c](t) = sqrt(std[c](t));
	}

	// in order to keep the relative importance between the target variables,
	// we have to compute a single scaling value! In our case, we compute the
	// mean of the standard deviation values
	for (size_t c = 0; c < std.size(); c++)
		std[c] = VectorXd::Ones(std[c].rows()) * (std[c].sum() / (double)std[c].rows());

	// finally, we have to apply the std scaling!
	for (size_t i = 0; i < this->m_trainset.size(); i++)
	{
		int lblIdx = this->m_trainset[i]->m_label.class_label;
		if (this->m_trainset[i]->m_label.vote_allowed)
		{
<<<<<<< HEAD
=======


>>>>>>> a852804173915d67e48d70c4aee0b141323b9b7b
			this->m_trainset[i]->m_label.regr_target = (this->m_trainset[i]->m_label.regr_target.array() / std[lblIdx].array()).matrix();
			this->m_trainset[i]->m_label.regr_target_gt =this->m_trainset[i]->m_label.regr_target;

			//Voting process without normalization
<<<<<<< HEAD
=======
			
>>>>>>> a852804173915d67e48d70c4aee0b141323b9b7b
			this->m_trainset[i]->m_label.regr_patch_center_norm_gt = (this->m_trainset[i]->m_label.regr_patch_center_norm_gt.array() / std[lblIdx].array()).matrix();
		
		}
	}
}


void DataLoaderHoughObject::DenormalizeRegressionTargets(std::vector<VectorXd> mean, std::vector<VectorXd> std)
{
	for (size_t i = 0; i < this->m_trainset.size(); i++)
	{
		int lblIdx = this->m_trainset[i]->m_label.class_label;
		if (this->m_trainset[i]->m_label.vote_allowed)
		{
			// multiply with std
			this->m_trainset[i]->m_label.regr_target = (this->m_trainset[i]->m_label.regr_target.array() * std[lblIdx].array()).matrix();
			// add the mean
			this->m_trainset[i]->m_label.regr_target += mean[lblIdx];
			// also set the gt offset!
			this->m_trainset[i]->m_label.regr_target_gt = this->m_trainset[i]->m_label.regr_target;
		}
	}
}











// PRIVATE / HELPER METHODS
void DataLoaderHoughObject::LoadPosTrainFile(vector<string>& vFilenames, vector<cv::Rect>& vBBox, vector<VectorXd>& vTarget,  vector<vector<cv::Point> >& vSegmasks, vector<int>& z_targets, vector<double>& pose_targets, vector<double>& ze_targets, int& num_target_dims, int& num_z)
{
    unsigned int size;
    ifstream in(m_hp->path_posAnnofile.c_str());
    if (in.is_open())
    {
<<<<<<< HEAD
		int dummy;
	    in >> size;
	    in >> num_target_dims;
		in >> num_z;
		cout << size << " " << num_target_dims << " " << num_z << endl;
	    
	    vFilenames.resize(size);
	    vTarget.resize(size);
	    vBBox.resize(size);
	    vSegmasks.resize(size);
		pose_targets.resize(size);
		z_targets.resize(size);
		ze_targets.resize(size);
        
=======
	int dummy;
        in >> size;
        in >> num_target_dims;
	in >> num_z;
        vFilenames.resize(size);
        vTarget.resize(size);
        vBBox.resize(size);
        vSegmasks.resize(size);
	pose_targets.resize(size);
	z_targets.resize(size);
	ze_targets.resize(size);
>>>>>>> a852804173915d67e48d70c4aee0b141323b9b7b
        for (unsigned int i=0; i<size; ++i)
        {
            // Read filename
            in >> vFilenames[i];

            // Read bounding box
            in >> vBBox[i].x;
            in >> vBBox[i].y;
            in >> vBBox[i].width;
            vBBox[i].width -= vBBox[i].x;
            in >> vBBox[i].height;
            vBBox[i].height -= vBBox[i].y;

            VectorXd tmpTarget = VectorXd::Zero(num_target_dims);
		    double tmpval;
		    for (int j = 0; j < tmpTarget.size(); j++)
		    {
				in >> tmpval;
				tmpTarget(j) = tmpval;
		    }

		    vTarget[i] = tmpTarget;

	 	    // scaling
		    vBBox[i].x = (int)((double)vBBox[i].x * m_hp->general_scaling_factor);
		    vBBox[i].y = (int)((double)vBBox[i].y * m_hp->general_scaling_factor);
		    vBBox[i].width = (int)((double)vBBox[i].width * m_hp->general_scaling_factor);
		    vBBox[i].height = (int)((double)vBBox[i].height * m_hp->general_scaling_factor);
		   	
		   	for (int j = 0; j < vTarget[i].rows(); j++)
				vTarget[i](j) = round(vTarget[i](j) * m_hp->general_scaling_factor);

		    in >> pose_targets[i];
		    in >> ze_targets[i];
		    in >> z_targets[i]; 

            // Read the segmasks
            if (m_hp->use_segmasks)
            {
                stringstream ss;
                ss << m_hp->path_posImages << "/" << vFilenames[i] << ".txt";
                ifstream inmasks(ss.str().c_str());

                if (inmasks.is_open())
                {
                    int num_points;
                    inmasks >> num_points;
                    vSegmasks[i].resize(num_points);
                    for (int p = 0; p < num_points; p++)
                    {
                        int y;
                        int x;
                        inmasks >> y;
                        inmasks >> x;
                        vSegmasks[i][p] = cv::Point(x, y);
                    }
                }
                else
                {
                	throw std::runtime_error("ERROR opening segmasks file: " + ss.str());
                }
            }
        }
        in.close();
    }
    else
    {
    	throw std::runtime_error("Train pos file not found" + m_hp->path_posAnnofile);
    }
}

void DataLoaderHoughObject::LoadNegTrainFile(vector<string>& vFilenames, vector<cv::Rect>& vBBox)
{
    unsigned int size, numop;
    ifstream in(m_hp->path_negAnnofile.c_str());
    vFilenames.clear();
    vBBox.clear();
    if (in.is_open())
    {
        in >> size;
        in >> numop;
        vFilenames.resize(size);
        if (numop > 0)
            vBBox.resize(size);
        else
            vBBox.clear();

        for (unsigned int i = 0; i < size; i++)
        {
            // Read filename
            in >> vFilenames[i];

            // Read bounding box (if available)
            if (numop > 0)
            {
                in >> vBBox[i].x;
                in >> vBBox[i].y;
                in >> vBBox[i].width;
                vBBox[i].width -= vBBox[i].x;
                in >> vBBox[i].height;
                vBBox[i].height -= vBBox[i].y;
            }

            // scaling
			vBBox[i].x = (int)((double)vBBox[i].x * m_hp->general_scaling_factor);
			vBBox[i].y = (int)((double)vBBox[i].y * m_hp->general_scaling_factor);
			vBBox[i].width = (int)((double)vBBox[i].width * m_hp->general_scaling_factor);
			vBBox[i].height = (int)((double)vBBox[i].height * m_hp->general_scaling_factor);

        }
        in.close();
    }
    else
    {
    	throw std::runtime_error("Train neg file not found " + m_hp->path_negAnnofile);
    }
}



Eigen::MatrixXd DataLoaderHoughObject::GeneratePatchPositionsFromRegion(const cv::Mat& img, const std::vector<cv::Mat>& img_features, int num_patches, const cv::Rect* include, const cv::Rect* exclude)
{
	Eigen::MatrixXd patch_locations = Eigen::MatrixXd(num_patches, 2);
	int offx = m_hp->patch_size[1] / 2;
	int offy = m_hp->patch_size[0] / 2;
	int patch_width = m_hp->patch_size[1];
	int patch_height = m_hp->patch_size[0];

	// ###############################################################
	// NO INCLUDE OR EXCLUDE REGION -> whole image can be used
	// ###############################################################
	if (include == 0 && exclude == 0)
	{
		Eigen::VectorXi rand_locs_x = randInteger(0, img.cols-patch_width, num_patches);
		Eigen::VectorXi rand_locs_y = randInteger(0, img.rows-patch_height, num_patches);
		for (int i = 0; i < num_patches; i++)
		{
			patch_locations(i, 0) = rand_locs_x(i);
			patch_locations(i, 1) = rand_locs_y(i);
		}
	}
	// ###############################################################
	// ONLY INCLUDE REGION -> patches only from include region
	// ###############################################################
	else if (exclude == 0)
	{
		Eigen::VectorXi rand_locs_x = randInteger(include->x, include->x+include->width-patch_width, num_patches);
		Eigen::VectorXi rand_locs_y = randInteger(include->y, include->y+include->height-patch_height, num_patches);
		for (int i = 0; i < num_patches; i++)
		{
			patch_locations(i, 0) = rand_locs_x(i);
			patch_locations(i, 1) = rand_locs_y(i);
		}
	}
	// ###############################################################
	// ONLY EXCLUDE REGION -> patches not from the exclude region!
	// ###############################################################
	else if (include == 0)
	{
		// lets do it the easy way ...
		int pCount = 0, nTry = 0;
		do
		{
			nTry++;
			if (nTry > num_patches*1000)
				throw std::runtime_error("DataLoader: Tried too many patch locations. Check your settings!");

			int x = randInteger(0, img.cols - patch_width);
			int y = randInteger(0, img.rows - patch_height);

			if ((x > exclude->x-5) && (x < exclude->x+exclude->width+4) && (y > exclude->y-5) && (y < exclude->y+exclude->height+4))
				continue;

			patch_locations(pCount, 0) = x;
			patch_locations(pCount, 1) = y;
			pCount++;
		}
		while (pCount < num_patches);
	}
	else
	{
		// ###############################################################
		// INCLUDE AND EXCLUDE REGION DEFINED -> hmmm, ok shouldn't happen that often ;)
		// ###############################################################
		int pCount = 0;
		int nTry = 0;
		do {
			nTry++;
			if(nTry > num_patches*1000)
				throw std::runtime_error("DataLoader: Tried too many patch locations. Check your settings!");

			int x = randInteger(include->x, include->x+include->width-patch_width);
			int y = randInteger(include->y, include->y+include->height-patch_height);

			if((x > exclude->x) && (x < exclude->x+exclude->width) && (y > exclude->y) && (y < exclude->y+exclude->height))
				continue;

			patch_locations(pCount, 0) = x;
			patch_locations(pCount, 1) = y;
			pCount++;
		}
		while (pCount < num_patches);
	}
	return patch_locations;
}

Eigen::MatrixXd DataLoaderHoughObject::GeneratePatchPositionsFromMask(const cv::Mat& img, const std::vector<cv::Mat>& img_features, int num_patches, vector<cv::Point> segmask_points)
{
	throw std::logic_error("DataLoader: get patches from segmask -> not implemented yet");
	return Eigen::MatrixXd::Zero(1, 1);
}

void DataLoaderHoughObject::SavePatchPositions(std::string savepath, std::vector<Eigen::MatrixXd> patchpositions)
{
	// write a new file with: #locations and locations (x,y)
	std::ofstream file;
	file.open(savepath.c_str(), ios::binary);
	file << patchpositions.size() << endl; // num images
	for (size_t i = 0; i < patchpositions.size(); i++)
	{
		file << patchpositions[i].rows() << " " << patchpositions[i].cols() << endl;
		file << patchpositions[i] << std::endl;
	}
	file.close();
}

std::vector<Eigen::MatrixXd> DataLoaderHoughObject::LoadPatchPositions(std::string loadpath)
{
	std::ifstream file;
	file.open(loadpath.c_str(), ios::in);
	int ni;
	file >> ni; // number of images
	vector<Eigen::MatrixXd> vec_patchpositions(ni);
	for (size_t i = 0; i < ni; i++)
	{
		int nr, nc; // number of rows and cols
		file >> nr >> nc;
		Eigen::MatrixXd patchpositions = Eigen::MatrixXd::Zero(nr, nc);
		for (int r = 0; r < nr; r++)
		{
			for (int c = 0; c < nc; c++)
			{
				int tmp;
				file >> tmp;
				patchpositions(r, c) = (double)tmp;
			}
		}
		vec_patchpositions[i] = patchpositions;
	}
	return vec_patchpositions;
}



void DataLoaderHoughObject::ExtractPatches(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset, const cv::Mat& img, const std::vector<cv::Mat>& img_features, Eigen::MatrixXd patch_locations, int img_index, int label, int z, double azimuth, double zenith, VectorXd vTarget, VectorXd vSize)
{
	int offx = m_hp->patch_size[1] / 2;
	int offy = m_hp->patch_size[0] / 2;
	int patch_width = m_hp->patch_size[1];
	int patch_height = m_hp->patch_size[0];
	Eigen::VectorXd patchCenter = Eigen::VectorXd::Zero(vTarget.rows());
	int npatches = patch_locations.rows();
	
	for (int i = 0; i < npatches; i++)
	{
		// patch position
		CvPoint pt = cvPoint(patch_locations(i, 0), patch_locations(i, 1));

		// create the patch data
		int datastore_id = static_cast<int>(image_feature_data.size());
		cv::Rect patch_roi = cv::Rect(pt.x, pt.y, patch_width, patch_height);
		vector<cv::Mat> patch_features(img_features.size());
		
		for (unsigned int c = 0; c < img_features.size(); c++)
		{
			cv::Mat roiImg(img_features[c], patch_roi);
			roiImg.copyTo(patch_features[c]);
		}
		
		// add a COPY of the data to the data-store! -> then, we can release all other data!
		DataLoaderHoughObject::image_feature_data.push_back(patch_features);
		SampleImgPatch patchdata;
		patchdata.features = DataLoaderHoughObject::image_feature_data[datastore_id];
		patchdata.x = 0;
		patchdata.y = 0;

		if (m_hp->split_function_type == SPLITFUNCTION_TYPE::HAAR_LIKE)
		{
			cv::Mat temp_mask;
			temp_mask.create(cv::Size(m_hp->patch_size[1], m_hp->patch_size[0]), CV_8U);
			temp_mask.setTo(cv::Scalar::all(1.0));
			cv::integral(temp_mask, patchdata.normalization_feature_mask, CV_32F);
		}

		// create the patch label information
		Eigen::VectorXd patch_offset_vector = Eigen::VectorXd::Zero(2);
		Eigen::VectorXd imgSize = Eigen::VectorXd::Zero(2);
		if (vTarget.size() > 0 && label == 1)
		{
			patch_offset_vector(0) = vTarget(0) - (pt.x + offx);
			patch_offset_vector(1) = vTarget(1) - (pt.y + offy);
			patchCenter(0) = (pt.x + offx);
			patchCenter(1) = (pt.y + offy);
			imgSize(0) = vSize(0); //cols
			imgSize(1) = vSize(1); //rows
		}
		
		bool allow_vote = false;
		if (label == 1) // we only consider binary classification here!
			allow_vote = true;
		
		LabelJointClassRegr patchlabel(1.0, label, z, azimuth, zenith, vTarget, patchCenter, imgSize, img_index, 1.0, patch_offset_vector, allow_vote);
		double sample_weight = 1.0; // obsolete

		LabelledSample<SampleImgPatch, LabelJointClassRegr>* sample = new LabelledSample<SampleImgPatch, LabelJointClassRegr>(patchdata, patchlabel, sample_weight, datastore_id);
		dataset.AddLabelledSample(sample);

		if (label == 1)
			this->m_num_pos++;
		else
			this->m_num_neg++;
	}
}


void DataLoaderHoughObject::UpdateSampleWeights(DataSet<SampleImgPatch, LabelJointClassRegr>& dataset)
{
	// get some statistics
	int num_pos = 0, num_neg = 0;
	for (size_t i = 0; i < dataset.size(); i++)
	{
		if (dataset[i]->m_label.class_label == 1)
			num_pos++;
		else
			num_neg++;
	}
	for (size_t i = 0; i < dataset.size(); i++)
	{
		dataset[i]->m_label.patch_id = i;
		if (dataset[i]->m_label.class_label == 1)
		{
			dataset[i]->m_label.class_weight = 1.0 / (double)num_pos / (double)m_num_classes * (double)(num_pos+num_neg);
			dataset[i]->m_label.class_weight_gt = dataset[i]->m_label.class_weight;
			dataset[i]->m_label.regr_weight = 1.0;
			dataset[i]->m_label.regr_weight_gt = 1.0;

			dataset[i]->m_label.hough_map_patch.resize(this->m_num_z);
			for (size_t zz = 0; zz < this->m_num_z; zz++)
				dataset[i]->m_label.hough_map_patch[zz] = cv::Mat::zeros(dataset[i]->m_label.img_size(1), dataset[i]->m_label.img_size(0), CV_32F);

		}
		else
		{
			dataset[i]->m_label.class_weight = 1.0 / (double)num_neg / (double)m_num_classes * (double)(num_pos+num_neg);
			dataset[i]->m_label.class_weight_gt = dataset[i]->m_label.class_weight;
			dataset[i]->m_label.regr_weight = 1.0;
			dataset[i]->m_label.regr_weight_gt = 1.0;
		}
	}
}






