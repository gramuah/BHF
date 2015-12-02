/*
// Author: Juergen Gall, BIWI, ETH Zurich
// Email: gall@vision.ee.ethz.ch
*/

#include "HoughDetector.h"


// Constructors / Destructors
HoughDetector::HoughDetector(RandomForest<SampleImgPatch, LabelJointClassRegr, TSplitFunctionImgPatch, SplitEvaluatorJointClassRegr<SampleImgPatch, AppContextJointClassRegr>, LeafNodeStatisticsJointClassRegr<AppContextJointClassRegr>, AppContextJointClassRegr>* rfin, AppContextJointClassRegr* apphpin) : m_rf(rfin), m_apphp(apphpin)
{
	m_pwidth = m_apphp->patch_size[1];
	m_pheight = m_apphp->patch_size[0];
}




void HoughDetector::DetectList()
{
    int numCPU = sysconf( _SC_NPROCESSORS_ONLN );

    // Read the filenames of the test images
    std::vector<std::string> filenames;
    this->ReadTestimageList(m_apphp->path_testFilenames, filenames);

    // Average the bbox sizes from the training images (we need it also for the backprojection variant!)
    std::vector<cv::Size> avg_bbox_size;
    avg_bbox_size.resize(m_apphp->num_z);
    std::vector<VectorXd> trainPose;
    int NT;
    
    avg_bbox_size = AverageBboxsizeFromTrainingdata(trainPose, NT);
    for(size_t zz = 0; zz < m_apphp->num_z; zz++){
    	avg_bbox_size[zz].height *= m_apphp->avg_bbox_scaling[0];
    	avg_bbox_size[zz].width *= m_apphp->avg_bbox_scaling[1];
    }

    if (!m_apphp->quiet){
	for(size_t zz = 0; zz < m_apphp->num_z; zz++){
	        std::cout << "Estimated bounding box size: " << avg_bbox_size[zz].height << " x " << avg_bbox_size[zz].width << " (h x w)" << std::endl;
	}
    }
    // Run detector for each image
    int full_start = clock();

    #pragma omp parallel for
    for (size_t i = 0; i < filenames.size(); i++)
    {
        int tstart = clock();

        // Load image
        cv::Mat img_raw = DataLoaderHoughObject::ReadImageData(m_apphp->path_testImages + "/" + filenames[i]);
        // scaling
	cv::Mat img;
	cv::Size new_size = cv::Size((int)((double)img_raw.cols*this->m_apphp->general_scaling_factor), (int)((double)img_raw.rows*this->m_apphp->general_scaling_factor));
	resize(img_raw, img, new_size, 0, 0, cv::INTER_LINEAR);

        // Storage for Hough Maps
       std::vector<std::vector<cv::Mat> > hough_maps(m_apphp->test_scales.size());
       std::vector<std::vector<cv::MatND> > hough_pose_maps(m_apphp->test_scales.size());
       for (size_t zz = 0; zz < hough_maps.size(); zz++){
		hough_maps[zz].resize(m_apphp->num_z);
		hough_pose_maps[zz].resize(m_apphp->num_z);
       }

        // Storage for Backprojections
        std::vector<std::vector<std::vector<std::vector<Vote> > > > vBackprojections; // scales, height, width, votes
        if (m_apphp->backproj_bbox_estimation)
        {
        	std::cout << std::endl << "WARNING: include the genearl scaling factor here!!!!!" << std::endl << std::endl;
            vBackprojections.resize(m_apphp->test_scales.size());
        }
        // Voting in the Hough Maps -> this fills the Hough-maps + Backprojections
		this->DetectPyramid(img, hough_maps, hough_pose_maps, vBackprojections, cvRect(-1, -1, -1, -1), i);

        // Store Hough Maps to HDD
        if (m_apphp->print_hough_maps)
        {
            for (size_t k = 0; k < hough_maps.size(); k++)
            {
		for(size_t zz = 0; zz < hough_maps[k].size(); zz++){
                	cv::Mat tmp;
            		cv::convertScaleAbs(hough_maps[k][zz], tmp, m_apphp->houghmaps_outputscale);
            		tmp.convertTo(tmp, CV_8U);
                	stringstream buffer;
                	buffer << m_apphp->path_houghimages << "detect-" << i << "_sc" << k << "_c_" << zz << ".png";
                	cv::imwrite(buffer.str(), tmp);
		}
            }
        }

        // Hough Map Post-processing - NMS - find bounding boxes
		if (m_apphp->return_bboxes)
		{
			this->DetectBoundingBoxes(img, NT, hough_maps, hough_pose_maps, vBackprojections, avg_bbox_size, trainPose, i);
		}

        // Status message
        if (!m_apphp->quiet)
        {
            cout << "Testing image " << i + 1 << " / " << filenames.size() << ": \t";
            cout << filenames[i].c_str() << " ... in " << flush;
            cout << (double)(clock() - tstart)/CLOCKS_PER_SEC/(double)numCPU << " sec" << endl;
        }
    }
    if (!m_apphp->quiet)
        cout << "Full time: " << (double)(clock() - full_start)/CLOCKS_PER_SEC/(double)numCPU << " sec" << endl;
}


void HoughDetector::DetectPyramid(const cv::Mat img, std::vector<std::vector<cv::Mat> >& hough_maps, std::vector<std::vector<cv::MatND> >& hough_pose_maps,  std::vector<std::vector<std::vector<std::vector<Vote> > > >& vBackprojections, cv::Rect ROI, int index)

{
    if (img.channels() == 1)
    	throw std::logic_error("Gray color images are not supported");
    else
    {
        // iterate the scales
        for (size_t i = 0; i < hough_maps.size(); i++)
        {
		
        	cv::Mat cLevel;
            cv::Size scale_size(int(img.cols*m_apphp->test_scales[i]+0.5), int(img.rows*m_apphp->test_scales[i]+0.5));
        	cv::resize(img, cLevel, scale_size, 0.0, 0.0, cv::INTER_LINEAR);
            this->DetectImage(cLevel, hough_maps[i], hough_pose_maps[i], vBackprojections[i], ROI, index);
        }
    }
}


// ########################################################################
// PRIVATE HELPER METHODS

void HoughDetector::ReadTestimageList(std::string path_testimages, std::vector<std::string>& filenames)
{
	char buffer[400];
	std::ifstream in(path_testimages.c_str());
	if (in.is_open())
	{
		size_t size;
		in >> size;
		in.getline(buffer,400);
		filenames.resize(size);
		for (size_t i = 0; i < size; i++)
		{
			in.getline(buffer, 400);
			filenames[i] = buffer;
		}
		in.close();
	}
	else
		throw std::runtime_error("HoughDetector: Image file not found: " + path_testimages);
}


std::vector<cv::Size> HoughDetector::AverageBboxsizeFromTrainingdata(std::vector<VectorXd>& trainPose, int& NT)
{
    std::vector<cv::Size> avg_bbox_size;
    std::ifstream in(m_apphp->path_posAnnofile.c_str());
    if (in.is_open())
    {
        size_t size, numop;
        std::string dummy_string;
        int dummy_int, num_latent_variables, z_vble;
        cv::Rect tmprect;
        std::vector<double> mean_w;// = 0.0;
        std::vector<double> mean_h;// = 0.0;
	std::vector<int> count;// = 0.0;
        in >> size;
	NT = size;
        in >> numop;
	in >> num_latent_variables;
	trainPose.resize(size);
	avg_bbox_size.resize(num_latent_variables);
	mean_w.resize(num_latent_variables, 0.0);
	mean_h.resize(num_latent_variables, 0.0);
        count.resize(num_latent_variables, 0.0);
	for (size_t i = 0; i < size; i++)
        {
	    trainPose[i] = Eigen::VectorXd::Zero(2);
            in >> dummy_string;
            // Read bounding box
            in >> tmprect.x;
            in >> tmprect.y;
            in >> tmprect.width;
            tmprect.width -= tmprect.x;
            in >> tmprect.height;
            tmprect.height -= tmprect.y;
	    
            // scaling
            tmprect.x = (int)((double)tmprect.x * this->m_apphp->general_scaling_factor);
			tmprect.y = (int)((double)tmprect.y * this->m_apphp->general_scaling_factor);
			tmprect.width = (int)((double)tmprect.width * this->m_apphp->general_scaling_factor);
			tmprect.height = (int)((double)tmprect.height * this->m_apphp->general_scaling_factor);

            in >> dummy_int;
            in >> dummy_int;
	    in >> trainPose[i](0);
	    in >> trainPose[i](1);
	    in >> z_vble;
 	    count[z_vble - 1] += 1;
            mean_w[z_vble - 1] += (double)tmprect.width;
            mean_h[z_vble - 1] += (double)tmprect.height;
        }
	for(size_t zz = 0; zz < num_latent_variables; zz++){
        	avg_bbox_size[zz].width = int(mean_w[zz] / (double)count[zz]);
        	avg_bbox_size[zz].height = int(mean_h[zz] / (double)count[zz]);
	}
        in.close();
    }
    else
    	throw std::runtime_error("HoughDetector: could not open positive training annotation file to estimate the bounding box size");
    return avg_bbox_size;
}


void HoughDetector::DetectImage(const cv::Mat img, std::vector<cv::Mat>& hough_map, std::vector<cv::MatND>& hough_pose_map, std::vector<std::vector<std::vector<Vote> > >& backprojections, cv::Rect ROI, int index)
{
    // extract features
    std::vector<cv::Mat> img_features;
    DataLoaderHoughObject::ExtractFeatureChannelsObjectDetection(img, img_features, m_apphp);

    // reset output image
    for (int zz=0; zz < m_apphp->num_z; zz++){
        	hough_map[zz] = cv::Mat::zeros(img.rows, img.cols, CV_32F); // ! NOT img.type() !!!
    }


    // prepare the backprojections structure (pre-allocate the data structure)
    int bp_preallocsize = 50;
    std::vector<std::vector<int> > backprojections_cnt;
    if (m_apphp->backproj_bbox_estimation)
    {
    	backprojections.resize(img.rows);
        backprojections_cnt.resize(backprojections.size());
        for (int y = 0; y < backprojections.size(); y++)
        {
        	backprojections[y].resize(img.cols);
            backprojections_cnt[y].resize(backprojections[y].size());
            for (int x = 0; x < backprojections[y].size(); x++)
            {
                backprojections[y][x].resize(bp_preallocsize);
                backprojections_cnt[y][x] = 0;
            }
        }
    }



    // Define the search region
	int startX = (ROI.x == -1) ? 0 : ROI.x;
	int endX   = (ROI.x == -1) ? img.cols - m_pwidth : ROI.x + ROI.width - m_pwidth;
	int startY = (ROI.y == -1) ? 0 : ROI.y;
	int endY   = (ROI.y == -1) ? img.rows - m_pheight : ROI.y + ROI.height - m_pheight;

	// set the pixel stride
	int voting_pixel_step = m_apphp->voting_grid_distance + 1;

	// define the patch offsets
	int xoffset = m_pwidth/2;
	int yoffset = m_pheight/2;

    // iterate the image
	std::vector<Node<SampleImgPatch, LabelJointClassRegr, TSplitFunctionImgPatch,     LeafNodeStatisticsJointClassRegr<AppContextJointClassRegr>, AppContextJointClassRegr>* > leafnodes;
	SampleImgPatch imgpatch;
	imgpatch.features = img_features;
	if (m_apphp->split_function_type == SPLITFUNCTION_TYPE::HAAR_LIKE)
	{
		cv::Mat temp_mask;
		temp_mask.create(cv::Size(img.cols, img.rows), CV_8U);
		temp_mask.setTo(cv::Scalar::all(1.0));
		cv::integral(temp_mask, imgpatch.normalization_feature_mask, CV_32F);
	}

	// we only have to fill the Sample in the labelled sample for testing ... so we use a dummy label
	LabelJointClassRegr dummy_label = LabelJointClassRegr(1.0, 0, 1.0, 0.0, 0.0, Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2), 0, 0, Eigen::VectorXd::Zero(2), false);
	LabelledSample<SampleImgPatch, LabelJointClassRegr>* labelled_sample =
			new LabelledSample<SampleImgPatch, LabelJointClassRegr>(imgpatch, dummy_label, 1.0, 0);
	for (int y = startY; y < endY; y += voting_pixel_step)
	{
		for (int x = startX; x < endX; x += voting_pixel_step)
		{
			// set the patch location in the image
			labelled_sample->m_sample.x = x;
			labelled_sample->m_sample.y = y;

			// ask the random forest
			m_rf->Test(labelled_sample, leafnodes);

			// do the Hough voting
			for (size_t l = 0; l < leafnodes.size(); l++)
			{
				float w = 1.0 / float(leafnodes.size()*leafnodes[l]->m_leafstats->m_offsets[1].size());


				// TODO: include the multi-class stuff here!!!!
				if (leafnodes[l]->m_leafstats->m_votes.size() != 2)
					throw std::runtime_error("m_votes should be for 2 classes!");

				// we iterate the number of votes for the class 1 (i.e., the foreground class)
				
				for (size_t v = 0; v < leafnodes[l]->m_leafstats->m_offsets[1].size(); v++)
				{
					int vote_x = int(double(x + xoffset) + leafnodes[l]->m_leafstats->m_offsets[1][v](0));
					int vote_y = int(double(y + yoffset) + leafnodes[l]->m_leafstats->m_offsets[1][v](1));


					if (vote_y >= 0 && vote_y < hough_map[0].rows && vote_x >= 0 && vote_x < hough_map[0].cols)
					{

						hough_map[leafnodes[l]->m_leafstats->m_latent_label[1][v]-1].at<float>(vote_y, vote_x) += (float)(w * leafnodes[l]->m_leafstats->m_vote_weights[1][0]);//

						if (m_apphp->backproj_bbox_estimation)
						{
							backprojections[y][x][backprojections_cnt[y][x]].x = vote_x;
							backprojections[y][x][backprojections_cnt[y][x]].y = vote_y;
							backprojections[y][x][backprojections_cnt[y][x]].w = w * leafnodes[l]->m_leafstats->m_class_histogram[1];
							backprojections_cnt[y][x] += 1;
							if (backprojections_cnt[y][x] == backprojections[y][x].size())
								backprojections[y][x].resize(backprojections_cnt[y][x] + bp_preallocsize);
						}
					}
				}

			}
		}
	}

	// resize the backprojections structure to the really used size
	if (m_apphp->backproj_bbox_estimation)
	{
		for (int y = 0; y < backprojections.size(); y++)
		{
			for (int x = 0; x < backprojections[y].size(); x++)
			{
				backprojections[y][x].resize(backprojections_cnt[y][x]);
			}
		}
	}

	// smooth result image

	for (int zz=0; zz < m_apphp->num_z; zz++)
		cv::GaussianBlur(hough_map[zz], hough_map[zz], cv::Size(3, 3), 0.0, 0.0, cv::BORDER_DEFAULT);

	//cv::namedWindow("Hough Map", CV_WINDOW_AUTOSIZE );
	//cv::imshow("Hough Map", hough_map[0]);
	//cv::waitKey(0);

	// delete the labelled sample, this is important otherwise, all the image features won't be free'd!
	delete(labelled_sample);
}


void HoughDetector::DetectImagePose(const cv::Mat img, cv::Mat& hough_map, int detected_z, int cx, int cy, cv::Rect ROI)
{
    // extract features
    std::vector<cv::Mat> img_features;
    DataLoaderHoughObject::ExtractFeatureChannelsObjectDetection(img, img_features, m_apphp);

    // Define the search region
    int startX = (ROI.x == -1) ? 0 : ROI.x;
    int endX   = (ROI.x == -1) ? img.cols - m_pwidth : ROI.x + ROI.width - m_pwidth;
    int startY = (ROI.y == -1) ? 0 : ROI.y;
    int endY   = (ROI.y == -1) ? img.rows - m_pheight : ROI.y + ROI.height - m_pheight;
    
    // set the pixel stride
    int voting_pixel_step = m_apphp->voting_grid_distance + 1;

    // define the patch offsets
    int xoffset = m_pwidth/2;
    int yoffset = m_pheight/2;

    // iterate the image
	std::vector<Node<SampleImgPatch, LabelJointClassRegr, TSplitFunctionImgPatch, LeafNodeStatisticsJointClassRegr<AppContextJointClassRegr>, AppContextJointClassRegr>* > leafnodes;
	SampleImgPatch imgpatch;
	imgpatch.features = img_features;
	if (m_apphp->split_function_type == SPLITFUNCTION_TYPE::HAAR_LIKE)
	{
		cv::Mat temp_mask;
		temp_mask.create(cv::Size(img.cols, img.rows), CV_8U);
		temp_mask.setTo(cv::Scalar::all(1.0));
		cv::integral(temp_mask, imgpatch.normalization_feature_mask, CV_32F);
	}

	// we only have to fill the Sample in the labelled sample for testing ... so we use a dummy label
	LabelJointClassRegr dummy_label = LabelJointClassRegr(1.0, 0, 1.0, 0.0, 0.0, Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2), Eigen::VectorXd::Zero(2), 0, 0, Eigen::VectorXd::Zero(2), false);
	LabelledSample<SampleImgPatch, LabelJointClassRegr>* labelled_sample =
			new LabelledSample<SampleImgPatch, LabelJointClassRegr>(imgpatch, dummy_label, 1.0, 0);
	for (int y = startY; y < endY; y += voting_pixel_step)
	{
		for (int x = startX; x < endX; x += voting_pixel_step)
		{
			// set the patch location in the image
			labelled_sample->m_sample.x = x;
			labelled_sample->m_sample.y = y;

			// ask the random forest
			m_rf->Test(labelled_sample, leafnodes);

			// do the Hough voting
			for (size_t l = 0; l < leafnodes.size(); l++)
			{
				// TODO: include the multi-class stuff here!!!!
				if (leafnodes[l]->m_leafstats->m_votes.size() != 2)
					throw std::runtime_error("m_votes should be for 2 classes!");

				// we iterate the number of votes for the class 1 (i.e., the foreground class)
				
				for (size_t v = 0; v < leafnodes[l]->m_leafstats->m_offsets[1].size(); v++)
				{
					int vote_x = int(double(x + xoffset) + leafnodes[l]->m_leafstats->m_offsets[1][v](0));
					int vote_y = int(double(y + yoffset) + leafnodes[l]->m_leafstats->m_offsets[1][v](1));

					int zD = leafnodes[l]->m_leafstats->m_latent_label[1][v]-1;
				
					if ((zD == detected_z) && (vote_x == cx) && (vote_y == cy))
						hough_map.at<float>(0, (int)leafnodes[l]->m_leafstats->img_id[v]) += 1;
					
				}

			}
		}
	}



	// delete the labelled sample, this is important otherwise, all the image features won't be free'd!
	delete(labelled_sample);
}


void HoughDetector::DetectBoundingBoxes(cv::Mat img, int NT, vector<vector<cv::Mat> >& hough_maps, std::vector<std::vector<cv::MatND> >& hough_pose_maps, std::vector<std::vector<std::vector<std::vector<Vote> > > >& vBackprojections, std::vector<cv::Size> avg_bbox_size, std::vector<VectorXd> trainPose, int img_id)
{
 // 0) Some initial settings
    int avg_bbox_w = avg_bbox_size[0].width;
    int avg_bbox_h = avg_bbox_size[0].height;

    // 1) Gauss filter (neglect the inter-scale smoothing)
    double gauss_sigma = 3.0;
    double gauss_sigma_sq = 9.0; //pow(gauss_sigma, 2.0);
    int gauss_w = m_apphp->hough_gauss_kernel; // should be 5
    size_t reference_scale = 0;


    if (m_apphp->nms_type == NMS_TYPE::NMS_GALL)
    {
        for (size_t k = 0; k < hough_maps.size(); k++)
        {
	    for (size_t zz = 0; zz < hough_maps[k].size(); zz++){
            	// resize the image
            	if (reference_scale != k)
            	{
                	cv::resize(hough_maps[k][zz], hough_maps[k][zz], cv::Size(hough_maps[reference_scale][zz].cols, hough_maps[reference_scale][zz].rows), 0.0, 0.0, cv::INTER_LINEAR);
            	}

            	// smooth it
            	cv::GaussianBlur(hough_maps[k][zz], hough_maps[k][zz], cv::Size(5, 5), 0.0, 0.0, cv::BORDER_DEFAULT);

            }
	}
    }
    else
    {
        for (size_t k = 0; k < hough_maps.size(); k++)
        {
		for (size_t zz = 0; zz < hough_maps[k].size(); zz++)
	        {
	        	if (m_apphp->nms_type == NMS_TYPE::NMS_ADAPT_KERNEL)
                        {
		                gauss_w = int((double)m_apphp->hough_gauss_kernel * m_apphp->test_scales[k] / 2.0) * 2 + 1;
		            	cv::GaussianBlur(hough_maps[k][zz], hough_maps[k][zz], cv::Size(gauss_w, gauss_w), 0.0, 0.0, cv::BORDER_DEFAULT);
		        }
        	        else
        	        {
				cv::GaussianBlur(hough_maps[k][zz], hough_maps[k][zz], cv::Size(5, 5), 0.0, 0.0, cv::BORDER_DEFAULT);


        	        }
		}
        }
    }


    // 2) Iteratively find modes (find mode + clear region in Hough image)
    int num_detections = 0;
    double detection_peak_height = 100.0;
    std::vector<Eigen::VectorXd> detection_boxes;
    std::vector<Eigen::VectorXd> detection_centers;
    std::vector<Eigen::VectorXd> detection_pose;
    std::vector<std::vector<Vote> > detection_backprojections;
    while (num_detections < m_apphp->max_detections_per_image && detection_peak_height > m_apphp->min_detection_peak_height)
    {
        // Find current max value + location
        cv::Point max_loc_tmp;
        cv::Point min_loc_tmp;
        double min_val_tmp;
        double max_val_tmp;
        size_t max_scale = 0;
	int max_z = 0;
        cv::Point max_loc;
	double cx, cy;
        cv::minMaxLoc(hough_maps[0][0], &min_val_tmp, &detection_peak_height, &min_loc_tmp, &max_loc);
        for (size_t zz = 1; zz < hough_maps[0].size(); zz++)
        {
        	cv::minMaxLoc(hough_maps[0][zz], &min_val_tmp, &max_val_tmp, &min_loc_tmp, &max_loc_tmp);
            	if (max_val_tmp > detection_peak_height)
            	{
                	detection_peak_height = max_val_tmp;
                	max_loc = max_loc_tmp;
                	max_scale = 0;
			max_z = zz;
            	}
        }
	
	for (size_t k = 1; k < hough_maps.size(); k++)
        {
		for (size_t zz = 0; zz < hough_maps[k].size(); zz++)
        	{
           		cv::minMaxLoc(hough_maps[k][zz], &min_val_tmp, &max_val_tmp, &min_loc_tmp, &max_loc_tmp);
            		if (max_val_tmp > detection_peak_height)
            		{

                		detection_peak_height = max_val_tmp;
                		max_loc = max_loc_tmp;
                		max_scale = k;
				max_z = zz;
            		}
        	}
	}
	
        // for the NMS_GALL case, we have to scale the max-locations from the reference scale to the detected scale!
        double interp_scale = m_apphp->test_scales[max_scale];
        if (m_apphp->nms_type == NMS_TYPE::NMS_GALL)
        {
            // 1) calc the interpolation - diff scale
            if (m_apphp->use_scale_interpolation)
            {
                if (max_scale > 0 && max_scale < (hough_maps.size()-1))
                {
                    Eigen::VectorXd interpvals = VectorXd::Zero(3);
                    interpvals(0) = (double)hough_maps[max_scale-1][max_z].at<float>(max_loc.y, max_loc.x);
					interpvals(1) = (double)hough_maps[max_scale][max_z].at<float>(max_loc.y, max_loc.x);
					interpvals(2) = (double)hough_maps[max_scale+1][max_z].at<float>(max_loc.y, max_loc.x);
                    if (interpvals(0) < interpvals(2))
                    {
                        double r = (interpvals(2)-interpvals(0))/((interpvals(1)-interpvals(0))+(interpvals(2)-interpvals(0)));
                        double scale_1 = m_apphp->test_scales[max_scale], scale_2 = m_apphp->test_scales[max_scale+1];
                        interp_scale = scale_1 + (scale_2-scale_1) * r;
                    }
                    else
                    {
                        double r = (interpvals(0)-interpvals(2))/((interpvals(1)-interpvals(2))+(interpvals(0)-interpvals(2)));
                        double scale_1 = m_apphp->test_scales[max_scale], scale_2 = m_apphp->test_scales[max_scale-1];
                        interp_scale = scale_1 + (scale_2-scale_1) * r;
                    }
                }
            }

            // 2) scaling from reference scale to detected scale. Not the original image scale!!
            max_loc.x = int((double)max_loc.x * m_apphp->test_scales[max_scale] / m_apphp->test_scales[reference_scale]);
            max_loc.y = int((double)max_loc.y * m_apphp->test_scales[max_scale] / m_apphp->test_scales[reference_scale]);
	    cx = int(max_loc.x);
	    cy = int(max_loc.y);
        }


        // Store this location as a bounding box estimate
        Eigen::VectorXd current_detection = Eigen::VectorXd::Zero(5);
 	Eigen::VectorXd current_center = Eigen::VectorXd::Zero(2);
	Eigen::VectorXd current_pose = Eigen::VectorXd::Zero(3);
        Eigen::VectorXd remove_region = Eigen::VectorXd::Zero(4);
        if (m_apphp->backproj_bbox_estimation)
        {
            // initial stuff for bbox estimation
            Eigen::VectorXd backproj_hist_x = Eigen::VectorXd::Zero(avg_bbox_w * 2);
            Eigen::VectorXd backproj_hist_y = Eigen::VectorXd::Zero(avg_bbox_h * 2);
            int backproj_hist_x_offset = int(avg_bbox_w);
            int backproj_hist_y_offset = int(avg_bbox_h);

            // (i) accumulate all backprojections for the current detection
            // (ii) re-estimate a new bounding box
            std::vector<Vote> tmp_back_proj_vector;
            int min_x = 100000, max_x = -1, min_y = 100000, max_y = -1; // TODO: I guess no image will be larger than 100000 pixel?!?
            int backprojection_kernel_w_half = int(m_apphp->backproj_bbox_kernel_w / 2);
            for (int vy = -backprojection_kernel_w_half; vy <= backprojection_kernel_w_half; vy++)
            {
                for (int vx = -backprojection_kernel_w_half; vx <= backprojection_kernel_w_half; vx++)
                {
                    int c_vote_y = max_loc.y + vy;
                    int c_vote_x = max_loc.x + vx;
                    if (c_vote_x < 0 || c_vote_y < 0 || c_vote_x >= vBackprojections[max_scale][0].size() || c_vote_y >= vBackprojections[max_scale].size())
                        continue;

                    // calculate the real distance to the center -> if this is to large -> continue
                    Eigen::VectorXd vec_cpos = Eigen::VectorXd::Zero(2);
                    vec_cpos(0) = (double)c_vote_x;
                    vec_cpos(1) = (double)c_vote_y;
                    Eigen::VectorXd vec_center = Eigen::VectorXd::Zero(2);
                    vec_center(0) = (double)max_loc.x;
                    vec_center(1) = (double)max_loc.y;
                    double center_dist = sqrt((vec_cpos-vec_center).dot(vec_cpos-vec_center));
                    if (center_dist > ((double)m_apphp->backproj_bbox_kernel_w / 2.0))
                        continue;

                    // now iterate all backprojections for the current pixel (inside the kernel width)
                    for (unsigned int v = 0; v < vBackprojections[max_scale][c_vote_y][c_vote_x].size(); v++)
                    {
                    	Eigen::VectorXd tmpPred = Eigen::VectorXd::Zero(2);
                        tmpPred(0) = double(c_vote_x + vBackprojections[max_scale][c_vote_y][c_vote_x][v].x) / double(m_apphp->test_scales[max_scale]);
                        tmpPred(1) = double(c_vote_y + vBackprojections[max_scale][c_vote_y][c_vote_x][v].y) / double(m_apphp->test_scales[max_scale]);
                        tmp_back_proj_vector.push_back(Vote(tmpPred, vBackprojections[max_scale][c_vote_y][c_vote_x][v].w));

                        int diff_x = c_vote_x + vBackprojections[max_scale][c_vote_y][c_vote_x][v].x - max_loc.x;
                        backproj_hist_x(backproj_hist_x_offset + diff_x) += vBackprojections[max_scale][c_vote_y][c_vote_x][v].w;
                        int diff_y = c_vote_y + vBackprojections[max_scale][c_vote_y][c_vote_x][v].y - max_loc.y;
                        backproj_hist_y(backproj_hist_y_offset + diff_y) += vBackprojections[max_scale][c_vote_y][c_vote_x][v].w;
                    }
                }
            }

            // normalize the
            backproj_hist_x = backproj_hist_x / backproj_hist_x.sum();
            backproj_hist_y = backproj_hist_y / backproj_hist_y.sum();

            double cumsum = 0.0;
            min_x = -1;
            max_x = -1;
            min_y = -1;
            max_y = -1;
            for (size_t cx = 0; cx < backproj_hist_x.size(); cx++)
            {
                cumsum += backproj_hist_x(cx);
                if (min_x == -1 && cumsum > m_apphp->backproj_bbox_cumsum_th)
                    min_x = int(double(cx - backproj_hist_x_offset + max_loc.x) / double(m_apphp->test_scales[max_scale]));
                if (max_x == -1 && cumsum > (1.0-m_apphp->backproj_bbox_cumsum_th))
                    max_x = int(double(cx - backproj_hist_x_offset + max_loc.x) / double(m_apphp->test_scales[max_scale]));
            }
            cumsum = 0.0;
            for (size_t cy = 0; cy < backproj_hist_y.size(); cy++)
            {
                cumsum += backproj_hist_y(cy);
                if (min_y == -1 && cumsum > m_apphp->backproj_bbox_cumsum_th)
                    min_y = int(double(cy - backproj_hist_y_offset + max_loc.y) / double(m_apphp->test_scales[max_scale]));
                if (max_y == -1 && cumsum > (1.0-m_apphp->backproj_bbox_cumsum_th))
                    max_y = int(double(cy - backproj_hist_y_offset + max_loc.y) / double(m_apphp->test_scales[max_scale]));
            }

            detection_backprojections.push_back(tmp_back_proj_vector);
            // Set the detection, including a clipping of bounding boxes to the image boundaries
            current_detection(0) = (double)max(min_x, 0);
            current_detection(1) = (double)max(min_y, 0);
            current_detection(2) = (double)min(max_x, img.cols-1);
            current_detection(3) = (double)min(max_y, img.rows-1);
            current_detection(2) = current_detection(2) - current_detection(0);
            current_detection(3) = current_detection(3) - current_detection(1);
            current_detection(4) = detection_peak_height;
            detection_boxes.push_back(current_detection);

            remove_region(0) = current_detection(0);
            remove_region(1) = current_detection(1);
            remove_region(2) = current_detection(2);
            remove_region(3) = current_detection(3);
        }
        else
        {
            // -> with the new interp_scale
	    double center_x = (double)max_loc.x / (double)m_apphp->test_scales[max_scale];
	    double center_y = (double)max_loc.y / (double)m_apphp->test_scales[max_scale];
	    current_center(0) = center_x;
            current_center(1) = center_y;
	    detection_centers.push_back(current_center);
            double min_x_bbox = (double)max_loc.x / (double)m_apphp->test_scales[max_scale] - (double)avg_bbox_size[max_z].width / 2.0 / interp_scale;
            double min_y_bbox = (double)max_loc.y / (double)m_apphp->test_scales[max_scale] - (double)avg_bbox_size[max_z].height / 2.0 / interp_scale;
            double max_x_bbox = min_x_bbox + (double)avg_bbox_size[max_z].width / interp_scale;
            double max_y_bbox = min_y_bbox + (double)avg_bbox_size[max_z].height / interp_scale;
            // [x, y, w, h]
            current_detection(0) = (double)max((int)min_x_bbox, 0);
            current_detection(1) = (double)max((int)min_y_bbox, 0);
            current_detection(2) = (double)min((int)max_x_bbox, img.cols-1);
            current_detection(3) = (double)min((int)max_y_bbox, img.rows-1);
            current_detection(2) = current_detection(2) - current_detection(0);
            current_detection(3) = current_detection(3) - current_detection(1);
            current_detection(4) = detection_peak_height;
            detection_boxes.push_back(current_detection);

            // set the remove region
            remove_region = current_detection;
        }
	cv::Rect patch_roi = cv::Rect(current_detection(0), current_detection(1), current_detection(2), current_detection(3));
	cv::Mat roiImg(img, patch_roi);
	cv::Mat cLevel;
        cv::Size scale_size(int(roiImg.cols*m_apphp->test_scales[max_scale]+0.5), int(roiImg.rows*m_apphp->test_scales[max_scale]+0.5));
      	cv::resize(roiImg, cLevel, scale_size, 0.0, 0.0, cv::INTER_LINEAR);
	cv::Mat poseVotes = cv::Mat::zeros(1, NT, CV_32F);
	
	cx = (max_loc.x - (current_detection(0) * m_apphp->test_scales[max_scale]));
	cy = (max_loc.y - (current_detection(1) * m_apphp->test_scales[max_scale]));
	
	this->DetectImagePose(cLevel, poseVotes, max_z, int(cx), int(cy), cvRect(-1, -1, -1, -1));

	cv::Point max_loc_tmp_pose;
	cv::Point min_loc_tmp_pose;
	double min_val_tmp_pose;
	double max_val_tmp_pose;
	cv::minMaxLoc(poseVotes, &min_val_tmp_pose, &max_val_tmp_pose, &min_loc_tmp_pose, &max_loc_tmp_pose);
	int imgID = max_loc_tmp_pose.x;
	current_pose(0) = (double)trainPose[imgID](0);
	current_pose(1) = (double)trainPose[imgID](1);
	current_pose(2) = imgID;
	
	detection_pose.push_back(current_pose);

        // Remove those regions in the Hough Map
        for (size_t k = 0; k < hough_maps.size(); k++)
        {
            int cx = remove_region(0) * m_apphp->test_scales[k];
            int cy = remove_region(1) * m_apphp->test_scales[k];
            int cw = remove_region(2) * m_apphp->test_scales[k];
            int ch = remove_region(3) * m_apphp->test_scales[k];
            // for the NMS_GALL type we have to rescale the positions to the reference scale!!!
            if (m_apphp->nms_type == NMS_TYPE::NMS_GALL)
            {
                cx = remove_region(0) * m_apphp->test_scales[reference_scale];
                cy = remove_region(1) * m_apphp->test_scales[reference_scale];
                cw = remove_region(2) * m_apphp->test_scales[reference_scale];
                ch = remove_region(3) * m_apphp->test_scales[reference_scale];
            }
	    for (int zz = 0; zz < m_apphp->num_z; zz++){
            	cv::Mat roi(hough_maps[k][zz], cv::Rect(cx, cy, cw, ch));
            	roi.setTo(cv::Scalar::all(0.0));}
        }

        // increase the detections counter
        num_detections++;
    }


    // store the detection images
    if (m_apphp->print_detection_images > 0)
    {
        int num_print_detections = min(m_apphp->print_detection_images, (int)detection_boxes.size());
        for (int d = 0; d < num_print_detections; d++)
        {
            cv::Point tl((int)detection_boxes[d](0), (int)detection_boxes[d](1));
            cv::Point br((int)detection_boxes[d](0)+(int)detection_boxes[d](2), (int)detection_boxes[d](1)+(int)detection_boxes[d](3));
            cv::rectangle(img, tl, br, cvScalar(200.0, 10.0, 10.0, 0.0), 2, 8, 0);
            if (m_apphp->backproj_bbox_estimation)
            {
                for (int bv = 0; bv < detection_backprojections[d].size(); bv++)
                {
                    cv::Point bvtl((int)detection_backprojections[d][bv].x, (int)detection_backprojections[d][bv].y);
                    cv::Point bvbr((int)detection_backprojections[d][bv].x+1, (int)detection_backprojections[d][bv].y+1);
                    cv::rectangle(img, bvtl, bvbr, cvScalar(10.0, 200.0, 10.0, 0.0), 1, 8, 0);
                }
            }
        }
        stringstream buffer;
        buffer << m_apphp->path_detectionimages << "detect-" << img_id << ".png";
        cv::imwrite(buffer.str(), img);
    }

    // scale the bounding boxes back to the original scale! (this has to be done after drawing the imgages with
    // the bounding boxes because the images are also scaled with the generl_scaling_factor)
	for (size_t i = 0; i < detection_boxes.size(); i++)
	{
		double detection_conf = detection_boxes[i](4);
		detection_boxes[i] /= this->m_apphp->general_scaling_factor;
		detection_boxes[i](4) = detection_conf;
	}

    // 3) Write the detection bounding boxes to HDD
    stringstream bbox_filename;
    bbox_filename << m_apphp->path_bboxes << "bboxes_testimg_" << img_id << ".txt";
    std::ofstream file(bbox_filename.str().c_str());
    file << detection_boxes.size() << endl;
    for (size_t d = 0; d < detection_boxes.size(); d++)
    {
        for (size_t j = 0; j < detection_boxes[d].size(); j++)
            file << detection_boxes[d](j) << " ";
        file << endl;
    }
    file.close();


    stringstream pose_filename;
    pose_filename << m_apphp->path_bboxes << "pose_testimg_" << img_id << ".txt";
    std::ofstream file_1(pose_filename.str().c_str());
    file_1 << detection_pose.size() << endl;
    for (size_t d = 0; d < detection_pose.size(); d++)
    {
        for (size_t j = 0; j < detection_pose[d].size(); j++)
            file_1 << detection_pose[d](j) << " ";
        file_1 << endl;
    }
    file_1.close();
}
