# BHF - Boosted Hough Forest

#### INTRODUCTION:

Boosted Hough Forest (BHF) is a framework for object detection and pose estimation with Hough Forests.

This is a repository with an implementation of the BHF model described in our BMVC 2015 paper.

We provide the codes and data needed to reproduce the experiments with the [Weizmann Cars ViewPoint dataset](http://www.wisdom.weizmann.ac.il/~vision/WCVP/index.html), which is included in the repository.


#### CITING

If you make use of this data and software, please cite the following references in any publications:


	@inproceedings{Redondo-Cabrera2015,
        Title                    = {Because better detections are still possible: Multi-aspect Object Detection with Boosted Hough Forest},
        Author                   = {Redondo-Cabrera, C. and Lopez-Sastre, R.~J.},
        Booktitle                = {BMVC},
        Year                     = {2015}
	}

	@inproceedings{Schulter13,
        Title                    = {Alternating Regression Forests for Object Detection and Pose Estimation},
        Author                   = {Samuel Schulter and Christian Leistner and Paul Wohlhart and Peter M. Roth and Horst Bischof},
        Booktitle                = {ICCV},
        Year                     = {2013}
	}	
	
	@article{Glasner2012,
	Title			= {Viewpoint-aware object detection and continuous pose estimation},
	Author                  = {Glasner, D. and Galun, M. and Alpert, S. and Basri, R. and Shakhnarovich, G.},
	Journal                 = {Image and Vision Computing},
	Year                    = {2012},
	Number                  = {12},
	Pages                   = {923--933},
	Volume                  = {30}
	}



#### REQUIREMENTS:

The BHF code is developed and tested under Ubuntu 14.04 and the following libraries are 
required to build the code:
+ gcc
+ cmake
+ libEigen3
+ openmp
+ libconfig++
+ libblitz
+ libboost

---

#### BUILDING THE CODE:

To build the code, just follow these instructions:

```Shell
    cd BHF/bin/houghdetection
    mkdir build
    cd build
    cmake ../../../src/apps/*
    make
```

---

#### HOW TO RUN THE BHF:

We use config files to communicate with the BHF. These config files are located at "../configs".

It is very simple to work with them, the different parameters are documented directly in the config file.

In order to reproduce the experiments in our paper these are the steps to follow:

+  Train and test the BHF model.

```Shell
   ./boostedHoughtForests ../configs/std.txt
```
+ To visualize the precision and recall curves run the following Matlab script located in "BHF/scripts_results":

```Shell
   cd BHF/scripts_results
   matlab #this opens matlab  
   eval_results('../bin/houghdetection/bindata') #use the bindata directory where the results have been saved
```

---
