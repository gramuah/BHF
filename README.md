# BHF - Boosted Hough Forest

#### REQUIREMENTS:


The BHF code is developed and tested under Ubuntu 14.04 and the following libraries are 
required to build the code:
+ gcc
+ cmake
+ libEigen3
+ openmp
+ libconfig++
+ libblitz

---

#### BUILDING THE CODE:

To build the code, just follow these instructions:

```Shell
    cd BHF/bin/houghdetection/build
    cmake ../../../src/apps/*
    make
```

---

#### HOW TO USE THE BINARIES:


Call:

```
    ./boostedHoughtForests <path-to-configfile>
```
Where the path to configuration file is located at "../configs/std.txt"

The config file defines all the settings for the Boosted Hought Forest. 

In general, the different parameters are documented directly in the config file. 

The data sets used are also specified in the config file. Please note that for data sets having only 
a training set, the data is randomly split with a certain ratio (also defined in the config file). 
This certain split can be stored and again loaded. So, when comparing different methods, you should 
always use the same random split for a fair comparison. To do so, simply save the random split for the 
first method to be evaluated (specified in the config file, save=1, load=0). For all other methods, only load the pre-defined data split (save=0, load=1). 

Example config files are provided in the corresponding folders, i.e., 
```
    "bin/*/configs/std.txt"
```
CHECK THE DIRECTORIES!!!

---

#### HOW TO CHECK RESULTS:
There is another folder called "BHF/scipts_results" which contains the script to plot the three graphics writen in Matlab.

To check your results with this graphics simply run the script "eval_results.m" giving the folder where bboxes are contained. 

For example,
```
	eval_results('../bindata');
```
---

#### CITING

If you make use of this data and software, please cite the following reference in any publications:

    @InProceedings{Redondo-CabreraBMVC 2015,
        Title                    = {Because better detections are still possible: Multi-aspect Object Detection with Boosted Hough Forest},
        Author                   = {Carolina Redondo-Cabrera and Roberto Javier López-Sastre},
        Booktitle                = {BMVC},
        Year                     = {2015}
    }
