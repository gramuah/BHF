# BHF - Boosted Hough Forest

#### REQUIREMENTS:


The code is developed and tested under Ubuntu 12.04 and the following libraries are 
required to build the code:
+ gcc
+ cmake
+ libEigen3
+ openmp
+ libconfig++
+ libblitz

##### For Microsoft Windows users:
    
Adapt the corresponding parts in the CMakeLists.txt files in:
            
      "ADF_ARF_Public/src/apps/adf_classification/"
      "ADF_ARF_Public/src/apps/arf_regression/"

---

#### HOW TO BUILD THE CODE:


In general, this code package includes two different applications, i.e., two different 
binaries. One is for classification tasks with ADF and the other for regression tasks with 
ARF. 

To build the code, change the directory to 

```
    ADF_ARF_Public/bin/*/build
```
where "\*" is either "adf_classification" or "arf_regression". 

Then, simply type:
 
```
    cmake ../../../src/apps/*
```
to create the Makefile. Again "\*" is either "adf_classification" or "arf_regression". 
Finally, type "make" to build the binaries. 

---

#### HOW TO USE THE BINARIES:


Call:

```
    ./icgrf-ml <path-to-configfile>
```

The config file defines all the settings for the Random Forest and the Alternating Decision/Regression 
Forest. 

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
CHANGE THE DIRECTORIES!!!

---

#### HOW TO USE THE CODE:
The Random Forest (RF) core is a generic implementation with templates (see folder ../../rflib/)
On the application site, you have to write the main.cpp code, i.e., the work flow of your application
and how the data is read.

Then, you can instantiate a RF and an ADF with templates:
Sample, Label, SplitFunction, SplitEvaluator, LeafNodeStatistics and AppContext
+ The Sample class defines how your data looks like
+ The Label class defines how the labels to your data look like
+ The SplitFunction class knows how to make splits with the given data
+ The SplitEvaluator class knows how to evaluate a potential split
+ The LeafNodeStatistics class has to aggregate all the incoming data (Sample and Label)
     in a leaf node, i.e., it defines the prediction of a leaf node.
Which methods these classes have to implement can be seen in the interfaces.h file (../../rflib)

Finally, you have to write code for reading data (e.g., from HDD) and fill the
DataSet<Sample, Label> objects
