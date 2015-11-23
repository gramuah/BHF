/*
 * AppContextML.h
 *
 * Author: Samuel Schulter, Paul Wohlhart, Christian Leistner, Amir Saffari, Peter M. Roth, Horst Bischof
 * Institution: Graz, University of Technology, Austria
 */

#ifndef APPCONTEXTML_H_
#define APPCONTEXTML_H_

#include <ostream>
#include <string>
#include <vector>
#include <libconfig.h++>
#include "icgrf.h"

#include "AppContext.h"

using std::cout;
using std::endl;


class AppContextML : public AppContext
{
public:
	
	AppContextML(int in_pred_type)
	{
		this->prediction_type = in_pred_type;
	}
	virtual ~AppContextML() {}

protected:

	inline void ValidateHyperparameters()
	{
		// validate the general settings in the config file
		if (this->method == RF_METHOD::NOTSET || this->path_output.empty() || this->debug_on == -1 ||
				this->quiet == -1)
		{
			cout << "Some general settings missing!" << endl;
			exit(-1);
		}

		if (!AppContext::ValidateStandardForestSettings())
		{
			cout << "General forest settings wrong!" << endl;
			exit(-1);
		}

		if (this->prediction_type == 0)
		{
			if (this->splitevaluation_type_classification == SPLITEVALUATION_TYPE_CLASSIFICATION::NOTSET)
			{
				cout << "specify a classification splitevaluation type!" << endl;
				exit(-1);
			}
		}
		else if (this->prediction_type == 1)
		{
			if (this->splitevaluation_type_regression == SPLITEVALUATION_TYPE_REGRESSION::NOTSET)
			{
				cout << "specify a classification splitevaluation type!" << endl;
				exit(-1);
			}
		}

		if (this->prediction_type == 1)
		{
			if (this->leafnode_regression_type == LEAFNODE_REGRESSION_TYPE::NOTSET)
			{
				cout << "specify a leafnode regression type!" << endl;
				exit(-1);
			}
		}

		if (this->method != RF_METHOD::STDRF && this->method != RF_METHOD::BOOSTINGTREES &&
			this->method != RF_METHOD::ADF && this->method != RF_METHOD::ONLINERF &&
			this->method != RF_METHOD::ONLINEADF && this->method != RF_METHOD::NGRF)
		{
			cout << "Only Standard-RF (0), BoostingTrees (1) and ADF (2) are allowed!" << endl;
			exit(-1);
		}

		if (this->method == RF_METHOD::ADF)
		{
			if (this->prediction_type == 0)
			{
				if (!AppContext::ValidateADFParameters())
				{
					cout << "ADF parameters have to be specified!" << endl;
					exit(-1);
				}
			}
			else
			{
				if (!AppContext::ValidateARFParameters())
				{
					cout << "ARF parameters have to be specified!" << endl;
					exit(-1);
				}
			}
		}

		if (!AppContext::ValidateMLDataset())
		{
			cout << "Dataset parameters are missing!" << endl;
			exit(-1);
		}

		if (this->path_sampleweight_progress.empty())
		{
			cout << "Specify a path for storing the weight evoluation!" << endl;
			exit(-1);
		}
	}

	// some member variables
	int prediction_type;

};



#endif /* APPCONTEXTML_H_ */
