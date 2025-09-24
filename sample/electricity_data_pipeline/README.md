# Electricity

### Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Model](#model)

## Introduction

This directory contains data sourcing scripts for multiple input files used within the electricity model. This project is under development, but conceptully the goal is to create a pipeline that takes regionally disaggregated county-level data and builds the data up to user-specified regions used within the electricity model. 

## Setup

Follow the instructions within the main repository ReadMe to setup your python environment. 

Be sure to have the 'bluesky' environment activated in a terminal, navigate to the 'sample\electricity_data_pipeline' directory in your terminal, and then run the following code:

```bash
python main.py --help
```

This should return a description of ```main.py```, as well as the options available within the command line interface. Now, check to see if we can run and solve a model by executing:


```bash
python main.py 
```

There are additional argument options you can pass when you run the script including **--data**, **--debug**, and **--output**.

#### --data
The data option allows for users to specify which data files they want to generate. Because this pipeline produces multiple output files, this option allows the flexibility to select outputs of interest. The data options can be easily expanded for future versions of this project, but are currently defined as: 
 - vre_capfactor
 - supplycurve 
 - transmission 
 - other_data 

When executing this code, if you do not utilize the data flag, the project will simply run the default settings specified in the `data_config.toml` file within the "executions" dictionary.   

#### --debug
The debug option allows for more details to be written to the log file. 

#### --output
The output option allows for the user to specify the stored location where the output files are saved. By default, the output files are saved within the *output* subdirectory.


### Configuration Settings
The configurations are specified within `data_config.toml` within the *settings* subdirectory. 

#### Executions
The executions dictionary contains the default selection for which methods to execute when running the project. The methods listed within the dictionary include:
 - vre_capfactor
 - supplycurve 
 - transmission 
 - other_data 

As described in [Setup](#setup), these default options can be overridden with the 
***--data*** flag.

#### Other Settings 
The first and last year settings define the range of years for input data. The population year specified the year to base the population weighting on. And the EIA860m_excel_name specified which [Form-EIA 860M](https://www.eia.gov/electricity/data/eia860m/) excel file base the supply curve off of. The EIA-860M is EIA survey data that collects monthly inventories of grid-connected electric generators greater than 1 MW in size. 

## Model

The model is currently divided into four main sections, vre_capfactor, supplycurve, transmission, and other_data. All relavent code is currently stored in `runner.py` within the *src* subdirectory. Future versions of this project will likely break this code out into separate files.

### vre_capfactor
This method generates the `CapFactorVRE.csv` input file used in the prototype within the electricity model. This file contains the hourly capacity factors by model region for onshore wind, offshore wind, and solar photovoltaic (PV). Future project development might instead use the weather database produced within the *sample/baseload_data_pipeline* subdirectory to produce the capacity factor profiles for these technologies. 

Hourly profiles for each technology type are developed for each county. The steps for creating the county-level profiles are described below. Once the county-level data is prepared, the `cw_r.csv` input file is used to crosswalk the county profiles to regional profiles where the profiles are averaged and then the three datasets are concatenated to create the final model input file. 

#### onshore wind
The source data for the onshore wind resource profiles comes from the Renewable Energy Laboratory (NREL) [United States Land-based Wind Supply Curves 2023](https://data.openei.org/submissions/6119) data files. We selected the Reference Access 2030 Moderate case with representative turbines with a hub-height of 115 meters and a rotor diameter of 170 meters. 

We download the both the supply curve data and the hourly representative profiles. We do not use the data from the supply curves directly in the model, but we download the file to select the median wind profile (sc_point_gid) for each county.  We then extract the hourly profile data for the selected points/counties and reformat the data. 

#### offshore wind
This is an area of future development. We currently do not have a method for county-level hourly offshore wind profiles. Instead, we have a placeholder function that takes the regional hourly profiles from the `CapFactorVRE_wnoff.csv` input file, assigns them to counties using the `cw_r.csv` (which assigns regions to counties) and `cw_wnoff.csv` (which identifies offshore wind eligible counties) files, and then later, reaggregates them back to the original regional definitions first passed in.  

#### solar PV
The source data for the solar PV profiles comes from NREL. Here we use the System Advisory Model (SAM) to develop representative hourly capacity factor profiles for 470 individual sites wtihin the `pvcf.csv` input file. We then map those sites to counties using the `cw_pv.csv` input file. 

Solar PV is broken out by the supply steps into utility-scale (step 1) and distributed generation (step 2). We assume the same capacity factor profiles for both technologies at the moment. 

### supplycurve 
This method generates the `SupplyCurve.csv` input file used in the prototype within the electricity model. This file contains the annual expected baseline capacity for each technology type. It reflects the current operating fleet, plus expected additions, and minus expected retirements. 

This method starts by reading in the [Form-EIA 860M](https://www.eia.gov/electricity/data/eia860m/) survey data. The `may_generator2024.xlsx` file is already included within the inputs, but users could download the latest version of the survey data and update the file name specified in the `data_config.toml` as described in the [Setup](#setup) section. Both the *Planned* and *Operating* data tables are used in the supply curve. 

Next, we create a county-level supply curve dataset. 
 - We use the `cw_tech.csv`, `cw_county.csv`, and `cw_status.csv` input files to reconfigure the raw input dataset for use within the prototype. 
 - We use the county list from `cw_r.csv` and the technology list from `cw_tech.csv` and the year range specified in the `data_config.toml` settings to create an index.
 - We create cumulative sums of current and planned capacity and then subtract cumulative sums of planned retired capacity to determine the baseline amount of operating capacity in each year. 
 - Lastly, we share out the operating capacity in each year into supply curve steps and then sum the data up from the unit-level to the county-level. 

The last step is to use the `cw_r.csv` to aggregate the data up from county level to the user-specified regional level. 

### transmission 
This method generates the international transmission input files, `TranLimitCapInt.csv` and `TranCostInt.csv`, used in the prototype within the electricity model. 

The `TranLineLimitCan_fips.csv` and `TranCostCan_fips.csv` input files are county-level levels and costs for transmission. We read those in and then use `cw_r.csv` to aggregate the data up from county level to the user-specified regional level. The values for these datasets are estimates based on NEMS. Future work on this project would replace these input datasets with survey data. 

### other_data 

Other data is a place holder for future work in this project. These datasets have been identified as also being regional input datasets for the electricity model. Inputs for this data are storaged at the EMM regional level, disaggregated to the county-level, and then reaggreated to the user-specified regional level. Future work would replace these datasets with county-level estimates for each input type. 

The values for these datasets are estimates based on NEMS. Future work on this project would replace these input datasets with survey data.