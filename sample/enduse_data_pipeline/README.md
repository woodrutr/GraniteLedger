# Residential (End Use Data)

### Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Model](#model)

## Introduction

This directory contains data sourcing scripts for the `EnduseShapes.csv` input file used within the residential model. This project sources building stock data from the National Renewable Energy Laboratory to create 8760 hourly electricity demand profiles by enduse category. 

## Setup

Follow the instructions within the main repository ReadMe to setup your python environment. 

Be sure to have the 'bsky' environment activated in a terminal, navigate to the 'sample\enduse_data_pipeline' directory in your terminal, and then run the following code:

```bash
python main.py --help
```

This should return a description of ```main.py```, as well as the options available within the command line interface. Now, check to see if we can run and solve a model by executing:


```bash
python main.py 
```

There are additional argument options you can pass when you run the script including **--debug** and **--output**.

#### --debug
The debug option allows for more details to be written to the log file. 

#### --output
The output option allows for the user to specify the stored location  where the output files are saved. By default, the output files are saved within the *output* subdirectory.


### Configuration Settings
Future development of this project will likely migrate the configuration settings out of the `config_setup.py` file and into a separate toml file, as is standard practice in the rest of the projects in this repository. However, in the meantime, the configurations are stashed directly within the python script. 

#### test_build_data
Once you run the project, you may see a warning pop up about test_build_data. If **test_build_data** is set to:
 - ***true***, a test version of the enduse input file is built, called `EnduseShapes_test.csv`. 
 - ***false***, the enduse input file is built, called `EnduseShapes.csv`. 
 
 The reason this option is provided is because the project takes over an hour to run and requires 10 GB of local storage.  If you are interested in running the full project, you can edit the **test_build_data** setting in the `config_setup.py` file. 

 #### build_stock_db
 Once you've built the building stock database, there is very little reason to rebuild it again. Once you have the database saved locally, you can switch **build_stock_db** to ***false*** and it will bypass building the database.

## Model
Running the model results in two scripts being executed within the `runner.py` file, `enduse_db.py` and `enduse_demand.py`.

### enduse_db.py

This script creates a stock_database that downloads from NREL's datalake datasets from both the [ResStock](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2024%2Fresstock_amy2018_release_2%2Ftimeseries_aggregates%2Fby_state%2Fupgrade%3D0%2F) and [ComStock](https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=nrel-pds-building-stock%2Fend-use-load-profiles-for-us-building-stock%2F2023%2Fcomstock_amy2018_release_2%2Ftimeseries_aggregates%2Fby_state%2Fupgrade%3D0%2F) models. The input file is based on the data on actual meteorological year (AMY) 2018 timeseries aggregates by state. We use the 2023 and 2024 data releases in this project. 

The project downloads and stores the data for each state and building type and stores the data as tables within the database. 

### enduse_demand.py

This script takes the raw data downloaded and saved within the database and aggregates it up into a form that is then used within the prototype model. There are four main processes executed to do this: stock_aggregation, hourly_aggregation, national_aggregations, and calculate_percent_cols. 

#### stock_aggregation 
Stock aggregation takes the raw tables downloaded and saved in the enduse database and processes them into user defined enduse categories. The process keeps only the electricity-related columns from the raw data.

It then reads in either the `comstock_assignment.csv` or the `restock_assignment.csv`.  These input files are used to group the raw NREL data by building type and equipment type into the final enduse category to be used within the prototype.

#### hourly_aggregation
The raw data from NREL comes in 15-minute intervals and is based on *Eastern Standard Time*. This process first aggregates the 15-minute data into hourly data. Next, it uses the `timezone_assignments.csv` to convert the EST data into local time for each state. Lastly, it adjust months March through October to roughly account for day light savings in all states except Arizona. 

#### national_aggregations
National aggregations adds the adjsuted state-level data to create a national electricity profile. Future development of this project would assign the state hourly profiles developed to model regions. 

#### calculate_percent_cols
Lastly, the electricity demand is converted from consumption in kilowatthours by enduse category in each hour to a share of total demand by enduse category for the year. The resulting shares are relatively small since there are 8760 hours in a year. 

Finally, an 'everything_else'  category is created and added to the file, which is just a representative flat demand (e.g., 1 / 8760). 

### runner.py

After `runner.py` finishes executing the methods from `enduse_db.py` and `enduse_demand.py` as described above, the file outputs the final csv to be used for the prototype. 
