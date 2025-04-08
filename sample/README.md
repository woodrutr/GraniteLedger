# SAMPLE

### Table of Contents

- [Introduction](#introduction)
- [BaseLoad Data Pipeline](#baseload-data-pipeline)
- [Create SQL Database](#create-sql-database)
- [Electricity Data Pipeline](#electricity-data-pipeline)
- [Enduse Data Pipeline](#enduse-data-pipeline)
- [Hydrogen Interactive Test Module](#hydrogen-interactive-dev)

## Introduction

The Sample directory include projects that are under various stages of development in support of the prototype model.  


## AIMMS Sympy
The code in this directory test out the process for converting AIMMS code into sympy and then sympy into pyomo.


## BaseLoad Data Pipeline
This directory contains data sourcing scripts for the `BaseLoad.csv` input file used within the residential model. This project sources individual weather station data from the National Oceanic and Atmospheric Administration and electricity demand data from EIA's [Hourly Electric Grid Monitor](https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48) data. 


## Create SQL Database
This directory contains example code of how to batch load csv tables into a SQL database. 


## Electricity Data Pipeline
This directory contains data sourcing scripts for multiple input files used within the electricity model. This project is incomplete, but conceptully the goal of this pipeline is to to build out a pipeline that takes regionally disaggregated county-level data and builds the data up to user-specified regions used within the electricity model. 


## Enduse Data Pipeline
This directory contains data sourcing scripts for the `EnduseShapes.csv` input file used within the residential model. This project sources building stock data from the National Renewable Energy Laboratory to create 8760 hourly electricity demand profiles by enduse category. 


## Hydrogen Interactive Dev
The Hydrogen model is a prototype to a general model framework for any energy system representable as a network of regions, hubs, and transportation arcs. It is provided with data and functions specific to Hydrogen production and distribution but aims to be agnostic to the energy carrier being modeled.

The version in this directory is meant to test the regional flexibility allowed by representing the underlying data in a graph-structure, and does not integrate with other models. 


## Integration Rebuild
This directory contains a new prototype integration layout for the main model. It includes test code for building modules flexibly and lays out a vision for the interchange of information between modules.  
