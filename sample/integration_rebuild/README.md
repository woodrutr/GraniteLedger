## Integration Rebuild

This project is under development.


### Introduction

In the main Bluesky Prototype, we demonstrate the feasibility of linking and solving multiple NEMS-like modules coded in Python and organized by the Pyomo optimization package (see the README found in [src/integrator](/src/integrator/README.md)). In these solve-algorithms, the code is written to instantiate and solve an integrated model comprised of the power systems, hydrogen, and demand modules (see [unified.py](/src/integrator/unified.py) and [gaussseidel.py](/src/integrator/gaussseidel.py)). 

In order to scale up the Bluesky Prototype, we need to create generalizable integration structures that are applicable beyond the specific modules in the prototype. We require a model architecture that allows us to avoid problem-specific and complicated integration code, both to enhance the navigability of the code and to reduce the overhead associated with advancing the model.

### What is in this repository?

Along with two toy model classes, LowRes and HighRes, **this directory provides an initial design for a new integration structure that compiles a set of build instructions executed by a build class and integrates modules dynamically based on user configuration files** 

These build instructions could be executed directly for any number of modules switched on for a run, in constrast with the complex tree of boolean logic currently in the main code of the prototype. 

Moving from a hard-coded integration design to a flexible, dynamic version generated a few key advancements:
1) **Cleaner, modifiable, and interactive configuration scripts** --  Users should be able to specify how to construct a specific problem with specific modules, which parameters or variables needed to be integrated, and which solve method to undertake.
2) **Model logic handled prior to executing any computationally expensive tasks by enhancing the settings class** -- Instead of just holding settings for the solver and desired time periods for each module, the *Settings* class needs to turn the configuration language contained in the TOML file into a series of commands to execute, calling methods that handle those executions directly.
3) **Standardization of Pyomo model construction with a Model class such that indices are easily labeled and accessible for any component in any module** -- A ledger or registry (`model.cols_dict`) of components and their indices for each module is provided by creating a new base *Model* class (model.py), inherited by all pyomo models, which contains methods that wrap both component construction and registry updates into singular, human-readable function calls.
4) **Handle unit conversions between passed parameters, variables, and duals** -- With access to all indices, labels, and ordering via `model.cols_dict`, we can automate the exchange of values between modules and conduct any scaling necessary (e.g., between a component indexed at the seasonal level versus one at the bi-annual level) via the *Interchange* class.
5) **A complete rewrite of the model construction algorithm that generates the "meta" Pyomo model** -- With newly compiled build instructions from *Settings*, we can create the necessary "meta" model, or parent Pyomo model that holds references to each module directly. In other words, we can unpack the configuration file into a set of executable statements handled by the *Build* class. This is currently still under development, but users can navigate to [Build.py](/sample/integration_rebuild/src/common/build.py) to examine the logic underneath the execute method.

### Instructions

When running scripts for this project, all pathways to modules and scripts assume that the working directory is set to *sample/integration_rebuild*.

We recommend users use *main.py* to instantiate the settings as described in the configuration file and create a model build. However, running this script on its own is not very informative. Users should explore the various attributes contained within the *Settings* and *Build* instances to get a sense for how we compile the configuration scripts into a heirarchical structure of Pyomo models contained within *Build*.

The following sub-headers describe the important files contained within the repository.

#### common/run_config.toml

Those familiar with the main prototype (if not, see the [main documentation](/src/README.md)) will recognize the configuration file. In this redesign, *run_config.toml* handles which modules are included in a model build, which components get integrated together, and other options similar to the main prototype.

For each technology specified in the configuration file, settings will specify an instance of the options chosen. For example, for *LowRes* (desribed below), with `techs = [7,8,9]`, *Settings* will create instructions to create *lowres7*, *lowres8*, and *lowres9*.

[!CAUTION]
In general, the user should mostly take note and advise on the structure. Any large-scale modifications of *run_config.toml* will likely cause issues as error catching and guidance on the structure is currently limited.

#### common/config_setup.py

This is the *Settings* class, which creates instructions for which modules to build, which crosswalk to create for *Interchange*, the solve-algorithm based on the configured integration and solve method, and options for checking for configuration (e.g., max iterations).

**We recommend instantiating the settings using main.py and exploring the various attributes contained within**. Examining the solve_algorithm instructions is the best way to learn how *Build* functions.

#### common/model.py

The parent class for the two toy optimization problems; a version of this can be found embedded in the main prototype (and applied to PowerModel).

The Model class contains a series of methods to build pyomo components (sets, parameters, variables, and constraints). These methods construct the model while updating an attribute in the model class, `model.cols_dict`, which keeps track of each component index labels and ordering. This is essential for dynamically integrating modules together, as any exchange of parameters or specification of a meta-level constraint needs to know index labels and locations (as these are not contained in the Pyomo components).

#### common/build.py

This class handles the instantiation and execution of an integrated model solve based on the settings instance provided. *Build* contains methods that instantiate the necessary attributes for exchanging parameters (*Interchange*), the modules for a specific build, the *Tolerance* instance that keeps track of objective values across cycles, and handles the execution of information exchange and solving the model(s). 

#### common/tolerance.py

*Tolerance* is an object type which works with *Build* to keep track of solve cycles and check whether convergence has occurred based on the options provided in *run_config.toml*. Current implementation is limited due to development of *Build*.

#### interchange/improvedcrosswalk.py

These are classes that instantiate the objects responsible for tracking dimension conversions between different models, automating aggregation and averaging of exchange parameters. This file contains the classes:

* *Unit*: Creates and stores the correspondences between a unit of measurement within some dimension, and a base unit. These correspondences allow the aggregation and disaggregation to and from the base unit.

* *UnitDimension*: Creates, stores, and manages all the Unit objects within a dimension, generated from a crosswalk file. The main purpose is to maintain conversion weights between units and perform conversions of series of data indexed by one unit into another unit.

* *ExchangeSeries*: The class that packages indexed data to be exchanged. It stores the raw data in a standard format of a dictionary in the form {tuple of indices : value} along with tracking the names of the indices and which indices are dimension quantities. The methods included perform actions such as presenting a given dimension to allow conversion along that dimension, reorganizing the series, and updating data.

* *Interchange*: Stores UnitDimension objects and models, and automates the exchange of parameter data from one model to another by either 1. converting an exchange series to the desired standards or 2. (optional) directly requesting the exchange series and required data from the model. It iteratively cycles through the dimensions and converts units one at a time to perform a full conversion.

#### interchange/cw_testing.py

This is a testing script that instantiates and test objects from improvedcrosswalk.py referenced above. It creates UnitDimensions from the crosswalk files time_crosswalk.csv and space_crosswalk.csv, instantiates several ToyModel objects and populates their parameter data, loads it all into an Interchange object, and then performs conversions with symbolic and numerical data as a test of functionality. You can inspect the output or elements stored in memory to see the changes the operations make. The test is primarily to show automatic detection of dimension indices and sequential conversion across different resolutions on the same data series. A Sympy series is also created to test the conversion process with symbolic math data.

#### models/HighRes.py

This is a simple optimization model that minimizes cost for meeting demand in each region with generation. Data is randomly created upon initialization.

#### models/LowRes.py

This is a simple transport and storage problem at a coarser temporal scale than HighRes. Data is randomly created upon initialization.




