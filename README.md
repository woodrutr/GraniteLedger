# BlueSky Prototype

- [BlueSky Prototype](#bluesky-prototype)
    - [Description](#description)
        - [Prototype Capabilities](#prototype-capabilities)
        - [Modules](#modules)
        - [Model Complexity and Computational Efficiency](#model-complexity-and-computational-efficiency)
        - [Feedback](#feedback)
    - [Documentation](#documentation)
        - [Model Documentation](#model-documentation)
        - [Code Documentation](#code-documentation)
        - [Integration Documentation](#integration-documentation)
    - [Visuals](#visuals)
    - [Set-Up](#set-up)
        - [Installation](#installation)
        - [Gitbash and bat file Method](#gitbash-and-bat-file-method)
    - [Usage](#usage)
        - [First Run](#first-run)
        - [Overview](#overview)
    - [Support](#support)
    - [Contributing](#contributing)
        - [Steps to Contribute](#steps-to-contribute)
        - [Guidelines](#guidelines)
    - [Authors and acknowledgment](#authors-and-acknowledgment)
    - [License](#license)
    - [Project status](#project-status)


## Description
[Project BlueSky](https://www.eia.gov/totalenergy/data/bluesky/) is an EIA initiative to develop an open source, next generation energy systems model, which will eventually be used to produce the Annual Energy Outlook ([AEO](https://www.eia.gov/outlooks/aeo/)) and International Energy Outlook ([IEO](https://www.eia.gov/outlooks/ieo/)). Our outlooks are focused on projecting realistic outcomes under a given set of assumptions, accounting for rapid technological innovation, new policy implementation, changing consumer preferences, shifting trade patterns, and the real-world friction associated with the adoption of novel or risky technology. To address these challenges, the next generation model is designed to be modular, flexible, transparent, and robust.

The BlueSky Prototype is the first step towards creating this next generation model. Our objective in releasing the prototype is to give the modeling community an early opportunity to experiment with the new framework and provide feedback. Feedback gathered from stakeholders will be used to develop a full-scale version of the model beginning in 2025.

There are four key features associated with the BlueSkype Prototype:

1. **A computationally efficient, modular structure that allows each sector to flexibly capture the underlying market behavior using different governing dynamics.** The protype contains three test modules representing electricity, hydrogen, and residential demand. Both the electricity and hydrogen modules employ linear optimization as a governing dynamic, while the residential sector adjusts demand levels based on prices from the other two sectors. The protoype includes an `integrator` module that formulates these modules into a single non-linear optimization problem using two different methods. This feature will allow the next generation model to be **modular and flexible**.
2. **Well-documented Python code.** In the prototype, we have tried to write Python code that is easy to follow. We have included both detailed, high-level descriptions of each module in markdown, as well as code documentation using [Sphinx](https://www.sphinx-doc.org/en/master/), which we use to aggregate doc strings embedded in the code to create documentation. Most open source models develop coding and documentation styles prior to their initial release. Even for models that are easy to apply, the underlying code and documentation are not always clear. We are inviting the community to provide feedback at this early stage so that code and documentation developed for the production version are as transparent as possible. We deliberately allowed the coding style to vary by module, with the hope that we'll receive feedback on which coding and in-line documentation approaches are easiest to understand. This feature will allow the next generation model to be **transparent**.
3. **An example data pipeline documented with Snakemake.** Energy system models are data intensive and require a multitude of data transformations to convert raw data from a known source into formatted model input data. We are experimenting with [Snakemake](https://snakemake.readthedocs.io/en/stable/), a tool that allows us to create a well-documented data management framework to organize, pre-process, and post-process model data. In the prototype, Snakemake is used in a standalone application to create a geospatial crosswalk involving both electricity operations and weather data. We are considering the use of Snakemake across the model to comprehensively manage input data. This feature will allow the next generation model to be **transparent**.
4. **Two methods to characterize model sensitivity.[Not ready yet for testing, not currently included in this repository]** One method uses an efficient way to approximate derivatives of model outputs with respect to inputs for systems of nonlinear equations using complex variables ([Lai et al., 2005](https://arc.aiaa.org/doi/abs/10.2514/6.2005-5944)). The method is broadly applicable to any module where the equations are continuously differentiable and has been implemented in applications beyond energy modeling. A second method is directly applicable to convex optimization problems and applies approximation methods for sensitivity analysis  on optimization models ([Castillo et al., 2006](https://link.springer.com/article/10.1007/s10957-005-7557-y)). These methods will allow us to ensure **robust** model results by testing model sensitivity. Additional methods to quantify model sensitvitiy and uncertainy will be implemented in the future.

The BlueSke Prototype is focused on demoing the four features above. The feedback gathered from community experimetation will be used to improve future versions of the next genration model. Given the prototype's limited scope, it does NOT provide results that are relevant to the real world. To emphasize this point, we have omitted any reference to a specific country, region, or timeframe in the input data. 


### Prototype Capabilities
The BlueSky Prototype has the following advancements in the model formulation: 
* Provides two options for solving the model to equilibrium: 1) Gauss-Seidel iteration and 2) A hybrid optimization-iteration method
* Includes nonlinear price responses in the residential module and endogenous nonlinear learning in the electricity module.
* Includes approximation methods that output the sensitivity of the solution with respect to uncertain inputs. 

The BlueSky Prototype has the following advancements in the code implementation: 
* Open-source code under an Apache 2.0 license built primarily on Python with the capability to be used with free solvers. 
* Written in a modular structure that allows modules to be run independently and also allows easy addition of modules. 
* Modules also have a modular “block” structure programmed in Pyomo that allows swapping of model capabilities.
* Flexibility to run under different temporal and spatial granularities and technological learning capabilities so the model size and complexity can be adjusted.
* A data pipeline and workflow process for input data into the model. The prototype separates the data from the modeling, allowing the flexibility of diverse data inputs without hardcoding.
* Efficient implementation using sparse indexing, mutable parameters, and shared variables in simultaneous optimization to speed up computation.
* A generic methodology for converting non-optimization formulations into optimization formulations with the Residential module as an example.
* Detailed and automated documentation.

Each of the individual modules, including the integrator, shows these capabilities highlighted above.

Note that the prototype only contains basic output visualization capabilities. This is because the prototype is not meant to include any capabilities of producing results that need to be analyzed. 


### Modules
Long-term models at EIA (NEMS and WEPS) should be thought of as a collection of modules as opposed to a consistent, combined model. Each sector needs its own governing dynamics, and thus its own mathematical formulation and code implementation. The modules in the prototype have been deliberately designed to mimic these differences among modules and among teams at EIA. Thus, while the prototype does strive to achieve some consistency in coding, the modules have their own mathematical formulation and code implementation that can seem different. We encourage you to explore these differences and give us feedback on their implementation.

We have also implemented two different methods for integration, with documentation linked below. The "integration" module is on the same levels as the modules and uses copies of the modules within its own folder. The "integrator" is a level above the modules and calls on the modules from their respective folders. It also allows standalone runs of the modules. We are currently testing both methods and aim to have a single method for release. You are welcome to provide feedback.

### Model Complexity and Computational Efficiency
The prototype provides three features that address model complexity and make solving the model more efficient than the current setup in NEMS/WEPS:
1. The computational modeling is programmed in Python, which allows models and data to be run concurrently in the same language and hold information in memory. The complexity of opening different software, running modules independently, and exchanging data not in memory is eliminated.
2. The prototype provides two different methods to solve the integrated model. The Gauss-Seidel method is made efficient through the first step of calling Python models through a Python script and storing information that has to be passed through in memory. Further, in the Gauss-Seidel and other iterative methods in the prototype (such as the linearization of nonlinear learning in the Electricity module) the starting point for the next iteration is already stored in memory and speeds up computation. This can be seen when looking at the times for iterative solves for the modules, which decrease as the iterations progress. The second method, a unified optimization method, integrates the modules into one mathematical structure that can contain a potentially diverse set of governing dynamics. Theoretically, if an efficient mathematical structure can be found, these methods are computationally more efficient than iterative methods of achieving equilibrium.  For example, in solving formulations with multiple players, iterative methods such as Gauss-Seidel (also referred to as diagonalization in the literature) tend to converge slower than combining problems into a nonlinear optimization problem ([Leyffer and Munson, 2010](https://www.tandfonline.com/doi/abs/10.1080/10556780903448052); [Ralph and Smeers, 2006](https://ieeexplore.ieee.org/abstract/document/4075721)). They take advantage of the mathematical structure of individual modules to produce a larger problem where the existence and uniqueness properties can be studied. The prototype shows this as the unified optimization method solves problems zzz times faster.
3. The prototype is flexible on the number of regions and years it needs to solve for, both in the modules and in integrated solves. The prototype can also be started in later years and does not need a particular starting and stopping year. This flexibility allows testing new complexity additions on small versions of the model without having to test on a full version.


### Feedback
We encourage feedback on the mathematical formulation and code implementation of the prototype. Any feedback or questions should be posted in the [Teams Channel](https://teams.microsoft.com/l/channel/19%3Ab0d4d80737a74fccbf47a2540074d48d%40thread.tacv2/Blue%20Sky%20-%20ALL%20EIA?groupId=46a5f4fe-7edb-4841-8fa2-9b315e675359&tenantId=545f0b0b-e440-4054-9c17-d250cf3556b2). In particular, we are soliciting feedback on the following questions:

(Group 1)
* How can we make it easier to understand the documentation, download the code, modify it, and run it?
* Are there general coding best-practices you have suggestions on?
* Is it easy to identify some advantages of this code implementation over what we currently have in NEMS/WEPS?
* Is it easy to identify some disadvantages of this code implementation over what we currently have in NEMS/WEPS?
* Are there portions of the optimization implementation that would be easier to do in AIMMS beyond sparse indexing and general advantages of AIMMS such as naming or sets, variables, and parameters?

(Groups 1 and 2)
* Is this framework nimble and flexible enough to code up the current capabilities of NEMS/WEPS?
* What are some capabilities you would like to see tested in this framework?
* Are we conveying a clear message with regards to the purpose of the prototype?
* Can you understand how the prototype and eventual next generation model will be an advancement from NEMS/WEPS and also different from other existing external models?
* Is there anything in the prototype that we should not release to the public because it is proprietary information or because it will pre-empt any other EIA analysis such as AEO2025?

We do not need feedback on the energy system representation or results at this point. We also do not have capabilities for debugging developed right now, and suggestions on those are welcome.
 

## Documentation

Documentation for each of the modules are below. We also include documentation for a sample data workflow tool we have implemented and would love feedback on.

### Model Documentation
* [Electricity Module](src/models/electricity/README.md)
* [Hydrogen Module](src/models/hydrogen/README.md)
* [Residential Module](src/models/residential/README.md)
* [Data Workflow](sample/data_pipeline/README.md)


### Code Documentation

[Instructions for HTML, Markdown, and PDF documentation](docs/README.md)


### Integration Documentation
We also have documentation for two drafts of how we intend to solve the modules together. Note, these are under development and don't have full functionality yet. We would appreciate feedback on both and which aspects of each are easy to understand. 

Our current method for integration is in the "integrator" module:
* [Integrator Module with Pyomo Concrete Models (Only tested on 3 regions, four years)](src/integrator/README.md)

Our second method, currently in the sample folder, is here:
* [Integration Module with Pyomo Blocks (Only tested on 1 region, two years)](sample/integration/README.md)




## Visuals

<img src="docs/images/PrototypeDiagram.png" alt="Diagram showing the BlueSky Prototype Structure and Components" />

## Set-Up

To set up the appropriate Python environment, you will need to install the following from the EIA Software Center:
* Anaconda (or Miniconda 3)
* Python 3.12
* Git
* VSCode
* TortoiseGit (Optional)

This guide will assume usage of VSCode as your IDE of choice; if this isn't the case, feel free to skip to the steps where you create and activate your Conda environment.

Helpful resources can be found in the [FAQ section](https://git.eia.gov/oea/nextgen-bluesky/bluesky_prototype/-/wikis/Frequently-Asked-Questions-(FAQ)) of the wiki page.

### Installation

Detailed instructions with screenshots can be found [Installation-FAQ](https://git.eia.gov/oea/nextgen-bluesky/bluesky_prototype/-/wikis/create-a-conda-environment-in-VS-Code)


1. **Clone the bluesky_prototype repo**

   * VSCode
   * TortoiseGit
   * Git Bash

```base
https://git.eia.gov/oea/nextgen-bluesky/bluesky_prototype.git
```

2. **Set up VSCode**
   * Open VSCode and navigate to the **bluesky_prototype** repository

   * Press `ctrl+shift+P` and search for *Python: Select Interpreter*. Select any Conda Environment listed. I recommend selecting the `"(Base)"` environment. If none are available, check whether you have Anaconda (or Miniconda) installed on your machine.

   * Press `ctrl+shift+P` and search for *Terminal: Select Default Profile*. Select `Command Prompt`.

   * Open a new terminal by pressing `` ctrl+shift+` ``

   * Check to see if this terminal is Conda-enabled; it should say `(Base)` or the name of the conda environment you have chosen as your Python interpreter

3. **Create the 'bsky' Conda environment**
   * The libraries needed to run any bluesky module are contained in the `bsky` conda environment described in conda_env.yml

      - If you are working from an Anaconda prompt outside of your IDE, [navigate to Anaconda and open an Anaconda Prompt](https://git.eia.gov/oea/nextgen-bluesky/bluesky_prototype/-/wikis/Clone-repo-and-create-environment-with-GitBash-and-Anaconda-prompt)
      - If you are working from a terminal within VSCode, use this terminal to execute the commands in the next few steps

   * Check existing environments and run `conda info --envs`

   * If "bsky" does not exist on your machine...
      - Run `conda env create -f conda_env.yml` in your terminal to create the environment
      - Run `conda activate bsky` to activate the environment

   * If "bsky" does exist on your machine...
      - Execute `conda activate bksy` to activate your environment
      - Run `conda env update --file conda_env.yml --prune` to check for updates
   
   * If the installation is corrupted and the environment will not activate
      - Run `conda remove -n bsky --all` to remove
      - Reinstall as if the environment does not exist as described above!

4. **Some libraries used are not available in Conda-Forge. Once activated, use pip to install the following non-Conda libraries**
   * `pip install highspy`

5. **You are ready to run the model!** If you've already created the environment, you only need to activate the environment upon your next session

### Gitbash and bat file Method

1. **Clone the bluesky_prototype repo** 

```base
git clone https://git.eia.gov/oea/nextgen-bluesky/bluesky_prototype.git
```
Helpful resources can be found in the [FAQ section](https://git.eia.gov/oea/nextgen-bluesky/bluesky_prototype/-/wikis/Frequently-Asked-Questions-(FAQ)) of the wiki page.

2. Navigate to the `envs` directory and open the `env-setup.bat` file in a text editor (**VS-Code** or **Notepadd++**)

3. Edit the **.bat** file and paste in the path to your `active.bat` in your anaconda installation on your computer *highlighted below 

```bash

call "C:\PATH-TO-ANACONDA\user\anaconda3\condabin\activate.bat"

```

4. Edit the **.bat** file and paste in the directory path to your cloned repo

```bash

cd "C:\PATH-TO-BLUSKY_REPO\gh_repos_n\bluesky_prototype" && conda env create -f conda_env.yml && conda activate bsky && pip install highspy


```
5. Save the file, and run it by double clicking on the file. 

6. You are ready to run the model using the **bsky** environment. 


## Usage

### First Run

Let's test the installation! 

'main.py' is the control script that, when executed, imports and runs methods contained within the integrator module to instantiate, build, and solve models. Be sure to have 'bsky' activated in a terminal, navigate to the 'bluesky_prototype' directory in your terminal, and then run the following code:
 
```bash
python main.py --help
```
This should return a description of 'main.py', as well as the options available within the command line interface. Now, check to see if we can run and solve a model by executing:

```bash
python main.py --mode elec
```

If this reports an objective value, you have successfully run the electricity model in standalone! The pattern is the same for running other versions/methods of the model, including integrated runs.

We can declare the mode using ['run_config.toml'](src/integrator/run_config.toml) by changing the 'default_mode'. Navigate to this file and change the option for default mode from `'gs-combo'` to `'unified-combo'`.

```
# RUN CONFIGURATIONS
# This file is used to establish the main model configurations users can edit when running

###################################################################################################
# Universial Inputs

#SETTINGS
# Mode choices=['elec','h2','unified-combo','gs-combo','residential']
# The default mode is only used when --mode argument is not passed by the user when running main.py
default_mode = 'gs-combo'

force_10 = false  # forces 10 iterations in solves that iterate, specify true or false (lowercase)
tol = 0.05  # percent tolerance allowed between iterations, specify a number less than 1
max_iter = 12 # max number of iterations 

```

Then, run 'main.py' without specifiying a mode:

```bash
python main.py
```

### Overview

The [Integrator Module](src/integrator/README.md) contains the methods you use to run modules in standalone and integrated modes. You can access and run these methods via ['main.py'](main.py) in the parent directory. All should be run from the top-level of the project via the 'main.py' file.  Running `python main.py --help` will display the mode options.

One of the goals of the prototype is to streamline control of model builds and allow for easily-accessible customization of runs. The majority of the options for both standalone and integrated model runs rely upon a configuration file located in the integrator folder: ['run_config.toml'](src/integrator/run_config.toml). 
 
Here you can find the file paths pointing to regions/years to synchronize the models. These switch (sw) files ([sw_reg for regions]('src/integrator/input/sw_reg.csv') and [sw_year for years]('src/integrator/input/sw_year.csv')) can be modified to include or remove years and regions from your model build. This file will be built out further as we continue to develop the prototype.

You can also navigate to the [Hydrogen](src/models/hydrogen/README.md) and [Electricity](src/models/electricity/README.md) modules for more details about with them directly. 

See the `--help` command as described above to see the available modes! More information can be found in the [Integrator Module](src/integrator/README.md) readme file!


## Support

Please check out the [Frequently Asked Questions (FAQ)](https://git.eia.gov/oea/nextgen-bluesky/bluesky_prototype/-/wikis/Frequently-Asked-Questions-(FAQ)) section for helpful resources


Please reach out to BlueSky team members for assistance:
[Teams Channel](https://teams.microsoft.com/l/channel/19%3Ab0d4d80737a74fccbf47a2540074d48d%40thread.tacv2/Blue%20Sky%20-%20ALL%20EIA?groupId=46a5f4fe-7edb-4841-8fa2-9b315e675359&tenantId=545f0b0b-e440-4054-9c17-d250cf3556b2) 
`Sauleh.Siddiqui@eia.gov`
`Adam.Heisey@eia.gov`
`Cara.Marcy@eia.gov`
`James.Willbanks@eia.gov`
`Nina.Vincent@eia.gov`
`Mackenzie.Jones@eia.gov`
`Jonathan.Inbal@eia.gov`
`Brian.Cultice@eia.gov`


## Contributing


We welcome contributions to our project and appreciate your effort to improve it! To ensure a smooth and organized workflow, please follow these steps when contributing to the main repository on our internal GitLab account:

### Steps to Contribute

1. **Clone the Repository**: Start by cloning the main repository to your local machine.
   ```bash
   git clone https://git.eia.gov/oea/nextgen-bluesky/bluesky_prototype.git
   cd main-repo
   ```

2. **Create a New Branch**: In the cloned repository, create a new branch for your contribution. Use a descriptive name for your branch that reflects the nature of your changes. For example:
   ```bash
   git checkout -b feature/new-awesome-feature
   ```

3. **Make Your Changes**: Implement your changes in the new branch. Make sure to follow our coding standards and include appropriate tests and documentation.

4. **Commit Your Changes**: Once your changes are ready, commit them with a meaningful commit message. Please make sure your commit messages are clear and concise.

5. **Push Your Branch**: Push your branch to the main repository on GitLab:
   ```bash
   git push origin feature/new-awesome-feature
   ```

6. **Open a Merge Request**: Open a merge request (MR) from your branch to the `main` branch of the main repository. Provide a detailed description of your changes and any relevant information that will help in reviewing your MR.

7. **Request a BlueSky Member for Merge**: Once your MR is ready for review, request a BlueSky member to review and merge your changes into the `main` branch. You can do this by tagging a BlueSky member in your merge request comments or by reaching out to them directly.

### Guidelines

- **Code Quality**: Ensure your code adheres to the project's [coding standards](https://git.eia.gov/oea/nextgen-bluesky/bluesky-sandbox/-/wikis/Documentation:-autodocstring) and is well-documented.
- **Documentation**: Update or add documentation as needed to explain your changes and how to use any new features.
- **Discussions**: Feel free to start a discussion if you have any questions or need clarification on any aspect of the project.

We appreciate your contributions and efforts to improve our project. Thank you for your collaboration!

If you have any questions or need further assistance, please don't hesitate to reach out to us.

## Authors and acknowledgment
`U.S. Energy Information Administration`

## License
The BlueSky code, as distributed here, is governed by [specific licenses](https://github.com/eiagov)

## Project status
Currently under development and review
