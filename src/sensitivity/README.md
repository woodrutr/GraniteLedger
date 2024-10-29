# Table of contents
- [Overview](#overview)
- [Use](#use)
- [Scripts](#scripts)
- [Issues](#issues)
- [Mathematical Details](#mathematical-details)
- [References](#references)
# Overview
The utilities in this directory provide tools to
1. Convert a Pyomo optimization model into a Sympy representation (referred to as "sympification"). 
2. Convert a sympification to a matrix representation of the sensitivities of decision variables, duals, and objective with respect to the parameters.
3. Fetch the sensitivity of any variable, dual, or the objective with respect to any parameter, and extrapolate the value of the former under any change in the vector of parameters.

The tools apply to any Pyomo optimization problem that is fully instantiated and solved as a stand-alone ConcreteModel type with equality and inequality constraints. All functions should be smooth. Mixed Integer Programs will not work.

Currently, large problems take prohibitively long to work with, so the script speed_test.py is provided, which lets you perform speed and accuracy tests on toy models of arbitrary size to gauge run-time before committing to a long run.

# Use

To run tests on the toy model, speed_test.py has all settings declared in the beginning of the scipt, and you can change the values in the various entries. The size of the toy models scale with $n=$(number of regions) * (number of hubs per region$)*($start year - end year$)$, and in testing, run-time is ~$O(n^2)$.

Before attempting to perform sensitivity analysis on a problem, a speed test can be used to see if it would run in an acceptable amount of time.

To run the speed test, simply run the speed_test script. It will execute all necessary functions given the declared values at the beginning of the file.

## Loading a model
Given a solved instance of a ConcreteModel, **model**, you can sympify it by creating and instance of AutoSympy:

```
import sensitivity_tools
auto = AutoSympy(model)
```

You can then create a sensitivity matrix instance with a list of parameters to consider as an argument, and then run the method get_sensitivities to return an instance of DifferentiableMapping:
```
M = auto.get_sensitivity_matrix([param1, param2])
diffmap = M.get_sensitivities()
```
The object diffmap now stores all your sensitivities, and you can pull them using the sensitivity method. So if you wanted to see the effect of a change in base_demand on production in 2025, for example, you could execute:
```
diffmap.sensitivity(production[2025], base_demand)
```

# Scripts and classes

## sensitivity_tools.py

This is the main utility file with general applicability.

It contains several classes:

### AutoSympy

This class is called with a pyomo model instance that has already been solved as an argument. The initialization of an AutoSympy instance will extract all the sets, parameters, variables, duals, and inequality and equality constraints from the model. It converts them to dictionaries of sympy symbols:values, and separates the duals by complementary slackness criteria.

The method get_sensitivity_matrix takes as an argument a list of parameters of interest, substitutes numeric values for all other parameters (nullifying their symbolic representation) and returns a SensitivtyMatrix object based on the symbolic representation and the parameters of interest.

### SensitivityMatrix
This class takes as input an AutoSympy object, a dictionary of duals:dual values produced from an AutoSympy object, and a dictionary of parameters:values split up between ones you want to substitute and ones you want to differentiate.

The initialization will create a series of Matrices corresponding to the blocks of the matrix U in [Castillo et al 2004](#references) by taking a variety of Jacobians and Hessians of symbolically represented functions that originated in a pyomo model. It will then combine them into a matrix U and a matrix S. The matrix U will be inverted, and if this succeeds, it will create an instance of the class DifferentiableMapping.

DifferentiableMapping keeps track of the symbols and their position in the variable and parameter vectors, and houses functions that allow you to extract sensitivities or extrapolate the change in variables, duals, or the objective.


### DifferentiableMapping

This class stores a set of dictionaries that map sympy symbols to their index in the vector of variables, duals, objective, and parameters to their position in the vector of parameters.

The method sensitivity will fetch the corresponding entry in the explicity sensitivity matrix expression for the sensitivity of the respective arguments to each other.

## babymodel.py

This script containts the class TestBabyModel, type ConcreteModel, as well as functions to generate the parameters and sets for the model.

The test model is a production and distribution model with regions, hubs, and transportation arcs, and capacity expansion, that minimizes the cost to satisfy a regional demand through production at hubs and transportation through arcs, of a resource that consumes electricity at minimum cost over the range of years. 

The topology of the transportation network is randomized, and the various capacities of hubs and properties of regions are also randomized. Because of this, it may be that the model is infeasible in some rare circumstances, in which case, simply generate a new one.

This model generator is included for testing purposes to evaluate run speed at various model sizes.

# Issues

While any model should be able to be sympified, in order for the resulting sensitivity calculation to work, the matrix U in SensitivityMatrix needs to be invertible. To that end, not all models will be "plug-and-play." In addition:

1. All constraints must be explicitly declared. If a variable is declared with a set domain, such as NonNegativeReals, it has an implicit lower-bound constraint which will not be picked up in the sympification and will likely result in an error when attempting to generate the final sensitivities.

2. Redundant constraints must be avoided. If a model has two constraints that are interchangeable, this may cause problems because the dual for one of the constraints may fail to satisfy the complementary slackness conditions in [Castillo et al 2006,2008](#references) and result in what is called a degenerate case, whose calculation is more complicated.

3. The matrix may be uninvertible despite best efforts, and further work and development is required to handle such cases.

4. Run-time is slow, and this method is currently not appropriate for large problems with several hundreds to thousands of variables on regular hardware. More work needs to be done to improve speed.

## Further work

Future development aims to streamline and abstract the handling of parameters, automate the extraction of lower bounds, prune redundant constraints, handle indexed data better, and speed up run time.

Additionally, one major goal is to be able to sympify a model that has not been solved; pickle the structure and retrieve it when needed; populate the data; filter the active, non-degenerate constraints; and obtain the sensitivity matrix without doing any symbolic differentiation on the spot.

# Mathematical Details

The method of calculating sensitivities is derived in [Castillo et al 2006](#references) and presented in simplified form in [Castillo et al 2008](#references)

# References
1. [Castillo et al 2006](https://link.springer.com/article/10.1007/s10957-005-7557-y)
2. [Castillo et al 2008](https://www.sciencedirect.com/science/article/abs/pii/S0951832008000999)
