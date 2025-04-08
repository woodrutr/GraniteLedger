# src.sensitivity.faster_sensitivity

faster_sensitivity

This file contains the class SensitivityMatrix which  takes in sympy objects that have been converted from pyomo. It builds the matrix of partials to be used in sensitivity analysis.

It also contains class AutoSympy which takes in pyomo models and converts the objects into sympy.

Finally, it contains class toy_model, the sensitivity method in action and then runs toy_model with input n=5.

The file babymodel.py can be also use this method by importing this file instead of sensitivity_tools.py by:
: from faster_sensitivity import \*

### Classes

| [`AutoSympy`](#src.sensitivity.faster_sensitivity.AutoSympy)(model)                                     | This class take in pyomo models and converts the objects into sympy.        |
|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| [`SensitivityMatrix`](#src.sensitivity.faster_sensitivity.SensitivityMatrix)(sympification, duals, ...) | This class takes in sympy objects that have been converted from pyomo.      |
| `date`                                                                                                  | date(year, month, day) --> date object                                      |
| `datetime`(year, month, day[, hour[, minute[, ...)                                                      | The year, month and day arguments are required.                             |
| `time`                                                                                                  | time([hour[, minute[, second[, microsecond[, tzinfo]]]]]) --> a time object |
| `timedelta`                                                                                             | Difference between two datetime values.                                     |
| `timezone`                                                                                              | Fixed offset from UTC implementation of tzinfo.                             |
| [`toy_model`](#src.sensitivity.faster_sensitivity.toy_model)(n)                                         | An example of the method in action that scales by the given 'n' value       |
| `tzinfo`                                                                                                | Abstract base class for time zone info objects.                             |

### *class* src.sensitivity.faster_sensitivity.AutoSympy(model)

This class take in pyomo models and converts the objects into sympy.
This is useful for problems that needs methods such as derivatives to be calculated on the equations.
We use these derivatives to calculate a sensitivity matrix that estimates the changes in variables due to changes in parameters

#### check_complimentarity_all()

#### generate_duals(constraints, duals)

Uses dual values and slack values to classify each constraint. It also stores the dual values for substitution later.

* **Parameters:**
  * **constraints** (*model.component_objects* *(**pyo.Constraint* *)*) – All of the constraint objects from the pyomo model
  * **duals** (*model.dual*) – All of the dual (or Suffix) objects from the pyomo model
* **Returns:**
  \_description_
* **Return type:**
  dict of lists and dicts

#### get_constraints()

This function converts all of the constraints in the pyomo object and converts the pyomo expressions into sympy expressions.

* **Returns:**
  Returns 2 dictionaries:
  equality_constraints: keys are tuples (constraint_name, index) and values are sympy expressions
  inequality_constraints: keys are tuples (constraint_name, index) and values are sympy expressions
* **Return type:**
  dict, dict

#### get_objective()

This converts the pyomo objective function into a sympy function.

* **Returns:**
  The pyomo objective function converted into sympy
* **Return type:**
  sympy equation

#### get_parameters()

Convert pyomo parameters into sympy objects.
This procedure creates sympy IndexedBase objects and sympy Symbol objects of similar names.
The IndexedBase datatype is necessary to parse the equations, but it does not work well with derivatives.
We will substitute in Symbols when the equations are all created, so they need to map to each other.
To keep the columns in order through all procedures, all parameters are given a unique column number by the variable “position”
This position is stored in the class dict param_position_map

* **Returns:**
  Returns 4 dictionaries:
  parameters: keys are pyomo parameter names and values are sympy IndexedBase objects with the same name
  parameter_values: keys are sympy symbols and values are the numerical values of the pyomo objects
  parameter_index_sets: keys are pyomo parameter names and values are lists of that parameters indices
  symbol_map: keys are pyomo parameters with an index and values are sympy symbols with a similarly styled name and index
* **Return type:**
  dict, dict, dict, dict

#### get_sensitivity_matrix(parameters_of_interest=None)

This function gathers all of the new sympy objects and creates a SensitivityMatrix object.

* **Parameters:**
  **parameters_of_interest** (*dict* *,* *optional*) – Specified subset of the parameters if more information is known about needless parameters, by default None
* **Returns:**
  a SensitivityMatrix object that contains the sensitivity matrix and commands to use it.
* **Return type:**
  [SensitivityMatrix](#src.sensitivity.faster_sensitivity.SensitivityMatrix)

#### get_sets()

Convert pyomo sets into sympy indexes

* **Returns:**
  The first dictionary has the pyomo objects’ names as keys and newly created sympy indexes as values.
  The second dictionary has the new sympy indexes as keys and the pyomo sets’ values as the dict values.
* **Return type:**
  dict, dict

#### get_variables()

Convert pyomo variables into sympy objects.
This procedure creates sympy IndexedBase objects and sympy Symbol objects of similar names.
The IndexedBase datatype is necessary to parse the equations, but it does not work well with derivatives.
We will substitute in Symbols when the equations are all created, so they need to map to each other.
To keep the columns in order through all procedures, all parameters are given a unique column number by the variable “position”
This position is stored in the class dict param_position_map

* **Returns:**
  Returns 2 dictionaries:
  variables: keys are pyomo variable names and values are sympy IndexedBase objects with the same name
  variable values: keys are sympy symbols and values are the numerical values of the pyomo objects
  It is also worth mentioning that this adds entries to the self.symbol_map in the same way parameters do.
  Symbol map entries have keys of IndexedBase objects and the values are their associated sympy Symbol
* **Return type:**
  dict, dict

### *class* src.sensitivity.faster_sensitivity.SensitivityMatrix(sympification, duals, parameters_of_interest)

This class takes in sympy objects that have been converted from pyomo.
It builds the matrix of partials to be used in sensitivity analysis.

#### generate_matrix()

This creates all of the matrices that will be combined into the U and S matrices.

* **Returns:**
  Returns 2 dictionaries. The first is the dictionary of matrix components with their names as keys.
  The second dictionary is a map from the symbols to their values.
* **Return type:**
  dict, dict

#### get_partial(x, a)

Retrieve the value of a particular partial derivative.
The value retrieved will be dx/da.

* **Parameters:**
  * **x** (*sp.Symbol*) – The symbol for the variable that you wish to know the change effect
  * **a** (*sp.Symbol*) – The symbol for the parameter that you wish to change to cause an effect on a variable
* **Returns:**
  The value of the partial derivative dx/da
* **Return type:**
  float

#### get_partials_matrix()

Calculate the matrix of all partials as U^(-1) \* S
Thus far, this is found to run the fastest when U^(-1) and S are numpy arrays

* **Returns:**
  Full partials matrix
* **Return type:**
  np.ndarray

#### get_sensitivity_range(x, a, percent)

The estimated values for “x” if the parameter “a” changes by percent% (as number 0% to 100%).
It will return values for an increase and decrease of the percent given.

* **Parameters:**
  * **x** (*sp.Symbol*) – The symbol for the variable that you wish to know the change effect
  * **a** (*sp.Symbol*) – The symbol for the parameter that you wish to change to cause an effect on a variable
  * **percent** (*float*) – A number 0-100 for the percent change in “a”
* **Returns:**
  Returns the estimated value for “x” if “a” is increased by percent% and decreased by percent%
* **Return type:**
  float, float

#### invert_U()

Calculates the inverse of the U matrix.
The fastest method found for this so far has been to convert to numpy and use its inverse function

* **Returns:**
  Calculated matrix for the inverse of U
* **Return type:**
  np.ndarray

#### matrix_assembly(components, subs_dict)

Combines matrix components to create U and S matrices from the literature.

* **Parameters:**
  * **components** (*dict*) – Dictionary of all precalculated matrix components
  * **subs_dict** (*dict*) – Dictionary that maps symbols to their values
* **Returns:**
  Returns the U and S matrices respectively with all symbols replaced by corresponding values
* **Return type:**
  sp.Matrix, sp.Matrix

#### matrix_sub(M, subs)

A function that substitutes values into a matrix.
This is the same result as sp.Matrix().subs(subs).
This speeds up runtime by only attempting to substitute values into symbols that actually exist in each cell.

* **Parameters:**
  * **M** (*sp.Matrix*) – The matrix to have symbols substituted for values
  * **subs** (*dict*) – Dictionary of values for sympy symbols
* **Returns:**
  The original matrix with all given values substituted into their symbols
* **Return type:**
  sp.Matrix

#### new_jacobian(f, values, map)

A function that returns the same result as Matrix.jacobian(values).
This speeds up runtime by only taking derivatives of symbols that exist.
The original function takes the derivative wrt everything in values.

* **Parameters:**
  * **f** (*sp.Matrix*) – Matrix of equations
  * **values** (*list*) – List of symbols that the function will take derivatives with respect to
  * **map** (*dict*) – Dictionary of column locations for each symbol
* **Returns:**
  Returns jacobian of given f matrix
* **Return type:**
  sp.Matrix

### *class* src.sensitivity.faster_sensitivity.toy_model(n)

An example of the method in action that scales by the given ‘n’ value

#### create_model()
