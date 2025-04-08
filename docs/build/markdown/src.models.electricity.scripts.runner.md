# src.models.electricity.scripts.runner

This file is a collection of functions that are used to build, run, and solve the electricity model.

### Functions

| [`build_elec_model`](#src.models.electricity.scripts.runner.build_elec_model)(all_frames, setin)     | building pyomo electricity model                                            |
|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `check_results`(results, SolutionStatus, ...)                                                        | Check results for termination condition and solution status                 |
| [`cost_learning_func`](#src.models.electricity.scripts.runner.cost_learning_func)(instance, tech, y) | function for updating learning costs by technology and year                 |
| `getLogger`([name])                                                                                  | Return a logger with the specified name, creating it if necessary.          |
| [`init_old_cap`](#src.models.electricity.scripts.runner.init_old_cap)(instance)                      | initialize capacity for 0th iteration                                       |
| `log_infeasible_constraints`(m[, tol, logger, ...])                                                  | Logs the infeasible constraints in the model.                               |
| [`run_elec_model`](#src.models.electricity.scripts.runner.run_elec_model)(settings[, solve])         | build electricity model (and solve if solve=True) after passing in settings |
| `select_solver`(instance)                                                                            | Select solver based on learning method                                      |
| [`set_new_cap`](#src.models.electricity.scripts.runner.set_new_cap)(instance)                        | calculate new capacity after solve iteration                                |
| [`solve_elec_model`](#src.models.electricity.scripts.runner.solve_elec_model)(instance)              | solve electicity model                                                      |
| [`update_cost`](#src.models.electricity.scripts.runner.update_cost)(instance)                        | update capital cost based on new capacity learning                          |

### Classes

| `Config_settings`(config_path[, args, test])       | Generates the model settings that are used to solve.   |
|----------------------------------------------------|--------------------------------------------------------|
| `Path`(\*args, \*\*kwargs)                         | PurePath subclass that can make system calls.          |
| `PowerModel`(\*args, \*\*kwds)                     | A PowerModel instance.                                 |
| `SolutionStatus`(\*values)                         |                                                        |
| `SolverStatus`(\*values)                           |                                                        |
| `TerminationCondition`(\*values)                   |                                                        |
| `TicTocTimer`([ostream, logger])                   | A class to calculate and report elapsed time.          |
| `datetime`(year, month, day[, hour[, minute[, ...) | The year, month and day arguments are required.        |

### src.models.electricity.scripts.runner.build_elec_model(all_frames, setin) → [PowerModel](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)

building pyomo electricity model

* **Parameters:**
  * **all_frames** (*dict* *of* *pd.DataFrame*) – input data frames
  * **setin** ([*Sets*](src.models.electricity.scripts.preprocessor.md#src.models.electricity.scripts.preprocessor.Sets)) – input settings Sets
* **Returns:**
  built (but unsolved) electricity model
* **Return type:**
  [PowerModel](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)

### src.models.electricity.scripts.runner.cost_learning_func(instance, tech, y)

function for updating learning costs by technology and year

* **Parameters:**
  * **instance** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – electricity pyomo model
  * **tech** (*int*) – technology type
  * **y** (*int*) – year
* **Returns:**
  updated capital cost based on learning calculation
* **Return type:**
  int

### src.models.electricity.scripts.runner.init_old_cap(instance)

initialize capacity for 0th iteration

* **Parameters:**
  **instance** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – unsolved electricity model

### src.models.electricity.scripts.runner.run_elec_model(settings: [Config_settings](src.common.config_setup.md#src.common.config_setup.Config_settings), solve=True) → [PowerModel](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)

build electricity model (and solve if solve=True) after passing in settings

* **Parameters:**
  * **settings** ([*Config_settings*](src.common.config_setup.md#src.common.config_setup.Config_settings)) – Configuration settings
  * **solve** (*bool* *,* *optional*) – solve electricity model?, by default True
* **Returns:**
  electricity model
* **Return type:**
  [PowerModel](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)

### src.models.electricity.scripts.runner.set_new_cap(instance)

calculate new capacity after solve iteration

* **Parameters:**
  **instance** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – solved electricity pyomo model

### src.models.electricity.scripts.runner.solve_elec_model(instance)

solve electicity model

* **Parameters:**
  **instance** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – built (but not solved) electricity pyomo model

### src.models.electricity.scripts.runner.update_cost(instance)

update capital cost based on new capacity learning

* **Parameters:**
  **instance** ([*PowerModel*](src.models.electricity.scripts.electricity_model.md#src.models.electricity.scripts.electricity_model.PowerModel)) – electricity pyomo model
