# src.models.hydrogen.model.actions

A sequencer for actions in the model.
This may change up a bit, but it is a place to assert control of the execution sequence for now

### Functions

| [`build_grid`](#src.models.hydrogen.model.actions.build_grid)(grid_data)                    | build a grid from grid_data                                                                                                                         |
|---------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| [`build_model`](#src.models.hydrogen.model.actions.build_model)(grid, \*\*kwds)             | build model from grd                                                                                                                                |
| `check_optimal_termination`(results)                                                        | This function returns True if the termination condition for the solver is 'optimal', 'locallyOptimal', or 'globallyOptimal', and the status is 'ok' |
| `getLogger`([name])                                                                         | Return a logger with the specified name, creating it if necessary.                                                                                  |
| [`load_data`](#src.models.hydrogen.model.actions.load_data)(path_to_input, \*\*kwds)        | load data for model                                                                                                                                 |
| [`make_h2_outputs`](#src.models.hydrogen.model.actions.make_h2_outputs)(output_path, model) | save model outputs                                                                                                                                  |
| [`quick_summary`](#src.models.hydrogen.model.actions.quick_summary)(solved_hm)              | print and return summary of solve                                                                                                                   |
| [`run_hydrogen_model`](#src.models.hydrogen.model.actions.run_hydrogen_model)(settings)     | run hydrogen model in standalone                                                                                                                    |
| `solve`(hm)                                                                                 | \_summary_                                                                                                                                          |
| [`solve_it`](#src.models.hydrogen.model.actions.solve_it)(hm)                               | solve hm                                                                                                                                            |
| `value`(obj[, exception])                                                                   | A utility function that returns the value of a Pyomo object or expression.                                                                          |

### Classes

| `Grid`([data])                                 |                                               |
|------------------------------------------------|-----------------------------------------------|
| `GridData`(data_folder[, regions_of_interest]) |                                               |
| `H2Model`(\*args, \*\*kwds)                    |                                               |
| `Path`(\*args, \*\*kwargs)                     | PurePath subclass that can make system calls. |
| `SolverResults`(\*args, \*\*kwargs)            |                                               |

### src.models.hydrogen.model.actions.build_grid(grid_data: [GridData](src.models.hydrogen.network.grid_data.md#src.models.hydrogen.network.grid_data.GridData)) → [Grid](src.models.hydrogen.network.grid.md#src.models.hydrogen.network.grid.Grid)

build a grid from grid_data

* **Parameters:**
  **grid_data** (*obj*) – GridData object to build grid from
* **Returns:**
  **Grid** – Grid object
* **Return type:**
  obj

### src.models.hydrogen.model.actions.build_model(grid: [Grid](src.models.hydrogen.network.grid.md#src.models.hydrogen.network.grid.Grid), \*\*kwds) → [H2Model](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model)

build model from grd

* **Parameters:**
  **grid** (*obj*) – Grid object to build model from
* **Returns:**
  **H2Model** – H2Model object
* **Return type:**
  obj

### src.models.hydrogen.model.actions.load_data(path_to_input: Path, \*\*kwds) → [GridData](src.models.hydrogen.network.grid_data.md#src.models.hydrogen.network.grid_data.GridData)

load data for model

* **Parameters:**
  **path_to_input** (*Path*) – Data folder path
* **Returns:**
  **GridData** – Grid Data object from path
* **Return type:**
  obj

### src.models.hydrogen.model.actions.make_h2_outputs(output_path, model)

save model outputs

* **Parameters:**
  **model** (*obj*) – Solved H2Model

### src.models.hydrogen.model.actions.quick_summary(solved_hm: [H2Model](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model)) → None

print and return summary of solve

* **Parameters:**
  **solved_hm** (*obj*) – Solved H2Model
* **Returns:**
  **res** – Printed summary
* **Return type:**
  str

### src.models.hydrogen.model.actions.run_hydrogen_model(settings)

run hydrogen model in standalone

* **Parameters:**
  **settings** (*obj*) – Config_setup instance

### src.models.hydrogen.model.actions.solve_it(hm: [H2Model](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model)) → SolverResults

solve hm

* **Parameters:**
  **hm** (*objH2Model*) – H2Model to solve
* **Returns:**
  **SolverResults** – results of solve
* **Return type:**
  obj
