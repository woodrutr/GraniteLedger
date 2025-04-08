# src.integrator.gaussseidel

Iteratively solve 2 models with GS methodology

see README for process explanation

### Functions

| `convert_elec_price_to_lut`(prices)                      | convert electricity prices to dictionary, look up table                                                              |
|----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `convert_h2_price_records`(records)                      | simple coversion from list of records to a dictionary LUT repeat entries should not occur and will generate an error |
| `getLogger`([name])                                      | Return a logger with the specified name, creating it if necessary.                                                   |
| `get_elec_price`(instance[, block])                      | pulls hourly electricity prices from completed PowerModel and de-weights them.                                       |
| `init_old_cap`(instance)                                 | initialize capacity for 0th iteration                                                                                |
| `namedtuple`(typename, field_names, \*[, ...])           | Returns a new subclass of tuple with named fields.                                                                   |
| `plot_it`(OUTPUT_ROOT[, h2_price_records, ...])          | cheap plotter of iterative progress                                                                                  |
| `poll_h2_demand`(model)                                  | Get the hydrogen demand by rep_year and region                                                                       |
| `poll_h2_prices_from_elec`(model, tech, regions)         | poll the step-1 H2 price currently in the model for region/year, averaged over any steps                             |
| `poll_hydrogen_price`(model[, block])                    | Retrieve the price of H2 from the H2 model                                                                           |
| `regional_annual_prices`(m[, block])                     | pulls all regional annual weighted electricity prices                                                                |
| `run_elec_model`(settings[, solve])                      | build electricity model (and solve if solve=True) after passing in settings                                          |
| [`run_gs`](#src.integrator.gaussseidel.run_gs)(settings) | Start the iterative GS process                                                                                       |
| `select_solver`(instance)                                | Select solver based on learning method                                                                               |
| `set_new_cap`(instance)                                  | calculate new capacity after solve iteration                                                                         |
| `simple_solve`(m)                                        | a simple solve routine                                                                                               |
| `simple_solve_no_opt`(m, opt)                            | Solve concrete model using solver factory object                                                                     |
| `update_cost`(instance)                                  | update capital cost based on new capacity learning                                                                   |
| `update_h2_prices`(model, h2_prices)                     | Update the H2 prices held in the model                                                                               |

### Classes

| `EI`(region, year, hour)                       | (region, year, hour)                                              |
|------------------------------------------------|-------------------------------------------------------------------|
| `Path`(\*args, \*\*kwargs)                     | PurePath subclass that can make system calls.                     |
| `residentialModule`([settings, loadFile, ...]) | This contains the Residential model and its associated functions. |

### src.integrator.gaussseidel.run_gs(settings)

Start the iterative GS process

* **Parameters:**
  **settings** (*obj*) – Config_settings object that holds module choices and settings
