# src.integrator.unified

Unifying the solve of both H2 and Elec and Res

Dev Notes:

1. The “annual demand” constraint that is present and INACTIVE is omitted here for clarity.
It may likely be needed - in some form - at a later time. Recall, the key linkages to share the
electrical demand primary variable are:

> - an annual level demand constraint
> - an accurate price-pulling function that can consider weighted duals from both constraints
1. This model has a 2-solve update cycle as commented on near the termination check

> - elec_prices gleaned from cycle[n] results -> solve cycle[n+1]
> - new_load gleaned from cycle[n+1] results -> solve cycle[n+2]
> - elec_pices gleaned from cycle[n+2]

### Functions

| `convert_elec_price_to_lut`(prices)                            | convert electricity prices to dictionary, look up table                                                              |
|----------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `convert_h2_price_records`(records)                            | simple coversion from list of records to a dictionary LUT repeat entries should not occur and will generate an error |
| `getLogger`([name])                                            | Return a logger with the specified name, creating it if necessary.                                                   |
| `get_elec_price`(instance[, block])                            | pulls hourly electricity prices from completed PowerModel and de-weights them.                                       |
| `init_old_cap`(instance)                                       | initialize capacity for 0th iteration                                                                                |
| `poll_h2_demand`(model)                                        | Get the hydrogen demand by rep_year and region                                                                       |
| `poll_hydrogen_price`(model[, block])                          | Retrieve the price of H2 from the H2 model                                                                           |
| `regional_annual_prices`(m[, block])                           | pulls all regional annual weighted electricity prices                                                                |
| `run_elec_model`(settings[, solve])                            | build electricity model (and solve if solve=True) after passing in settings                                          |
| [`run_unified`](#src.integrator.unified.run_unified)(settings) | Runs unified solve method based on                                                                                   |
| `select_solver`(instance)                                      | Select solver based on learning method                                                                               |
| `set_new_cap`(instance)                                        | calculate new capacity after solve iteration                                                                         |
| `simple_solve`(m)                                              | a simple solve routine                                                                                               |
| `simple_solve_no_opt`(m, opt)                                  | Solve concrete model using solver factory object                                                                     |
| `update_cost`(instance)                                        | update capital cost based on new capacity learning                                                                   |
| `update_h2_prices`(model, h2_prices)                           | Update the H2 prices held in the model                                                                               |

### Classes

| `Config_settings`(config_path[, args, test])   | Generates the model settings that are used to solve.                      |
|------------------------------------------------|---------------------------------------------------------------------------|
| `EI`(region, year, hour)                       | (region, year, hour)                                                      |
| `HI`(region, year)                             | (region, year)                                                            |
| `defaultdict`                                  | defaultdict(default_factory=None, /, [...]) --> dict with default factory |
| `deque`                                        | deque([iterable[, maxlen]]) --> deque object                              |
| `residentialModule`([settings, loadFile, ...]) | This contains the Residential model and its associated functions.         |

### src.integrator.unified.run_unified(settings: [Config_settings](src.common.config_setup.md#src.common.config_setup.Config_settings))

Runs unified solve method based on

* **Parameters:**
  **settings** ([*Config_settings*](src.common.config_setup.md#src.common.config_setup.Config_settings)) – Instance of config_settings containing run options, mode and settings
