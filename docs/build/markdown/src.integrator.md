# src.integrator package

## Submodules

## src.integrator.config_setup module

This file contains Config_settings class. It establishes the main settings used when running
the model. It takes these settings from the run_config.toml file. It contains universal configurations
(e.g., configs that cut across modules and/or solve options) and module specific configs.

<!-- !! processed by numpydoc !! -->

### *class* src.integrator.config_setup.Config_settings(config_path: Path, args: Namespace | None = None, test=False, years_ow=[], regions_ow=[])

Bases: `object`

Generates the model settings that are used to solve. Settings include:
- Iterative Solve Config Settings
- Spatial Config Settings
- Temporal Config Settings
- Electricity Config Settings
- Other

<!-- !! processed by numpydoc !! -->

## src.integrator.gaussseidel module

Iteratively solve 2 models with GS methodology

see README for process explanation

<!-- !! processed by numpydoc !! -->

### src.integrator.gaussseidel.run_gs(settings)

Start the iterative GS process

* **Parameters:**
  **settings**
  : Config_settings object that holds module choices and settings

<!-- !! processed by numpydoc !! -->

## src.integrator.progress_plot module

A plotter that can be used for combined solves

<!-- !! processed by numpydoc !! -->

### src.integrator.progress_plot.plot_it(h2_price_records=[], elec_price_records=[], h2_obj_records=[], elec_obj_records=[], h2_demand_records=[], elec_demand_records=[], load_records=[], elec_price_to_res_records=[])

cheap plotter of iterative progress

<!-- !! processed by numpydoc !! -->

### src.integrator.progress_plot.plot_price_distro(price_records: list[float])

cheap/quick analyisis and plot of the price records

<!-- !! processed by numpydoc !! -->

## src.integrator.runner module

A gathering of functions for running models solo

<!-- !! processed by numpydoc !! -->

### src.integrator.runner.run_elec_solo(settings: [Config_settings](#src.integrator.config_setup.Config_settings) | None = None)

Runs electricity model by itself as defined in settings

### Parameters

settings: Config_settings
: Contains configuration settings for which regions, years, and switches to run

<!-- !! processed by numpydoc !! -->

### src.integrator.runner.run_h2_solo(settings: [Config_settings](#src.integrator.config_setup.Config_settings) | None = None)

Runs hydrogen model by itself as defined in settings

### Parameters

settings: Config_settings
: Contains configuration settings for which regions and years to run

<!-- !! processed by numpydoc !! -->

### src.integrator.runner.run_residential_solo(settings: [Config_settings](#src.integrator.config_setup.Config_settings) | None = None)

Runs residential model by itself as defined in settings

### Parameters

settings: Config_settings
: Contains configuration settings for which regions and years to run

<!-- !! processed by numpydoc !! -->

### src.integrator.runner.run_standalone(settings: [Config_settings](#src.integrator.config_setup.Config_settings))

Runs standalone methods based on settings selections; running 1 or more modules

* **Parameters:**
  **settings**
  : Instance of config_settings containing run options, mode and settings

<!-- !! processed by numpydoc !! -->

## src.integrator.unified module

Unifying the solve of both H2 and Elec and Res

Dev Notes:

(1).  The “annual demand” constraint that is present and INACTIVE 
: > is omitted here for clarity.  It may likely be needed–in some form–at a later
  <br/>
  time.  Recall, the key linkages to share the electrical demand primary variable are:
  : (a).  an annual level demand constraint 
    (b).  an accurate price-pulling function that can consider weighted duals
    <br/>
    > from both constraints [NOT done]

(2).  This model has a 2-solve update cycle as commented on near the termination check
: elec_prices gleaned from      cycle[n] results -> solve cycle[n+1]
  new_load gleaned from         cycle[n+1] results -> solve cycle[n+2]
  elec_pices gleaned from       cycle[n+2]

<!-- !! processed by numpydoc !! -->

### src.integrator.unified.run_unified(settings: [Config_settings](#src.integrator.config_setup.Config_settings))

Runs unified solve method based on

* **Parameters:**
  **settings**
  : Instance of config_settings containing run options, mode and settings

<!-- !! processed by numpydoc !! -->

## src.integrator.utilities module

A gathering of utility functions for dealing with model interconnectivity

Dev Note:  At some review point, some decisions may move these back & forth with parent
models after it is decided if it is a utility job to do …. or a class method.

Additionally, there is probably some renaming due here for consistency

<!-- !! processed by numpydoc !! -->

### *class* src.integrator.utilities.EI(region, year, hour)

Bases: `tuple`

(region, year, hour)

<!-- !! processed by numpydoc !! -->

#### hour

Alias for field number 2

<!-- !! processed by numpydoc !! -->

#### region

Alias for field number 0

<!-- !! processed by numpydoc !! -->

#### year

Alias for field number 1

<!-- !! processed by numpydoc !! -->

### *class* src.integrator.utilities.HI(region, year)

Bases: `tuple`

(region, year)

<!-- !! processed by numpydoc !! -->

#### region

Alias for field number 0

<!-- !! processed by numpydoc !! -->

#### year

Alias for field number 1

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.convert_elec_price_to_lut(prices: list[tuple[[EI](#src.integrator.utilities.EI), float]])

convert electricity prices to dictionary, look up table

* **Parameters:**
  **prices**
  : list of prices
* **Returns:**
  dict[EI, float]
  : dict of prices

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.convert_h2_price_records(records: list[tuple[[HI](#src.integrator.utilities.HI), float]])

simple coversion from list of records to a dictionary LUT
repeat entries should not occur and will generate an error

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.create_temporal_mapping(sw_temporal)

Combines the input mapping files within the electricity model to create a master temporal
mapping dataframe. The df is used to build multiple temporal parameters used within the  model.
It creates a single dataframe that has 8760 rows for each hour in the year.
Each hour in the year is assigned a season type, day type, and hour type used in the model.
This defines the number of time periods the model will use based on cw_s_day and cw_hr inputs.

* **Returns:**
  dataframe
  : a dataframe with 8760 rows that include each hour, hour type, day, day type, and season.
    It also includes the weights for each day type and hour type.

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.get_annual_wt_avg(elec_price: DataFrame)

takes annual weighted average of hourly electricity prices

* **Parameters:**
  **elec_price**
  : hourly electricity prices
* **Returns:**
  dict[HI, float]
  : annual weighted average electricity prices

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.get_elec_price(instance: [PowerModel](src.models.electricity.scripts.md#src.models.electricity.scripts.electricity_model.PowerModel) | ConcreteModel, block=None)

pulls hourly electricity prices from completed PowerModel and de-weights them

Prices from the duals are weighted by the day and year weights applied in the OBJ function
This function retrieves the prices for all hours and removes the day and annual weights to
return raw prices (and the day weights to use as needed)

* **Parameters:**
  **instance**
  : solved electricity model

  **block: ConcreteModel**
  : reference to the block if the electricity model is a block within a larger model
* **Returns:**
  pd.DataFrame
  : df of raw prices and the day weights to re-apply (if needed)
    columns: [r, y, hour, day_weight, raw_price]

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.get_output_root()

get the name of the output dir, which includes the name of the mode type and a timestamp

* **Returns:**
  path
  : output directory path

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.make_dir(dir_name)

generates an output directory to write model results, output directory is the date/time
at the time this function executes. It includes subdirs for vars, params, constraints.

* **Returns:**
  string
  : the name of the output directory

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.poll_h2_demand(model: [PowerModel](src.models.electricity.scripts.md#src.models.electricity.scripts.electricity_model.PowerModel))

Get the hydrogen demand by rep_year and region

Use the Generation variable for h2 techs

NOTE:  Not sure about day weighting calculation here!!

* **Returns:**
  dict[HI, float]
  : dictionary of prices by H2 Index: price

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.poll_h2_prices_from_elec(model: [PowerModel](src.models.electricity.scripts.md#src.models.electricity.scripts.electricity_model.PowerModel), tech, regions: Iterable)

poll the step-1 H2 price currently in the model for region/year, averaged over any steps

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.poll_hydrogen_price(model: [H2Model](src.models.hydrogen.model.md#src.models.hydrogen.model.h2_model.H2Model) | ConcreteModel, block=None)

Retrieve the price of H2 from the H2 model

* **Parameters:**
  **model**
  : the model to poll

  **block: optional**
  : block model to poll
* **Returns:**
  list[tuple[HI, float]]
  : list of H2 Index, price tuples

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.poll_year_avg_elec_price(price_list: list[tuple[[EI](#src.integrator.utilities.EI), float]])

retrieve a REPRESENTATIVE price at the annual level from a listing of prices

This function computes the AVERAGE elec price for each region-year combo

* **Parameters:**
  **price_list**
  : input price list
* **Returns:**
  dict[HI, float]
  : a dictionary of (region, year): price

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.regional_annual_prices(m: [PowerModel](src.models.electricity.scripts.md#src.models.electricity.scripts.electricity_model.PowerModel) | ConcreteModel, block=None)

pulls all regional annual weighted electricity prices

* **Parameters:**
  **m**
  : solved PowerModel

  **block**
  : solved block model if applicable, by default None
* **Returns:**
  dict[HI, float]
  : dict with regional annual electricity prices

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.select_solver(instance: ConcreteModel)

Select solver based on learning method

* **Parameters:**
  **instance**
  : electricity pyomo model
* **Returns:**
  solver type (?)
  : The pyomo solver

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.setup_logger(output_dir)

initiates logging, sets up logger in the output directory specified

* **Parameters:**
  **output_dir**
  : output directory path

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.simple_solve(m: ConcreteModel)

a simple solve routine

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.simple_solve_no_opt(m: ~pyomo.core.base.PyomoModel.ConcreteModel, opt: <pyomo.opt.base.solvers.SolverFactoryClass object at 0x00000210E1538590>)

Solve concrete model using solver factory object

* **Parameters:**
  **m**
  : Pyomo model

  **opt: SolverFactory**
  : Solver object initiated prior to solve

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.update_elec_demand(self, elec_demand: dict[[HI](#src.integrator.utilities.HI), float])

Update the external electical demand parameter with demands from the H2 model

* **Parameters:**
  **elec_demand**
  : the new demands broken out by hyd index (region, year)

<!-- !! processed by numpydoc !! -->

### src.integrator.utilities.update_h2_prices(model: [PowerModel](src.models.electricity.scripts.md#src.models.electricity.scripts.electricity_model.PowerModel), h2_prices: dict[[HI](#src.integrator.utilities.HI), float])

Update the H2 prices held in the model

* **Parameters:**
  **h2_prices**
  : new prices

<!-- !! processed by numpydoc !! -->

## Module contents

<!-- !! processed by numpydoc !! -->
