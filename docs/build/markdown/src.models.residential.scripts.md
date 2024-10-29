# src.models.residential.scripts package

## Submodules

## src.models.residential.scripts.residential module

Residential Model.
This file contains the residentialModule class which contains a representation of residential
electricity prices and demands.

<!-- !! processed by numpydoc !! -->

### *class* src.models.residential.scripts.residential.residentialModule(settings: [Config_settings](src.integrator.md#src.integrator.config_setup.Config_settings) | None = None, loadFile: str | None = None, load_df: DataFrame | None = None, calibrate: bool | None = False)

Bases: `object`

This contains the Residential model and its associated functions. Once an object is instantiated, it can calculate new Load values for updated prices.
It can also calculate estimated changes to the Load if one of the input variables is changed by a specified percent.
The model will be created in a symbolic form to be easily manipulated, and then values can be filled in for calculations.

<!-- !! processed by numpydoc !! -->

#### baseYear *= 0*

#### complex_step_sensitivity(prices, change_var, percent)

This estimates how much the output Load will change due to a change in one of the input variables.
It can calculate these values for changes in price, price elasticity, income, income elasticity, or long term trend.
The Load calculation requires input prices, so this function requires that as well for the base output Load.
Then, an estimate for Load is calculated for the case where the named ‘change_var’ is changed by ‘percent’ %.

* **Parameters:**
  **prices**
  : Price values used to calculate the Load value

  **change_var**
  : Name of variable of interest for sensitivity. This can be:
    : ‘income’, ‘i_elas’, ‘price’, ‘p_elas’, ‘trendGR’

  **percent**
  : A value 0 - 100 for the percent that the variable of interest can change.
* **Returns:**
  dataframe
  : Indexed values for the calculated Load at the given prices, the Load if the variable of interest is increased by ‘percent’%, and the Load if the variable of interest is decreased by ‘percent’%

<!-- !! processed by numpydoc !! -->

#### hr_map *= Empty DataFrame Columns: [] Index: []*

#### loads *= {}*

#### make_block(prices, pricesindex)

Updates the value of ‘Load’ based on the new prices given.
The new prices are fed into the equations from the residential model.
The new calculated Loads are used to constrain ‘Load’ in pyomo blocks.

* **Parameters:**
  **prices**
  : Pyomo Parameter of newly updated prices

  **pricesindex**
  : Pyomo Set of indexes that matches the prices given
* **Returns:**
  pyo.Block
  : Block containing constraints that set ‘Load’ variable equal to the updated load values

<!-- !! processed by numpydoc !! -->

#### prices *= {}*

#### sensitivity(prices, change_var, percent)

This estimates how much the output Load will change due to a change in one of the input variables.
It can calculate these values for changes in price, price elasticity, income, income elasticity, or long term trend.
The Load calculation requires input prices, so this function requires that as well for the base output Load.
Then, an estimate for Load is calculated for the case where the named ‘change_var’ is changed by ‘percent’ %.

* **Parameters:**
  **prices**
  : Price values used to calculate the Load value

  **change_var**
  : Name of variable of interest for sensitivity. This can be:
    : ‘income’, ‘i_elas’, ‘price’, ‘p_elas’, ‘trendGR’

  **percent**
  : A value 0 - 100 for the percent that the variable of interest can change.
* **Returns:**
  dataframe
  : Indexed values for the calculated Load at the given prices, the Load if the variable of interest is increased by ‘percent’%, and the Load if the variable of interest is decreased by ‘percent’%

<!-- !! processed by numpydoc !! -->

#### update_load(p)

Takes in Dual pyomo Parameters or dataframes to update Load values

* **Parameters:**
  **p**
  : Pyomo Parameter or dataframe of newly updated prices from Duals
* **Returns:**
  pandas DataFrame
  : Load values indexed by region, year, and hour

<!-- !! processed by numpydoc !! -->

#### view_output_load(values: DataFrame, regions: list[int] = [1], years: list[int] = [2023])

This is used to display the updated Load values after calculation. It will create a graph for each region and year combination.

* **Parameters:**
  **values**
  : The Load values calculated in update_load

  **regions**
  : The regions to be displayed

  **years**
  : The years to be displayed

<!-- !! processed by numpydoc !! -->

#### view_sensitivity(values: DataFrame, regions: list[int] = [1], years: list[int] = [2023])

This is used by the sensitivity method to display graphs of the calculated values

* **Parameters:**
  **values**
  : indexed values for the Load, upper change, and lower change

  **regions**
  : regions to be graphed

  **years**
  : years to be graphed

<!-- !! processed by numpydoc !! -->

### src.models.residential.scripts.residential.run_residential(settings: [Config_settings](src.integrator.md#src.integrator.config_setup.Config_settings))

This runs the residential model in stand-alone mode. It can run update_load to calculate new Load values based on prices,
or it can calculate the new Load value along with estimates for the Load if one of the input variables changes.

* **Parameters:**
  **settings**
  : information given from run_config to set several values

<!-- !! processed by numpydoc !! -->

## Module contents

<!-- !! processed by numpydoc !! -->
