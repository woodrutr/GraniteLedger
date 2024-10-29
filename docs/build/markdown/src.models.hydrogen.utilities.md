# src.models.hydrogen.utilities package

## Submodules

## src.models.hydrogen.utilities.h2_functions module

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.utilities.h2_functions.get_demand(hm: [H2Model](src.models.hydrogen.model.md#src.models.hydrogen.model.h2_model.H2Model), region, time)

get demand for region at time. If mode not standard, just increase demand by 5% per year

* **Parameters:**
  **hm**
  : model

  **region**
  : region

  **time**
  : year
* **Returns:**
  float
  : demand

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.utilities.h2_functions.get_elec_price(hm: [H2Model](src.models.hydrogen.model.md#src.models.hydrogen.model.h2_model.H2Model), region, year)

get electricity price in region, year
Parameters
———-
hm : H2Model

> \_model

region
: region

year
: year

* **Returns:**
  float
  : electricity price in region and year

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.utilities.h2_functions.get_electricity_consumption_rate(hm: [H2Model](src.models.hydrogen.model.md#src.models.hydrogen.model.h2_model.H2Model), tech)

the electricity consumption rate for technology type tech
Parameters
———-
hm : H2Model

> model

tech
: technology type

* **Returns:**
  float
  : GWh per kg H2

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.utilities.h2_functions.get_electricty_consumption(hm: [H2Model](src.models.hydrogen.model.md#src.models.hydrogen.model.h2_model.H2Model), region, year)

get electricity consumption for region, year
Parameters
———-
hm : H2Model

> model

region
: region

year
: year

* **Returns:**
  float
  : the elecctricity consumption for a region and year in the model

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.utilities.h2_functions.get_gas_price(hm: [H2Model](src.models.hydrogen.model.md#src.models.hydrogen.model.h2_model.H2Model), region, year)

get gas price for region, year

* **Parameters:**
  **hm**
  : model

  **region**
  : region

  **year**
  : year
* **Returns:**
  float
  : gas price in region and year

<!-- !! processed by numpydoc !! -->

### src.models.hydrogen.utilities.h2_functions.get_production_cost(hm: [H2Model](src.models.hydrogen.model.md#src.models.hydrogen.model.h2_model.H2Model), hub, tech, year)

return production cost for tech at hub in year

* **Parameters:**
  **hm**
  : model

  **hub**
  : hub

  **tech**
  : technology type

  **year**
  : year
* **Returns:**
  float
  : production cost of H2 for tech at hub in year

<!-- !! processed by numpydoc !! -->

## Module contents

<!-- !! processed by numpydoc !! -->
