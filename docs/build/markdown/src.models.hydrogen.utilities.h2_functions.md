# src.models.hydrogen.utilities.h2_functions

This file is a collection of functions that are used in support of the hydrogen model.

### Functions

| [`get_demand`](#src.models.hydrogen.utilities.h2_functions.get_demand)(hm, region, time)                                     | get demand for region at time.                            |
|------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| [`get_elec_price`](#src.models.hydrogen.utilities.h2_functions.get_elec_price)(hm, region, year)                             | get electricity price in region, year                     |
| [`get_electricity_consumption_rate`](#src.models.hydrogen.utilities.h2_functions.get_electricity_consumption_rate)(hm, tech) | the electricity consumption rate for technology type tech |
| [`get_electricty_consumption`](#src.models.hydrogen.utilities.h2_functions.get_electricty_consumption)(hm, region, year)     | get electricity consumption for region, year              |
| [`get_gas_price`](#src.models.hydrogen.utilities.h2_functions.get_gas_price)(hm, region, year)                               | get gas price for region, year                            |
| [`get_production_cost`](#src.models.hydrogen.utilities.h2_functions.get_production_cost)(hm, hub, tech, year)                | return production cost for tech at hub in year            |

### src.models.hydrogen.utilities.h2_functions.get_demand(hm: [H2Model](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model), region, time)

get demand for region at time. If mode not standard, just increase demand by 5% per year

* **Parameters:**
  * **hm** ([*H2Model*](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model)) – model
  * **region** (*str*) – region
  * **time** (*int*) – year
* **Returns:**
  demand
* **Return type:**
  float

### src.models.hydrogen.utilities.h2_functions.get_elec_price(hm: [H2Model](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model), region, year)

get electricity price in region, year

* **Parameters:**
  * **hm** ([*H2Model*](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model)) – \_model
  * **region** (*str*) – region
  * **year** (*int*) – year
* **Returns:**
  electricity price in region and year
* **Return type:**
  float

### src.models.hydrogen.utilities.h2_functions.get_electricity_consumption_rate(hm: [H2Model](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model), tech)

the electricity consumption rate for technology type tech

* **Parameters:**
  * **hm** ([*H2Model*](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model)) – model
  * **tech** (*str*) – technology type
* **Returns:**
  GWh per kg H2
* **Return type:**
  float

### src.models.hydrogen.utilities.h2_functions.get_electricty_consumption(hm: [H2Model](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model), region, year)

get electricity consumption for region, year

* **Parameters:**
  * **hm** ([*H2Model*](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model)) – model
  * **region** (*str*) – region
  * **year** (*int*) – year
* **Returns:**
  the elecctricity consumption for a region and year in the model
* **Return type:**
  float

### src.models.hydrogen.utilities.h2_functions.get_gas_price(hm: [H2Model](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model), region, year)

get gas price for region, year

* **Parameters:**
  * **hm** ([*H2Model*](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model)) – model
  * **region** (*str*) – region
  * **year** (*int*) – year
* **Returns:**
  gas price in region and year
* **Return type:**
  float

### src.models.hydrogen.utilities.h2_functions.get_production_cost(hm: [H2Model](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model), hub, tech, year)

return production cost for tech at hub in year

* **Parameters:**
  * **hm** ([*H2Model*](src.models.hydrogen.model.h2_model.md#src.models.hydrogen.model.h2_model.H2Model)) – model
  * **hub** (*str*) – hub
  * **tech** (*str*) – technology type
  * **year** (*int*) – year
* **Returns:**
  production cost of H2 for tech at hub in year
* **Return type:**
  float
