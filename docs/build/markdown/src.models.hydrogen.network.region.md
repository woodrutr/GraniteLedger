# src.models.hydrogen.network.region

## REGION CLASS

Class objects are regions, which have a natural tree-structure. Each region can have a parent region
and child regions (subregions), a data object, and a set of hubs.

### Functions

| `getLogger`([name])   | Return a logger with the specified name, creating it if necessary.   |
|-----------------------|----------------------------------------------------------------------|

### Classes

| [`Region`](#src.models.hydrogen.network.region.Region)(name[, grid, kind, data, parent])   |    |
|--------------------------------------------------------------------------------------------|----|

### *class* src.models.hydrogen.network.region.Region(name, grid=None, kind=None, data=None, parent=None)

#### absorb_subregions()

delete subregions, acquire their hubs and subregions

#### absorb_subregions_deep()

absorb subregions recursively so that region becomes to the deepest level in the hierarchy

#### add_hub(hub)

add a hub to region

* **Parameters:**
  **hub** ([*Hub*](src.models.hydrogen.network.hub.md#src.models.hydrogen.network.hub.Hub)) – hub to add

#### add_subregion(subregion)

make a region a subregion of self

* **Parameters:**
  **subregion** ([*Region*](#src.models.hydrogen.network.region.Region)) – new subregion

#### aggregate_subregion_data(subregions)

combine the data from subregions and assign it to self

* **Parameters:**
  **subregions** (*list*) – list of subregions

#### assigned_names *= {}*

#### create_subregion(name, data=None)

create a subregion

* **Parameters:**
  * **name** (*str*) – subregion name
  * **data** (*DataFrame* *,* *optional*) – subregion data. Defaults to None.

#### delete()

delete self, reassign hubs to parent, reassign children to parent

#### display_children()

display child regions

#### display_hubs()

display hubs

#### get_data(quantity)

pull data from region data

* **Parameters:**
  **quantity** (*str*) – name of data field in region data
* **Returns:**
  value of data
* **Return type:**
  str, float

#### remove_hub(hub)

remove hub from region

* **Parameters:**
  **hub** ([*Hub*](src.models.hydrogen.network.hub.md#src.models.hydrogen.network.hub.Hub)) – hub to remove

#### remove_subregion(subregion)

remove a subregion from self

* **Parameters:**
  **subregion** ([*Region*](#src.models.hydrogen.network.region.Region)) – subregion to remove

#### update_data(df)

change region data

* **Parameters:**
  **df** (*DataFrame*) – new data

#### update_parent(new_parent)

change parent region

* **Parameters:**
  **new_parent** ([*Region*](#src.models.hydrogen.network.region.Region)) – new parent region
