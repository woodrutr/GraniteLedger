# src.models.hydrogen.network.grid

## GRID CLASS

> This is the central class that binds all the other classes together. No class
> instance exists in a reference that isn’t fundamentally contained in a grid.
> The grid is used to instantiate a model, read data, create the regionality
> and hub / arc network within that regionality, assign data to objects and more.

> notably, the grid is used to coordinate internal methods in various classes to
> make sure that their combined actions keep the model consistent and accomplish
> the desired task.

### Classes

| [`Grid`](#src.models.hydrogen.network.grid.Grid)([data])   |    |
|------------------------------------------------------------|----|
| `GridData`(data_folder[, regions_of_interest])             |    |
| `Hub`(name, region[, data])                                |    |
| `Region`(name[, grid, kind, data, parent])                 |    |
| `Registry`()                                               |    |
| `TransportationArc`(origin, destination, capacity)         |    |

### *class* src.models.hydrogen.network.grid.Grid(data: [GridData](src.models.hydrogen.network.grid_data.md#src.models.hydrogen.network.grid_data.GridData) | None = None)

#### aggregate_hubs(hublist, region)

combine all hubs in hublist into a single hub, and place them in region. Arcs that
connect to any of these hubs also get aggegated into arcs that connect to the new hub and
their original origin / destination that’s not in hublist.

* **Parameters:**
  * **hublist** (*list*) – list of hubs to aggregate
  * **region** ([*Region*](src.models.hydrogen.network.region.md#src.models.hydrogen.network.region.Region)) – region to place them in

#### arc_generation(df)

generate arcs from the arc data

* **Parameters:**
  **df** (*DataFrame*) – arc data

#### build_grid(vis=True)

builds a grid fom the GridData by recursively adding regions starting at top-level region
‘world’.

* **Parameters:**
  **vis** (*bool* *,* *optional*) – if True, will generate an image of the hub-network with regional color-coding. Defaults to True.

#### collapse(region_name)

make a region absorb all it’s sub-regions and combine all its and its childrens hubs into one

* **Parameters:**
  **region_name** (*str*) – region to collapse

#### collapse_level(level)

collapse all regions at a specific level of depth in the regional hierarchy, with world = 0

* **Parameters:**
  **level** (*int*) – level to collapse

#### combine_arcs(arclist, origin, destination)

combine a set of arcs into a single arc with given origin and destination

* **Parameters:**
  * **arclist** (*list*) – list of arcs to aggregate
  * **origin** (*str*) – new origin hub
  * **destination** (*str*) – new destination hub

#### connect_subregions()

create an arc for all hubs in bottom-level regions to whatever hub is located in their
parent region

#### create_arc(origin, destination, capacity, cost=0.0)

Creates and arc from origin to destination with given capacity and cost

* **Parameters:**
  * **origin** (*str*) – origin hub name
  * **destination** (*str*) – destination hub name
  * **capacity** (*float*) – capacity of arc
  * **cost** (*float* *,* *optional*) – cost of transporting 1kg H2 along arc. Defaults to 0.

#### create_hub(name, region, data=None)

creates a hub in a given region

* **Parameters:**
  * **name** (*str*) – hub name
  * **region** ([*Region*](src.models.hydrogen.network.region.md#src.models.hydrogen.network.region.Region)) – Region hub is placed in
  * **data** (*DataFrame* *,* *optional*) – dataframe of hub data to append. Defaults to None.

#### create_region(name, parent=None, data=None)

creates a region with a given name, parent region, and data

* **Parameters:**
  * **name** (*str*) – name of region
  * **parent** ([*Region*](src.models.hydrogen.network.region.md#src.models.hydrogen.network.region.Region) *,* *optional*) – parent region. Defaults to None.
  * **data** (*DataFrame* *,* *optional*) – region data. Defaults to None.

#### delete(thing)

deletes a hub, arc, or region

* **Parameters:**
  **thing** ([*Hub*](src.models.hydrogen.network.hub.md#src.models.hydrogen.network.hub.Hub) *,* *Arc* *, or* [*Region*](src.models.hydrogen.network.region.md#src.models.hydrogen.network.region.Region)) – thing to delete

#### load_hubs()

load hubs from data

#### recursive_region_generation(df, parent)

cycle through a region dataframe, left column to right until it hits data column, adding
new regions and subregions according to how it is hierarchically structured. Future versions
should implement this with a graph structure for the data instead of a dataframe, which
would be more natural.

* **Parameters:**
  * **df** (*DataFrame*) – hierarchically structured dataframe of regions and their data.
  * **parent** ([*Region*](src.models.hydrogen.network.region.md#src.models.hydrogen.network.region.Region)) – Parent region

#### test()

test run

#### visualize()

visualize the grid network using graphx

#### write_data()

\_write data to file
