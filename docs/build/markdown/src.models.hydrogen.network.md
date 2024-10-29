# src.models.hydrogen.network package

## Submodules

## src.models.hydrogen.network.grid module

### GRID CLASS

> This is the central class that binds all the other classes together. No class
> instance exists in a reference that isn’t fundamentally contained in a grid.
> The grid is used to instantiate a model, read data, create the regionality
> and hub / arc network within that regionality, assign data to objects and more.

> notably, the grid is used to coordinate internal methods in various classes to
> make sure that their combined actions keep the model consistent and accomplish
> the desired task.
<!-- !! processed by numpydoc !! -->

### *class* src.models.hydrogen.network.grid.Grid(data: [GridData](#src.models.hydrogen.network.grid_data.GridData) | None = None)

Bases: `object`

<!-- !! processed by numpydoc !! -->

#### aggregate_hubs(hublist, region)

combine all hubs in hublist into a single hub, and place them in region. Arcs that connect to any of these hubs also get aggegated into arcs that connect to the new hub
: and their original origin / destination that’s not in hublist.

* **Parameters:**
  **hublist**
  : list of hubs to aggregate

  **region**
  : region to place them in

<!-- !! processed by numpydoc !! -->

#### arc_generation(df)

generate arcs from the arc data

* **Parameters:**
  **df**
  : arc data

<!-- !! processed by numpydoc !! -->

#### build_grid(vis=True)

builds a grid fom the GridData by recursively adding regions starting at top-level region
: ‘world’.

* **Parameters:**
  **vis**
  : if True, will generate an image of the hub-network with regional color-coding. Defaults to True.

<!-- !! processed by numpydoc !! -->

#### collapse(region_name)

make a region absorb all it’s sub-regions and combine all its and its childrens hubs into one

* **Parameters:**
  **region_name**
  : region to collapse

<!-- !! processed by numpydoc !! -->

#### collapse_level(level)

collapse all regions at a specific level of depth in the regional hierarchy, with world = 0

* **Parameters:**
  **level**
  : level to collapse

<!-- !! processed by numpydoc !! -->

#### combine_arcs(arclist, origin, destination)

combine a set of arcs into a single arc with given origin and destination

* **Parameters:**
  **arclist**
  : list of arcs to aggregate

  **origin**
  : new origin hub

  **destination**
  : new destination hub

<!-- !! processed by numpydoc !! -->

#### connect_subregions()

create an arc for all hubs in bottom-level regions to whatever hub is located in their parent region

<!-- !! processed by numpydoc !! -->

#### create_arc(origin, destination, capacity, cost=0.0)

Creates and arc from origin to destination with given capacity and cost

* **Parameters:**
  **origin**
  : origin hub name

  **destination**
  : destination hub name

  **capacity**
  : capacity of arc

  **cost**
  : cost of transporting 1kg H2 along arc. Defaults to 0.

<!-- !! processed by numpydoc !! -->

#### create_hub(name, region, data=None)

creates a hub in a given region

* **Parameters:**
  **name**
  : hub name

  **region**
  : Region hub is placed in

  **data**
  : dataframe of hub data to append. Defaults to None.

<!-- !! processed by numpydoc !! -->

#### create_region(name, parent=None, data=None)

creates a region with a given name, parent region, and data

* **Parameters:**
  **name**
  : name of region

  **parent**
  : parent region. Defaults to None.

  **data**
  : region data. Defaults to None.

<!-- !! processed by numpydoc !! -->

#### delete(thing)

deletes a hub, arc, or region

* **Parameters:**
  **thing**
  : thing to delete

<!-- !! processed by numpydoc !! -->

#### load_hubs()

load hubs from data

<!-- !! processed by numpydoc !! -->

#### recursive_region_generation(df, parent)

cycle through a region dataframe, left column to right until it hits data column, adding new regions and subregions according to how it is hierarchically structured.
: Future versions should implement this with a graph structure for the data instead of a dataframe, which would be more natural.

* **Parameters:**
  **df**
  : hierarchically structured dataframe of regions and their data.

  **parent**
  : Parent region

<!-- !! processed by numpydoc !! -->

#### test()

test run

<!-- !! processed by numpydoc !! -->

#### visualize()

visualize the grid network using graphx

<!-- !! processed by numpydoc !! -->

#### write_data()

\_write data to file

<!-- !! processed by numpydoc !! -->

## src.models.hydrogen.network.grid_data module

grid_data is the the data object that grids are generated from. It reads in raw data with a region 
filter, and holds it in one structure for easy access

<!-- !! processed by numpydoc !! -->

### *class* src.models.hydrogen.network.grid_data.GridData(data_folder: Path, regions_of_interest: list[str] | None = None)

Bases: `object`

<!-- !! processed by numpydoc !! -->

## src.models.hydrogen.network.hub module

### HUB CLASS

class objects are individual hubs, which are fundamental units of production in
the model. Hubs belong to regions, and connect to each other with transportation
arcs.

<!-- !! processed by numpydoc !! -->

### *class* src.models.hydrogen.network.hub.Hub(name, region, data=None)

Bases: `object`

<!-- !! processed by numpydoc !! -->

#### add_inbound(arc)

add an inbound arc to hub

* **Parameters:**
  **arc**
  : add an inbound arc to hub

<!-- !! processed by numpydoc !! -->

#### add_outbound(arc)

add an outbound arc to hub

* **Parameters:**
  **arc**
  : arc to add

<!-- !! processed by numpydoc !! -->

#### change_region(new_region)

move hub to new region

* **Parameters:**
  **new_region**
  : region hub should be moved to

<!-- !! processed by numpydoc !! -->

#### cost(technology, year)

return a cost value in terms of data fields

* **Parameters:**
  **technology**
  : technology type

  **year**
  : year
* **Returns:**
  float
  : a cost value

<!-- !! processed by numpydoc !! -->

#### display_outbound()

print all outbound arcs from hub

<!-- !! processed by numpydoc !! -->

#### get_data(quantity)

fetch quantity from hub data

* **Parameters:**
  **quantity**
  : name of data field to fetch
* **Returns:**
  float or str
  : quantity to be fetched

<!-- !! processed by numpydoc !! -->

#### remove_inbound(arc)

remove an inbound arc from hub

* **Parameters:**
  **arc**
  : arc to remove

<!-- !! processed by numpydoc !! -->

#### remove_outbound(arc)

remove an outbound arc from hub

* **Parameters:**
  **arc**
  : arc to remove

<!-- !! processed by numpydoc !! -->

## src.models.hydrogen.network.region module

### Region class:

> Class objects are regions, which have a natural tree-structure. Each region
> can have a parent region and child regions (subregions), a data object, and
> a set of hubs.
<!-- !! processed by numpydoc !! -->

### *class* src.models.hydrogen.network.region.Region(name, grid=None, kind=None, data=None, parent=None)

Bases: `object`

<!-- !! processed by numpydoc !! -->

#### absorb_subregions()

delete subregions, acquire their hubs and subregions

<!-- !! processed by numpydoc !! -->

#### absorb_subregions_deep()

absorb subregions recursively so that region becomes to the deepest level in the hierarchy

<!-- !! processed by numpydoc !! -->

#### add_hub(hub)

add a hub to region

* **Parameters:**
  **hub**
  : hub to add

<!-- !! processed by numpydoc !! -->

#### add_subregion(subregion)

make a region a subregion of self

* **Parameters:**
  **subregion**
  : new subregion

<!-- !! processed by numpydoc !! -->

#### aggregate_subregion_data(subregions)

combine the data from subregions and assign it to self

* **Parameters:**
  **subregions**
  : list of subregions

<!-- !! processed by numpydoc !! -->

#### assigned_names *= {}*

#### create_subregion(name, data=None)

create a subregion

* **Parameters:**
  **name**
  : subregion name

  **data**
  : subregion data. Defaults to None.

<!-- !! processed by numpydoc !! -->

#### delete()

delete self, reassign hubs to parent, reassign children to parent

<!-- !! processed by numpydoc !! -->

#### display_children()

display child regions

<!-- !! processed by numpydoc !! -->

#### display_hubs()

display hubs

<!-- !! processed by numpydoc !! -->

#### get_data(quantity)

pull data from region data

* **Parameters:**
  **quantity**
  : name of data field in region data
* **Returns:**
  str, float
  : value of data

<!-- !! processed by numpydoc !! -->

#### remove_hub(hub)

remove hub from region

* **Parameters:**
  **hub**
  : hub to remove

<!-- !! processed by numpydoc !! -->

#### remove_subregion(subregion)

remove a subregion from self

* **Parameters:**
  **subregion**
  : subregion to remove

<!-- !! processed by numpydoc !! -->

#### update_data(df)

change region data

* **Parameters:**
  **df**
  : new data

<!-- !! processed by numpydoc !! -->

#### update_parent(new_parent)

change parent region

* **Parameters:**
  **new_parent**
  : new parent region

<!-- !! processed by numpydoc !! -->

## src.models.hydrogen.network.registry module

### REGISTRY CLASS

> This class is the central registry of all objects in a grid. It preserves them
> in dicts of object-name:object so that they can be looked up by name.
> it also should serve as a place to save data in different configurations for
> faster parsing - for example, depth is a dict that organizes regions according to
> their depth in the region nesting tree.
<!-- !! processed by numpydoc !! -->

### *class* src.models.hydrogen.network.registry.Registry

Bases: `object`

<!-- !! processed by numpydoc !! -->

#### add(thing)

add a thing to the registry. Thing can be Hub,Arc, or Region

* **Parameters:**
  **thing**
  : thing to add to registry
* **Returns:**
  Arc, Region, or Hub
  : thing being added gets returned

<!-- !! processed by numpydoc !! -->

#### remove(thing)

remove thing from registry

* **Parameters:**
  **thing**
  : thing to remove

<!-- !! processed by numpydoc !! -->

#### update_levels()

update dictionary of regions by level

<!-- !! processed by numpydoc !! -->

## src.models.hydrogen.network.transportation_arc module

### TRANSPORTATION ARC CLASS

> objects in this class represent individual transportation arcs. An arc can
> exist with zero capacity, so they only represent *possible* arcs.
<!-- !! processed by numpydoc !! -->

### *class* src.models.hydrogen.network.transportation_arc.TransportationArc(origin, destination, capacity, cost=0)

Bases: `object`

<!-- !! processed by numpydoc !! -->

#### change_destination(new_destination)

change the destination hub of arc

* **Parameters:**
  **new_destination**
  : new destination hub

<!-- !! processed by numpydoc !! -->

#### change_origin(new_origin)

change the origin hub of arc

* **Parameters:**
  **new_origin**
  : new origin hub

<!-- !! processed by numpydoc !! -->

#### disconnect()

disconnect arc from it’s origin and destination

<!-- !! processed by numpydoc !! -->

## Module contents

<!-- !! processed by numpydoc !! -->
