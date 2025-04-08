# src.models.hydrogen.network.hub

## HUB CLASS

class objects are individual hubs, which are fundamental units of production in the model. Hubs
belong to regions, and connect to each other with transportation arcs.

### Classes

| [`Hub`](#src.models.hydrogen.network.hub.Hub)(name, region[, data])   |    |
|-----------------------------------------------------------------------|----|

### *class* src.models.hydrogen.network.hub.Hub(name, region, data=None)

#### add_inbound(arc)

add an inbound arc to hub

* **Parameters:**
  **arc** (*Arc*) – add an inbound arc to hub

#### add_outbound(arc)

add an outbound arc to hub

* **Parameters:**
  **arc** (*Arc*) – arc to add

#### change_region(new_region)

move hub to new region

* **Parameters:**
  **new_region** ([*Region*](src.models.hydrogen.network.region.md#src.models.hydrogen.network.region.Region)) – region hub should be moved to

#### cost(technology, year)

return a cost value in terms of data fields

* **Parameters:**
  * **technology** (*str*) – technology type
  * **year** (*int*) – year
* **Returns:**
  a cost value
* **Return type:**
  float

#### display_outbound()

print all outbound arcs from hub

#### get_data(quantity)

fetch quantity from hub data

* **Parameters:**
  **quantity** (*str*) – name of data field to fetch
* **Returns:**
  quantity to be fetched
* **Return type:**
  float or str

#### remove_inbound(arc)

remove an inbound arc from hub

* **Parameters:**
  **arc** (*Arc*) – arc to remove

#### remove_outbound(arc)

remove an outbound arc from hub

* **Parameters:**
  **arc** (*Arc*) – arc to remove
