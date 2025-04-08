# src.models.hydrogen.network.registry

## REGISTRY CLASS

This class is the central registry of all objects in a grid. It preserves them in dicts of
object-name:object so that they can be looked up by name. it also should serve as a place to save
data in different configurations for faster parsing - for example, depth is a dict that organizes
regions according to their depth in the region nesting tree.

### Classes

| `Hub`(name, region[, data])                                    |    |
|----------------------------------------------------------------|----|
| `Region`(name[, grid, kind, data, parent])                     |    |
| [`Registry`](#src.models.hydrogen.network.registry.Registry)() |    |
| `TransportationArc`(origin, destination, capacity)             |    |

### *class* src.models.hydrogen.network.registry.Registry

#### add(thing)

add a thing to the registry. Thing can be Hub,Arc, or Region

* **Parameters:**
  **thing** (*Arc* *,* [*Region*](src.models.hydrogen.network.region.md#src.models.hydrogen.network.region.Region) *, or* [*Hub*](src.models.hydrogen.network.hub.md#src.models.hydrogen.network.hub.Hub)) – thing to add to registry
* **Returns:**
  thing being added gets returned
* **Return type:**
  Arc, [Region](src.models.hydrogen.network.region.md#src.models.hydrogen.network.region.Region), or [Hub](src.models.hydrogen.network.hub.md#src.models.hydrogen.network.hub.Hub)

#### remove(thing)

remove thing from registry

* **Parameters:**
  **thing** (*Arc* *,* [*Hub*](src.models.hydrogen.network.hub.md#src.models.hydrogen.network.hub.Hub) *, or* [*Region*](src.models.hydrogen.network.region.md#src.models.hydrogen.network.region.Region)) – thing to remove

#### update_levels()

update dictionary of regions by level
