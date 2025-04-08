# src.models.hydrogen.network.transportation_arc

## TRANSPORTATION ARC CLASS

objects in this class represent individual transportation arcs. An arc can exist with zero capacity,
so they only represent *possible* arcs.

### Classes

| [`TransportationArc`](#src.models.hydrogen.network.transportation_arc.TransportationArc)(origin, destination, capacity)   |    |
|---------------------------------------------------------------------------------------------------------------------------|----|

### *class* src.models.hydrogen.network.transportation_arc.TransportationArc(origin, destination, capacity, cost=0)

#### change_destination(new_destination)

change the destination hub of arc

* **Parameters:**
  **new_destination** ([*Hub*](src.models.hydrogen.network.hub.md#src.models.hydrogen.network.hub.Hub)) – new destination hub

#### change_origin(new_origin)

change the origin hub of arc

* **Parameters:**
  **new_origin** ([*Hub*](src.models.hydrogen.network.hub.md#src.models.hydrogen.network.hub.Hub)) – new origin hub

#### disconnect()

disconnect arc from it’s origin and destination
