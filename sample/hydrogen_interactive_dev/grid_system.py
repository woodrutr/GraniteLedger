'''
Region class:
~~~~~~~~~~~~
    
    Class objects are regions, which have a natural tree-structure. Each region
    can have a parent region and child regions (subregions), a data object, and 
    a set of hubs.

HUB CLASS
~~~~~~~~~

class objects are individual hubs, which are fundamental units of production in
the model. Hubs belong to regions, and connect to each other with transportation
arcs.

TRANSPORTATION ARC CLASS
~~~~~~~~~~~~~~~~~~~~~~~~
    
    objects in this class represent individual transportation arcs. An arc can
    exist with zero capacity, so they only represent *possible* arcs.

REGISTRY CLASS
~~~~~~~~~~~~~~
     
    This class is the central registry of all objects in a grid. It preserves them 
    in dicts of object-name:object so that they can be looked up by name.
    it also should serve as a place to save data in different configurations for
    faster parsing - for example, depth is a dict that organizes regions according to
    their depth in the region nesting tree.


'''
###################################################################################################
# Setup

# Import packages and scripts
import numpy as np
from pyomo.environ import *
# TODO: redo pyomo wildcard import to avoid sphinx issues
import highspy as hp
import pandas as pd
import plotly
import plotly.express as px
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
import time

###################################################################################################



class Region:
    def __init__(self, name, grid=None, data=None, parent=None):
        """ initialize region with name in grid

        Args:
            name (str): name
            grid (Grid, optional): grid it belongs to. Defaults to None.
            data (DataFrame, optional): region data. Defaults to None.
            parent (Grid, optional): parent grid. Defaults to None.
        """        
        self.name = name
        self.parent = parent
        self.children = {}
        self.hubs = {}
        self.data = data

        if self.parent != None:
            self.depth = self.parent.depth + 1
            self.grid = parent.grid

        else:
            self.depth = 0
            self.grid = grid

    def display_children(self):
        """prints a list of subregions
        """        
        for child in self.children.values():
            print(child.name, child.depth)
            child.display_children()

    def display_hubs(self):
        """prints a list of hubs in the region
        """        
        for hub in self.hubs.values():
            print(hub.name)

    def update_parent(self, new_parent):
        """changes the parent region (used in aggregation)

        Args:
            new_parent (Region): new parent region
        """        
        if self.parent != None:
            del self.parent.children[self.name]
            self.parent = new_parent
            self.parent.add_subregion(self)
            self.depth = new_parent.depth + 1

        else:
            self.parent = new_parent
            self.parent.add_subregion(self)

    def create_subregion(self, name, data=None):
        """creates a subregion with given name and data

        Args:
            name (str): name
            data (DataFrame, optional): new data object to append to subregion. Defaults to None.
        """        
        self.grid.create_region(name, self, data)

    def add_subregion(self, subregion):
        """takes an existing region as an arg and assigns it as a subregion

        Args:
            subregion (Region): region to add
        """        
        self.children.update({subregion.name: subregion})

    def remove_subregion(self, subregion):
        """removes subregion from children

        Args:
            subregion (Region): subregion to remove
        """        
        self.children.pop(subregion.name)

    def add_hub(self, hub):
        """add a hub to hubslist
        Args:
            hub Hub): hub to add
        """        
        self.hubs.update({hub.name: hub})

    def remove_hub(self, hub):
        """removes hub from hublist

        Args:
            hub (Hub): hub to remove
        """        
        del self.hubs[hub.name]

    def delete(self):
        """deletes region from all parent/child regional references and moves hubs to parent
        """        
        for hub in self.hubs.values():
            hub.change_region(self.parent)

        for child in list(self.children.values()):
            child.update_parent(self.parent)
            self.parent.add_subregion(child)

        if self.name in self.parent.children.keys():
            self.parent.remove_subregion(self)

    def absorb_subregions(self):
        """absorb subregions by deleting them and therefore gaining their hubs. If self has no data, aggregate child regions data and acquire it
        """        
        subregions = list(self.children.values())

        if self.data is None:
            self.aggregate_subregion_data(subregions)

        for subregion in subregions:
            self.grid.delete(subregion)

        del subregions

    def absorb_subregions_deep(self):
        """absorb subregions and also their subregions etc
        """        
        subregions = list(self.children.values())

        for subregion in subregions:
            subregion.absorb_subregions_deep()
            print('deleting: ', subregion.name)
            if self.data is None:
                self.aggregate_subregion_data(subregions)
            self.grid.delete(subregion)

        del subregions

    def update_data(self, df):
        """update region data with df

        Args:
            df (DataFrame): new region data
        """        
        self.data = df

    def aggregate_subregion_data(self, subregions):
        """ aggregate data from subregions according to summable / meanable classification

        Args:
            subregions (iterable): iterable of subregions to aggregate
        """        
        temp_child_data = pd.concat([region.data for region in subregions], axis=1).transpose()
        new_data = pd.DataFrame(
            columns=self.grid.data.summable['region'] + self.grid.data.meanable['region']
        )

        for column in temp_child_data.columns:
            if column in self.grid.data.summable['region']:
                new_data[column] = [temp_child_data[column].sum()]
            if column in self.grid.data.meanable['region']:
                new_data[column] = [temp_child_data[column].mean()]

        self.update_data(new_data.squeeze())

    def get_data(self, quantity):
        """ fetch data from region data

        Args:
            quantity (str): data column to fetch

        Returns:
            str, float: content of quantity cell
        """        
        return self.data[quantity]


class Hub:
    def __init__(self, name, region, data=None):
        """ initialize a hub with name, located in region and data

        Args:
            name (str): hub name
            region (Region): region located in
            data (DataFrame, optional): data for hub. Defaults to None.
        """        
        self.name = name
        self.region = region
        self.data = data.fillna(0)

        # outbound and inbound dictionaries mapping names of hubs to the arc objects
        self.outbound = {}
        self.inbound = {}

        self.x = data.iloc[0]['x']
        self.y = data.iloc[0]['y']

    def change_region(self, new_region):
        """move hub to new region

        Args:
            new_region (Region): new home region
        """        
        self.region = new_region
        new_region.add_hub(self)

    def display_outbound(self):
        """display outbound transportation arcs
        """        
        for arc in self.outbound.values():
            print('name:', arc.origin.name, 'capacity:', arc.capacity)

    
    #Add and remove arc functions
    #only modifies itself

    def add_outbound(self, arc):
        """add an outbound arc

        Args:
            arc (TransportationArc): arc to add
        """        
        self.outbound[arc.destination.name] = arc

    def add_inbound(self, arc):
        """add inbound arc
        Args:
            arc (TransportationArc): arc to add
        """        
        self.inbound[arc.origin.name] = arc

    def remove_outbound(self, arc):
        """remove an outbound arc

        Args:
            arc (TransportationArc): arc to remove
        """        
        del self.outbound[arc.destination.name]

    def remove_inbound(self, arc):
        del self.inbound[arc.origin.name]

    def get_data(self, quantity):
        """fetch data 

        Args:
            quantity (str): column name for quantity

        Returns:
            str, float: the quantity
        """        
        return self.data.iloc[0][quantity]

    def cost(self, technology, year):
        """ return a base cost for a technology type as a function (can be function of any hub or regional data)

        Args:
            technology (str): technology type
            year (int): the year to calculate cost for

        Returns:
            float: variable production cost for a unit of product
        """        

        # any function can be inserted here, the idea is to have hubs be able to refer to their regional data to incorporate in function
        if technology == 'PEM':
            return self.region.data['electricity_cost'] * 45
        elif technology == 'SMR':
            return self.region.data['gas_cost']
        else:
            return 0


class TransportationArc:
    def __init__(self, origin, destination, capacity, cost=0):
        """ initialize a transportation arc from origin to destination with given capacity and cost

        Args:
            origin (Hub): origin hub
            destination (Hub): destination hub
            capacity (float): capacity of arc
            cost (float, optional): cost to transport 1 unit of product across arc. Defaults to 0.
        """        
        self.name = (origin.name, destination.name)
        self.origin = origin
        self.destination = destination
        self.capacity = capacity
        self.cost = cost

    def change_origin(self, new_origin):
        """ change the origin hub to new_origin

        Args:
            new_origin (Hub): new origin hub
        """        
        self.name = (new_origin.name, self.name[1])
        self.origin = new_origin

    def change_destination(self, new_destination):
        """ change the destination hub to new_destination

        Args:
            new_destination (Hub): new destination hub
        """        
        self.name = (self.name[0], new_destination.name)
        self.destination = new_destination

    def disconnect(self):
        """ disconnect an arc from its endpoints
        """        
        self.origin.remove_outbound(self)
        self.destination.remove_inbound(self)



class Registry:
    def __init__(self):
        """ initialize a registry for a grid.
        """        
        self.regions = {}
        self.depth = {i: [] for i in range(10)}
        self.hubs = {}
        self.arcs = {}
        self.max_depth = 0

    def add(self, thing):
        """ add a thing to the registry

        Args:
            thing (Region, Hub, Arc): thing to add

        Returns:
            Region, Hub, Arc: the thing added
        """        
        if type(thing) == Hub:
            self.hubs[thing.name] = thing
            return thing
        elif type(thing) == TransportationArc:
            self.arcs[thing.name] = thing
            return thing
        elif type(thing) == Region:
            self.regions[thing.name] = thing
            self.depth[thing.depth].append(thing.name)
            if thing.depth > self.max_depth:
                self.max_depth = thing.depth
            return thing

    def remove(self, thing):
        """ remove a thing from the registry

        Args:
            thing (Region, Hub, Arc): thing to remove
        """        
        if type(thing) == Hub:
            del self.hubs[thing.name]
        elif type(thing) == Region:
            del self.regions[thing.name]

        elif type(thing) == TransportationArc:
            del self.arcs[thing.name]

    def update_levels(self):
        """ update the dictionary of regional nesting levels to regions
        """        
        self.depth = {i: [] for i in range(10)}
        for region in self.regions.values():
            self.depth[region.depth].append(region.name)
        pass




class Grid:
    def __init__(self, data=None):
        """create a new grid with data 

        Args:
            data (Data, optional): data object for the grid. Defaults to None.
        """        
        if data != None:
            self.data = data

    def visualize(self):
        """ visualize the grid using networkx visualization
        """        
        G = nx.DiGraph()
        positions = {}
        for hub in self.registry.hubs.values():
            if hub.region.depth == 1:
                color = 'green'
                size = 100
            elif hub.region.depth == 2:
                color = 'red'
                size = 50

            else:
                color = 'blue'
                size = 30

            G.add_node(hub.name, pos=(hub.x, hub.y), color=color)
            positions[hub.name] = (hub.x, hub.y)
        edges = [arc for arc in self.registry.arcs.keys()]

        G.add_edges_from(edges)

        node_colors = [G.nodes[data]['color'] for data in G.nodes()]

        nx.draw(G, positions, with_labels=False, node_size=50, node_color=node_colors)
        plt.show()

    # Creation methods for region, hub, and arc.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # All classes should refer to these methods when creating instances so that everything
    # is centralized. The methods will have return values so they can also be accessed during creation
    # within their class. In some cases, the natural procedure should be to initiate the creation within
    # another instance of the class so that the return value can be taken advantage of.
    
    def create_region(self, name, parent=None, data=None):
        """ create a new region with name, parent, data

        Args:
            name (str): region name
            parent (Region, optional): parent region. Defaults to None.
            data (DataFrame, optional): region data. Defaults to None.

        Returns:
            Region: newly created region
        """        
        if parent == None:
            return self.registry.add_region(Region(name, parent=parent, grid=self, data=data))
        else:
            parent.add_subregion(
                (self.registry.add(Region(name, parent=parent, grid=self, data=data)))
            )

    def create_arc(self, origin, destination, capacity, cost=0.0):
        """ create a new arc 

        Args:
            origin (Hub): origin hub
            destination (Hub): destination hub
            capacity (float): arc capacity
            cost (float), optional): cost of transporting a unit of product. Defaults to 0.0
        """        
        self.registry.add(TransportationArc(origin, destination, capacity, cost))
        origin.add_outbound(self.registry.arcs[(origin.name, destination.name)])
        destination.add_inbound(self.registry.arcs[(origin.name, destination.name)])

    def create_hub(self, name, region, data=None):
        """ create a new hub with name in region and with data

        Args:
            name (str): hub name
            region (Region): region located in
            data (DataFrame, optional): hub data. Defaults to None.
        """        
        region.add_hub(self.registry.add(Hub(name, region, data)))

    
    #delete function (works on arcs, hubs, and regions)
    def delete(self, thing):
        """ delete all references to thing from entire grid

        Args:
            thing (Region, Hub, Arc): thing to delete from grid
        """        
        if type(thing) == Region:
            thing.delete()
            self.registry.remove(thing)

        if type(thing) == Hub:
            for arc in list(thing.outbound.values()):
                self.delete(arc)
            for arc in list(thing.inbound.values()):
                self.delete(arc)

            thing.region.remove_hub(thing)
            self.registry.remove(thing)

        if type(thing) == TransportationArc:
            thing.disconnect()
            self.registry.remove(thing)

    def build_grid(self):
        """ build grid from Data object
        """        
        self.registry = Registry()
        self.world = Region('world', grid=self, data=self.data.regions)
        self.recursive_region_generation(self.data.regions, self.world)
        self.load_hubs()
        self.arc_generation(self.data.arcs)
        self.visualize()

    def recursive_region_generation(self, df, parent):
        """ generate hierarchical region list from dataframe recursively

        Args:
            df (DataFrame): region dataframe in particular format (see readme)
            parent (Region): parent region called under
        """        
        if df.columns[0] == 'data':
            for index, row in df.iterrows():
                # print(row[1:])
                parent.update_data(row[1:])
        else:
            for region in df.iloc[:, 0].unique():
                if type(region) == str and region != 'None':
                    # print(df.columns[0]+':',region)
                    parent.create_subregion(region)
                    self.recursive_region_generation(
                        df[df[df.columns[0]] == region][df.columns[1:]], parent.children[region]
                    )
                elif region == 'None':
                    self.recursive_region_generation(
                        df[df[df.columns[0]].isna()][df.columns[1:]], parent
                    )

                else:
                    self.recursive_region_generation(
                        df[df[df.columns[0]].isna()][df.columns[1:]], parent
                    )

    def arc_generation(self, df):
        """ generate arcs from arc data df

        Args:
            df (DataFrame): dataframe of arcs
        """        
        for index, row in df.iterrows():
            self.create_arc(
                self.registry.hubs[row.origin], self.registry.hubs[row.destination], row['capacity']
            )

    def connect_subregions(self):
        """ connect all hubs in the last level of the regional hierarchy to the hub in their parent region
        """        
        for hub in self.registry.hubs.values():
            if hub.region.children == {}:
                for parent_hub in hub.region.parent.hubs.values():
                    self.create_arc(hub, parent_hub, 10000000)

    def load_hubs(self):
        """ create hubs from data
        """        
        for index, row in self.data.hubs.iterrows():
            self.create_hub(
                row['hub'],
                grid.registry.regions[row['region']],
                data=pd.DataFrame(row[2:]).transpose().reset_index(),
            )

    def aggregate_hubs(self, hublist, region):
        """ combine hubs in hublist to a single hub and place in region

        Args:
            hublist (iterable): iterable of hubs to aggregate
            region (Region): region to place them in
        """        
        temp_hub_data = pd.concat([hub.data for hub in hublist])
        new_data = pd.DataFrame(columns=self.data.summable['hub'] + self.data.meanable['hub'])

        for column in temp_hub_data.columns:
            if column in self.data.summable['hub']:
                new_data[column] = [temp_hub_data[column].sum()]
            if column in self.data.meanable['hub']:
                new_data[column] = [temp_hub_data[column].mean()]

        name = '_'.join([hub.name for hub in hublist])
        self.create_hub(name, region, new_data)

        inbound = {}
        outbound = {}

        for hub in hublist:
            for arc in hub.inbound.values():
                if arc.origin not in hublist:
                    if arc.origin.name not in inbound.keys():
                        inbound[arc.origin.name] = [arc]
                    else:
                        inbound[arc.origin.name].append(arc)
            for arc in hub.outbound.values():
                if arc.destination not in hublist:
                    if arc.destination.name not in outbound.keys():
                        outbound[arc.destination.name] = [arc]
                    else:
                        outbound[arc.destination.name].append(arc)

        for origin in list(inbound.keys()):
            self.combine_arcs(inbound[origin], self.registry.hubs[origin], self.registry.hubs[name])
        for destination in list(outbound.keys()):
            self.combine_arcs(
                outbound[destination], self.registry.hubs[name], self.registry.hubs[destination]
            )

        del inbound
        del outbound

        for hub in hublist:
            self.delete(hub)

        del hublist

    def combine_arcs(self, arclist, origin, destination):
        """ combine all arcs in arclist to a single arc from origin to destination and aggregate their data

        Args:
            arclist (list): list of arcs
            origin (Hub): origin hub
            destination (Hub): destination hub
        """        
        capacity = sum([arc.capacity for arc in arclist])
        cost = sum([arc.cost * arc.capacity for arc in arclist]) / capacity
        self.create_arc(origin, destination, capacity, cost)
        for arc in arclist:
            self.delete(arc)

    def write_data(self):
        """write data to csv
        """        
        hublist = [hub for hub in list(self.registry.hubs.values())]
        hubdata = pd.concat(
            [
                pd.DataFrame({'hub': [hub.name for hub in hublist]}),
                pd.concat([hub.data for hub in hublist]).reset_index(),
            ],
            axis=1,
        )
        hubdata.to_csv('saveddata.csv', index=False)

        regionlist = [
            region for region in list(self.registry.regions.values()) if not region.data is None
        ]
        regiondata = pd.concat(
            [
                pd.DataFrame({'region': [region.name for region in regionlist]}),
                pd.concat([region.data for region in regionlist], axis=1).transpose().reset_index(),
            ],
            axis=1,
        )
        regiondata = regiondata[
            ['region'] + self.data.summable['region'] + self.data.meanable['region']
        ]
        regiondata.to_csv('regiondatasave.csv', index=False)

        arclist = [arc for arc in list(self.registry.arcs.values())]
        arcdata = pd.DataFrame(
            {
                'origin': [arc.origin.name for arc in arclist],
                'destination': [arc.destination.name for arc in arclist],
                'capacity': [arc.capacity for arc in arclist],
                'cost': [arc.cost for arc in arclist],
            }
        )
        arcdata.to_csv('arcdatasave.csv', index=False)

    def collapse(self, region_name):
        """ absorb all subregions of region with region_name, and their subregions, recursively, into that region

        Args:
            region_name (str): name of region to collapse
        """        
        self.registry.regions[region_name].absorb_subregions_deep()
        self.aggregate_hubs(
            list(self.registry.regions[region_name].hubs.values()),
            self.registry.regions[region_name],
        )
        self.registry.update_levels()
        self.visualize()

    def build_model(self, mode='standard'):
        """ build a pyomo model from current region data

        Args:
            mode (str, optional): mode to run in. Defaults to 'standard'.
        """        
        self.model = Model(self, mode)

    def test(self):
        """run a test that builds a model from data and then solves
        """        
        start = time.time()
        self.build_model()
        self.model.start_build()
        self.model.solve(self.model.m)
        end = time.time()
        print(end - start)

    def collapse_level(self, level):
        """collapse all regions at level in the regional hierarchy
        Args:
            level (int): the level of the regional hierarchy
        """        
        for region in self.registry.depth[level]:
            self.collapse(region)


class Data:
    def __init__(self):
        """ Initialize Data object by reading from csv's
        """        
        self.regions = pd.read_csv('input/regions.csv', index_col=False)
        self.hubs = pd.read_csv('input/hubs.csv', index_col=False)
        self.arcs = pd.read_csv('input/transportation_arcs.csv', index_col=False)

        params = pd.ExcelFile('input/parameter_list.xlsx')

        self.hub_params = pd.read_excel(params, 'hub', index_col=False)
        self.region_params = pd.read_excel(params, 'region', index_col=False)
        self.arc_params = pd.read_excel(params, 'arc', index_col=False)
        self.global_params = pd.read_excel(params, 'global', index_col=False)
        self.technologies = [
            column_name.split('_')[2]
            for column_name in self.hubs.columns
            if column_name.lower().startswith('production_capacity')
        ]
        

        self.summable = {
            'hub': self.hub_params[self.hub_params['aggregation_type'] == 'sum'][
                'parameter'
            ].tolist(),
            'region': self.region_params[self.region_params['aggregation_type'] == 'sum'][
                'parameter'
            ].tolist(),
            'arc': self.arc_params[self.arc_params['aggregation_type'] == 'sum'][
                'parameter'
            ].tolist(),
        }
        self.meanable = {
            'hub': self.hub_params[self.hub_params['aggregation_type'] == 'mean'][
                'parameter'
            ].tolist(),
            'region': self.region_params[self.region_params['aggregation_type'] == 'mean'][
                'parameter'
            ].tolist(),
            'arc': self.arc_params[self.arc_params['aggregation_type'] == 'mean'][
                'parameter'
            ].tolist(),
        }
        self.fixed_production_cost = {}
        for technology in self.technologies:
            self.fixed_production_cost[technology] = self.global_params.loc[
                self.global_params.parameter == 'fixed_cost_' + technology
            ].reset_index()['default_value'][0]

    def updata_data():
        pass


class Functions:
    def __init__(self, model, mode='standard', start_year=2022, end_year=2070):
        """intialize class that stores functions to build model from.

        Args:
            model (Model): model builder
            mode (str, optional): mode to run in. Defaults to 'standard'.
            start_year (int, optional):start year. Defaults to 2022.
            end_year (int, optional): end  year. Defaults to 2070.
        """        
        self.grid = model.grid
        self.registry = model.registry
        self.data = model.data
        self.mode = mode
        self.start_year = start_year
        self.end_year = end_year

        if self.mode == 'standard':
            self.time = 'annual'

    def get_capacity(self, hub, tech):
        """ get capacity for hub

        Args:
            hub (str): hub
            tech (str): technology type

        Returns:
            float: capacity
        """        
        if self.mode == 'standard':
            return self.registry.hubs[hub].get_data('production_capacity_' + tech)

    def get_production_cost(self, hub, tech, year):
        """ get production cost

        Args:
            hub (str): hub
            tech (str): technology type
            year (int): year

        Returns:
            float: production cost for unit of product
        """        
        if self.mode == 'standard':
            if tech == 'PEM':
                return self.registry.hubs[hub].region.data['electricity_cost'] * 45
            elif tech == 'SMR':
                return self.registry.hubs[hub].region.data['gas_cost']
            else:
                return 0

        pass

    def get_elec_price(self, hub, year):
        """ get electricity price

        Args:
            hub (str): hub
            year (int): year

        Returns:
            float: electricity price
        """        
        if self.mode == 'standard':
            return self.registry.hubs[hub].region.data['electricity_cost'] * 45

    def get_demand(self, region, time):
        """ get the demand for region and time

        Args:
            region (str): region
            time (int): time / year

        Returns:
            float: demand for region and time
        """        
        if self.mode == 'standard':
            return self.registry.regions[region].get_data('demand') * 1.10 ** (
                time - self.start_year
            )

        elif self.mode == 'integrated':
            pass

        return 0

    def objective(self):
        pass

    def constraints(self):
        pass


class Model:
    def __init__(self, grid, mode):
        """ initialize a model maker

        Args:
            grid (Grid): grid object the model is based on
            mode (str): mode to run in
        """        
        self.grid = grid
        self.registry = grid.registry
        self.hubs = grid.registry.hubs
        self.regions = grid.registry.regions
        self.arcs = grid.registry.arcs
        self.data = grid.data
        self.functions = Functions(self, mode)

    def start_build(self):
        """ build model
        """        
        self.m = ConcreteModel()
        self.generate_sets(self.m)
        self.generate_parameters(self.m)
        self.generate_variables(self.m)
        self.generate_constraints(self.m)
        self.m.production_cost = Objective(rule=self.total_cost(self.m))

    def generate_sets(self, m):
        """generate the sets 

        Args:
            m (ConcreteModel): model instance to populate sets for
        """        
        m.hubs = Set(initialize=self.hubs.keys())
        m.arcs = Set(initialize=self.arcs.keys())
        m.regions = Set(initialize=self.regions.keys())
        m.technology = Set(initialize=self.data.technologies)
        m.year = RangeSet(self.functions.start_year, self.functions.end_year)

    def generate_parameters(self, m):
        """ generate the parameters for a model

        Args:
            m (ConcreteModel): model instance to populate parameters for
        """        
        m.cost = Param(
            m.hubs,
            m.technology,
            m.year,
            initialize={
                (hub, tech, year): self.hubs[hub].cost(tech, year)
                for hub in m.hubs
                for tech in m.technology
                for year in m.year
            },
        )
        m.capacity = Param(
            m.hubs,
            m.technology,
            initialize={
                (hub, tech): self.functions.get_capacity(hub, tech)
                for hub in m.hubs
                for tech in m.technology
            },
        )
        m.demand = Param(
            m.regions,
            m.year,
            mutable=True,
            initialize={(region, year): self.function.get_demand(region, year)},
        )
        m.electricty_price = Param(
            m.hub,
            m.year,
            mutable=True,
            initialize={(hub, year): self.function.get_elec_price(region, year)},
        )

    def generate_variables(self, m):
        """ generate variables for model m

        Args:
            m (ConcreteModel): model to generate variables for
        """        
        m.h2_volume = Var(m.hubs, m.technology, m.year, within=NonNegativeReals)
        m.transportation_volume = Var(m.arcs, m.year, within=NonNegativeReals)
        m.capacity_expansion = Var(
            m.hubs, m.technology, m.year, within=NonNegativeReals, initialize=0.0
        )
        m.trans_capacity_expansion = Var(m.arcs, m.year, within=NonNegativeReals, initialize=0.0)

    def generate_constraints(self, m):
        """ generate the constraints for model m

        Args:
            m (ConcreteModel): model to generate constraints for
        """        
        m.capacity_constraint = Constraint(
            m.hubs, m.technology, m.year, rule=self.capacity_constraint
        )
        m.transportation_constraint = Constraint(
            m.arcs, m.year, rule=self.transportation_capacity_constraint
        )
        m.demand_constraint = Constraint(m.regions, m.year, rule=self.demand_constraint)

    def mass_balance(self, m, hub, year):
        """ the mass balance for a hub in a given year

        Args:
            m (ConcreteModel): model constraint is for
            hub (str): hub
            year (int): year

        Returns:
            float: the mass balance
        """        
        return (
            sum(m.h2_volume[hub, tech, year] for tech in m.technology)
            + sum(
                m.transportation_volume[arc.name, year] for arc in self.hubs[hub].inbound.values()
            )
            - sum(
                m.transportation_volume[arc.name, year] for arc in self.hubs[hub].outbound.values()
            )
        )

    def transportation_capacity_constraint(self, m, origin, destination, year):
        """ transportation capacity constraint

        Args:
            m (ConcreteModel): model 
            origin (str): origin hub
            destination (str): destination hub
            year (int): year 

        Returns:
            boolean: tranportation volume <= capacity + capacity expansion
        """        
        return m.transportation_volume[(origin, destination), year] <= self.arcs[
            (origin, destination)
        ].capacity + sum(
            m.trans_capacity_expansion[(origin, destination), year]
            for year in range(m.year[1], year)
        )
        

    def capacity_constraint(self, m, hub, tech, year):
        """ production capacity constraint

        Args:
            m (ConcreteModel): model
            hub (str): str
            tech (str): str
            year (int): str

        Returns:
            boolean: production <= capacity + capacity expansion
        """        
        return m.h2_volume[hub, tech, year] <= self.functions.get_capacity(hub, tech) + sum(
            m.capacity_expansion[hub, tech, year] for year in range(max(m.year[1], year - 20), year)
        )

    def demand_constraint(self, m, region, year):
        """ demand constraint

        Args:
            m (ConcreteModel): model
            region (str): region
            year (int): year

        Returns:
           boolean: mass balance = demand for all years and regions
        """        
        if len(self.regions[region].hubs) == 0:
            return Constraint.Feasible
        else:
            return sum(
                self.mass_balance(m, hub, year) for hub in self.regions[region].hubs.keys()
            ) == self.functions.get_demand(region, year)

    def production_cost(self, m):
        """ calculate the produciton cost for model m

        Args:
            m (ConcreteModel): model

        Returns:
           float: cost
        """        
        return sum(
            m.h2_volume[hub, tech, year] * m.cost[hub, tech, year]
            for hub in m.hubs
            for tech in m.technology
            for year in m.year
        )

    def transportation_cost(self, m):
        """ transportation cost for model m

        Args:
            m (ConcreteModel): mdoel

        Returns:
            float: transportation cost
        """        
        return sum(m.transportation_volume[arc, year] * 0.12 for arc in m.arcs for year in m.year)

    def prod_capacity_expansion_cost(self, m):
        """ total production capacity expansion cost for model m

        Args:
            m (ConcreteModel): model

        Returns:
            float: total cost for whole model instance
        """        
        return sum(
            m.capacity_expansion[hub, tech, year] * 10
            for hub in m.hubs
            for tech in m.technology
            for year in m.year
        )

    def trans_capacity_expansion_cost(self, m):
        """ total transportation capacity expansion cost for model m

        Args:
            m (CocreteModel): model

        Returns:
            float: total transportation capacity expansion cost for whole model instance
        """        
        return sum(m.trans_capacity_expansion[arc, year] * 3 for arc in m.arcs for year in m.year)

    def capacity_expansion_cost(self, m):
        """ total capacity expansion cost for all types

        Args:
            m (ConcreteModel): model

        Returns:
           float: total cost of capacity expansion
        """        
        return self.prod_capacity_expansion_cost(m) + self.trans_capacity_expansion_cost(m)

    def solve(self, m):
        """ solve model

        Args:
            m (ConcreteModel): model
        """        
        solver = SolverFactory('appsi_highs')
        solver.solve(m, tee=True).write()

    def total_cost(self, m):
        """ total cost for everything in the model

        Args:
            m (ConcreteModel): model

        Returns:
            float: total cost
        """        
        return (
            self.production_cost(m) + self.transportation_cost(m) + self.capacity_expansion_cost(m)
        )


def visualize_regions(data):
    """ create a hierarchical treemap of all regions with their produciton values for a given run

    Args:
        data (DataFrame): dataframe of regions
    """    
    data.fillna('None', inplace=True)
    data['all'] = 'world'

    fig = px.treemap(
        data_frame=data,
        path=['all'] + [name for name in data.columns if name.lower().endswith('region')],
        values='H2Capacity',
    )
    fig.write_html('regions.html', auto_open=True)


grid = Grid(Data())
grid.build_grid()

