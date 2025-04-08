from src.common.Model import Model
from src.common.config_setup import Config_settings

import pyomo.environ as pyo
import random
from collections import defaultdict
from typing import List, Literal

class LowRes(Model):
    def __init__(
        self
    ):
        # Initialize the Model
        Model.__init__(self)

    @classmethod
    def build(
        cls,
        settings: Config_settings,
        components: defaultdict | None = None,
        **kwargs
    ):
        """Build method for LowRes cost minimization problem

        Parameters
        ----------
        settings : Config_settings
            Settings object
        components : defaultdict | None, optional
            Components dictionary that contains basic data, by default None
        recipe: defaultdict | None, optional
            Dictionary containing information from config to build
        
        Returns
        -------
        LowRes
            An instance of the LowRes class
        """

        # build: declaring sets -- create data object
        if not components:
            components = cls.preprocess(settings)  
       
        # build: declaring sets -- create instance and build model
        model = cls()
        model.build_sets(components)
        model.build_params(components)
        model.build_variables(components)
        model.build_constraints(components)
        model.build_objective(components)

        # build: assign technology to model (placeholder for genuine options pass)
        if kwargs:
            for key in kwargs:
                setattr(model, key, kwargs[key])
        
        return model

    def build_sets(
        self,
        components: defaultdict
    ):
        # build: declaring sets-- set up base-sets
        for name in ["season", "region", "year"]:
            self.declare_set(
                sname = name,
                sdata = components["indices"][name],
                scols = [name]
            )

        for name in ["arcs", "hubs"]:
            set_members = list(components[name].keys())
            self.declare_set(
                sname = name,
                sdata = set_members,
                scols = [name]
            )

        # build: declaring sets -- declare regional hubs and main hubs sets
        def hub_index_init(hubs, type):
            hub_list = []
            for hub in hubs:
                if components["hubs"][hub]["type"] == type:
                    hub_list.append((hub, components["hubs"][hub]["region"]))
            return hub_list

        self.declare_set(
            'main_hubs',
            hub_index_init(self.hubs, "main"),
            ["hubs", "region"]
        )

        self.declare_set(
            'regional_hubs',
            hub_index_init(self.hubs, "regional"),
            ["hubs", "region"]
        )
           
        # build: declaring sets -- create full time set
        self.declare_ordered_time_set(
            'timestep',
            self.year, self.season
        )

        # build -- Create indexed sets for hubs for nodes-in and nodes-out
        def NodesOut_init(arcs):
            res = defaultdict(list)
            for element in arcs:
                res[element[0]].append(element[1])
            return res
        NodesOut_dict = NodesOut_init(self.arcs)

        self.declare_set(
            sname = 'NodesOut',
            sdata = NodesOut_dict,
            scols = ["Nodes"]
        )

        def NodesIn_init(arcs):
            res = defaultdict(list)
            for element in arcs:
                res[element[1]].append(element[0])
            return res
        NodesIn_dict = NodesIn_init(self.arcs)

        self.declare_set(
            sname = 'NodesIn',
            sdata = NodesIn_dict,
            scols = ["Nodes"]
        )

        # build -- Set up hub node sets
        for new_index in ["production_cost_index", "storage_cost_index", "storage_capacity_index", "production_capacity_index"]:
            self.declare_set_with_sets(
                new_index,
                self.hubs
            )

        # build -- set up transport arc sets
        for new_index in ["transport_cost_index", "transport_capacity_index"]:
            self.declare_set_with_sets(
                new_index,
                self.arcs
            )

        self.declare_set_with_sets(
            "production_index",
            self.hubs, self.timestep
        )

        # build: variables -- transport along arcs
        self.declare_set_with_sets(
            'transport_index',
            self.arcs, self.timestep
        )

        self.declare_set_with_sets(
            'storage_index',
            self.hubs, self.timestep
        )

        # build: declaring sets -- build time set for storage constraint over time
        self.declare_shifted_time_set(
            'storage_timestep',
            1,
            "lag",
            self.timestep
        )

    def build_params(
        self,
        components
    ):
        @self.ParameterExpression(self.production_cost_index, mutable = True)
        def production_cost(self, n):
            return components["hubs"][n]["production_cost"]
        
        @self.ParameterExpression(self.storage_cost_index, mutable = True)
        def storage_cost(self, n):
            return components["hubs"][n]["storage_cost"]

        @self.ParameterExpression(self.storage_capacity_index, mutable = True)
        def storage_capacity(self, n):
            return components["hubs"][n]["storage_capacity"]

        @self.ParameterExpression(self.production_capacity_index, mutable = True)
        def production_capacity(self, n):
            return components["hubs"][n]["production_capacity"]
        
        @self.ParameterExpression(self.main_hubs, mutable = True)
        def demand(self, n, r):
            return components["hubs"][n]["demand"]

        @self.ParameterExpression(self.transport_cost_index, mutable = True)
        def transport_cost(self, i, j):
            return components["arcs"][(i,j)]["transport_cost"]

        @self.ParameterExpression(self.transport_capacity_index, mutable = True)
        def transport_capacity(self, i, j):
            return components["arcs"][(i,j)]["transport_capacity"]

    def build_variables(
        self,
        components
    ):
        # build: variables -- production at hubs
        self.declare_var("production", self.production_index, return_var = False)
        self.declare_var("transport", self.transport_index, return_var = False)

        # build: variables -- Flow into and out of storage
        self.declare_var('flow_into_storage', self.storage_index, return_var = False)
        self.declare_var('flow_out_storage', self.storage_index, return_var = False)
        self.declare_var('storage', self.storage_index, return_var = False)

    def build_constraints(
        self,
        components: defaultdict
    ):
        # build -- set up constraints
        @self.Expression(self.main_hubs, self.timestep)
        def flow_into_hub(self, hub, region, timestep): 
            return sum(self.transport[in_hub, hub, timestep] for in_hub in self.NodesIn[hub])

        @self.Expression(self.hubs, self.timestep)
        def flow_out_of_hub(self, hub, timestep):
            return sum(self.transport[hub,out_hub, timestep] for out_hub in self.NodesOut[hub])

        # build: constraints -- Flow balance at main hubs
        @self.ConstraintExpression(self.main_hubs, self.timestep)
        def market_clearing(self, hub, region, timestep):
            return self.production[hub, timestep] + self.flow_into_hub[hub, region, timestep] + self.flow_out_storage[hub, timestep] == \
                   self.demand[hub, region] + self.flow_out_of_hub[hub, timestep] + self.flow_into_storage[hub, timestep]

        # build: flow balance at regional hubs
        @self.ConstraintExpression(self.regional_hubs, self.timestep)
        def transport_balance(self, hub, region, timestep):
            return self.production[hub, timestep] + self.flow_out_storage[hub, timestep] == \
                    self.flow_out_of_hub[hub, timestep] + self.flow_into_storage[hub, timestep]

        # build: storage cumulative constraint
        @self.ConstraintExpression(self.hubs, self.storage_timestep)
        def storage_link(self, hub, timestep):
           return self.storage[hub, timestep] == self.storage[hub, timestep-1] + \
                (self.flow_into_storage[hub, timestep] - self.flow_out_storage[hub, timestep])

        # build: storage constraint (can't store beyond capacity)
        @self.ConstraintExpression(self.hubs, self.timestep)
        def storage_capacity_limit(self, hub, timestep):
            return self.storage[hub, timestep] <= self.storage_capacity[hub]

        # build: production constraint (can't produce beyond capacity)
        @self.ConstraintExpression(self.hubs, self.timestep)
        def production_capacity_limit(self, hub, timestep):
            return self.production[hub, timestep] <= self.production_capacity[hub]
        
        # build: transport constraint
        @self.ConstraintExpression(self.arcs, self.timestep)
        def transport_capacity_limit(self, in_hub, out_hub, timestep):
            return self.transport[in_hub, out_hub, timestep] <= self.transport_capacity[in_hub, out_hub]
    
    def build_objective(
        self, 
        components: defaultdict
    ):

        @self.Expression()
        def total_production_cost(self):
            return sum(self.production_cost[hub]*self.production[hub,timestep] for hub,timestep in self.production_index)

        @self.Expression()
        def total_transport_cost(self):
            return sum(self.transport_cost[in_hub, out_hub] * self.transport[in_hub, out_hub, timestep] for \
                in_hub, out_hub, timestep in self.transport_index)

        @self.Expression()
        def total_storage_cost(self):
            return sum(self.storage_cost[hub] * self.storage[hub, timestep] + \
                self.storage_cost[hub] * self.storage[hub, timestep] for hub, timestep in self.storage_index)

        @self.Objective(sense = pyo.minimize)
        def objective(self):
            return self.total_production_cost + \
                self.total_transport_cost + \
                self.total_storage_cost


    @classmethod
    def preprocess(
        cls,
        settings: Config_settings,
    ):
        components = defaultdict()
        # Build a transport problem with storage
        components["indices"] = cls._generate_index(settings)

        components["hubs"] = cls._generate_hubs(
            components["indices"]
        )
        
        components["arcs"] = cls._generate_arcs(
            components["indices"], 
            components["hubs"]
        )
        return components

    @classmethod
    def _generate_index(
        cls,
        settings: Config_settings,
    ):
        indices = defaultdict()

        # _generate_index -- setting up vectors (we could do this with an excel sheet or something)
        indices["season"] = set(range(1,5)) # Four seasons
        indices["region"] = set(settings.regions)
        indices["year"] = set(settings.years)
        return indices

    @classmethod
    def _generate_hubs(
        cls,
        indices: defaultdict,
        demand_bound: tuple | None = (100000, 5000000),
        capacity_bound: tuple | None = (1000000, 8000000),
        storage_bound: tuple | None = (100000, 3000000),
        cost_bound: tuple | None = (3, 10),
        storage_cost_bound: tuple | None = (0.01, 0.75)
    ):
        hubs = defaultdict()
        # _generate_hubs -- Main hubs in each region

        hub_index = 0
        for region in indices["region"]:
            # random generation of regional hubs with production
            n_region_hubs = random.randint(2,6)
            for hub in range(0, n_region_hubs):
                hubs[hub_index] = defaultdict()
                hubs[hub_index]["region"] = region
                if hub == 0:
                    hubs[hub_index]["type"] = "main"
                else:
                    hubs[hub_index]["type"] = "regional"
                hub_index += 1

        # _generate_hubs_ -- Provide demand at each hub (main hub)
        for hub in hubs.keys():
            if hubs[hub]["type"] == "main":
                hubs[hub]["demand"] = random.randint(demand_bound[0], demand_bound[1])
            else:
                hubs[hub]["demand"] = 0
        
        # _generate_hubs -- Set production capacity and costs in each node
        for hub in hubs.keys():
            hubs[hub]["production_capacity"] = random.randint(capacity_bound[0], capacity_bound[1])
            hubs[hub]["storage_capacity"] = random.randint(storage_bound[0], storage_bound[1])
            hubs[hub]["production_cost"] = random.uniform(cost_bound[0], cost_bound[1])
            hubs[hub]["storage_cost"] = random.uniform(storage_cost_bound[0], storage_cost_bound[1])

        # _generate_hubs -- Set up indices
        indices["hubs"] = defaultdict()
        indices["hubs"]["main"] = []
        indices["hubs"]["regional"] = []
        for hub in hubs.keys():
            if hubs[hub]["type"] == "main":
                indices["hubs"]["main"].append(hub)
            else:
                indices["hubs"]["regional"].append(hub)
        return hubs
    
    @classmethod
    def _generate_arcs(
        cls,
        indices: defaultdict,
        hubs: defaultdict,
        cost_bound: tuple[int, int] | None = (0.1, 1),
        capacity_bound: tuple[int, int] | None = (1000000, 2000000)
    ):
        arcs = defaultdict()

        # _generate_arcs -- connect main grid
        main_hubs = indices["hubs"]["main"]
        for index in range(1, len(main_hubs)):
            start = main_hubs[index]
            end = random.choice(main_hubs[0:index])

            # define details of arc
            for element in [(start,end), (end,start)]:
                arcs[element] = defaultdict()
                arcs[element]["transport_cost"] = random.uniform(cost_bound[0], cost_bound[1])
                arcs[element]["transport_capacity"] = random.randint(capacity_bound[0], capacity_bound[1])
            
        # _generate_arcs -- Set up non-main arcs, stratified by region
        for region in indices["region"]:
            region_hubs = []
            main_hub = None
            for hub in indices["hubs"]["regional"]:
                if hubs[hub]["region"] == region:
                    region_hubs.append(hub)
            
            for hub in indices["hubs"]["main"]:
                if hubs[hub]["region"] == region:
                    main_hub = hub

            # connect regional hubs to main (and potentially to each other)
            for hub in region_hubs:
                start = hub
                end = main_hub

                element = (start, end)
                arcs[element] = defaultdict()
                arcs[element]["transport_cost"] = random.uniform(cost_bound[0], cost_bound[1])
                arcs[element]["transport_capacity"] = random.randint(capacity_bound[0], capacity_bound[1])
        
        # generate_arcs -- update indices
        indices["arcs"] = [x for x in arcs.keys()]
        return arcs

            
                

                

            