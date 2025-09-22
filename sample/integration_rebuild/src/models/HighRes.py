from graniteledger.src.common.Model import Model
from graniteledger.src.common.config_setup import Config_settings

import random
from pathlib import Path
from collections import defaultdict
from typing import List, Literal
from itertools import product
import pyomo.environ as pyo
from graniteledger.src.common.config_setup import Config_settings

class HighRes(Model):
    def __init__(
        self
    ):
        # Initialize the Model
        Model.__init__(self)

    @classmethod
    def build(
        cls,
        settings: Config_settings,
        *args,
        components: defaultdict | None = None,
        **kwargs
    ):
        """Build method for HighRes cost minimization problem

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
        HighRes
            An instance of the HighRes class
        """
        # Construct Data
        if not components:
            components = cls.preprocess(settings)  

        # Initialize the object
        model = cls()
        model.build_sets(components)
        model.build_params(components)
        model.build_vars(components)
        model.build_expressions(components)

        # Placeholder for options push
        # if recipe:
        #     for option in recipe:
        #         setattr(model, option, recipe[option])

        return model

    def build_sets(
        self,
        components: defaultdict
    ):
        # Base Sets (e.g. singletons)
        for index in components["indices"]:
            self.declare_set(
                sname = index,
                sdata = components["indices"][index],
                scols = [index]
            )
        
        # Sets for specific parameters and variables
        self.declare_set_with_sets(
            "demand_index",
            self.hour, self.day, self.season, self.region, self.year
        )

        # Declare cost parameter
        self.declare_set_with_sets(
            "cost_index",
            self.hour, self.day, self.season, self.tech, self.region, self.year
        )

        self.declare_set_with_sets(
            'generation_index',
            self.cost_index
        )

        self.reorganize_index_set(
            sname = 'generation_index', 
            new_sname = 'demand_sum_index',
            create_indexed_set = True,
            reorg_set_sname='demand_index'
        )

    def build_params(
        self,
        components: defaultdict
    ):
        @self.ParameterExpression(self.demand_index, mutable = True)
        def demand(self, h, d, s, r, y):
            return components["demand"][(h,d,s,r,y)]

        @self.ParameterExpression(self.cost_index, mutable = True)
        def cost(self, h, d, s, t, r, y):
            return components["cost"][(h,d,s,t,r,y)]
    
    def build_vars(
        self,
        components: defaultdict
    ):
        self.declare_var(
            'generation',
            self.generation_index
        )

    def build_expressions(
        self,
        components: defaultdict
    ):
        @self.ConstraintExpression(self.demand_sum_index)
        def market_clearing(self, h, d, s, r, y):
            return self.demand[h,d,s,r,y] <= sum(self.generation[h,d,s,tech,r,y] for tech in self.demand_sum_index[h,d,s,r,y])

        @self.Objective(sense = pyo.minimize)
        def objective(self):
            return sum(self.generation[element] * self.cost[element] for element in self.generation_index)

    @classmethod
    def preprocess(
        cls,
        settings: Config_settings
    ) -> defaultdict:

        # preprocess -- Create basic index vectors
        components = defaultdict()
        components["indices"] = cls._generate_index(settings)

        # preprocess -- Assign index names to each component
        components["index_names"] = defaultdict()
        for index_name in components["indices"]:
            components["index_names"][index_name] = [index_name]

        # preprocess -- Set up new index sets
        components["index_names"]["demand_index"] = ["hour", "day", "season", "region", "year"]
        components["index_names"]["cost_index"] = ["hour", "day", "season", "tech", "region", "year"]
        components["index_names"]["capacity_index"] = ["hour", "day", "season", "tech", "region", "year"]

        # preprocess -- Generate data with settings inputs at tech, day, hr, season
        components["cost"] = cls._generate_values(
            0.5,
            10,
            components["indices"],
            *components["index_names"]["cost_index"]
        )

        components["demand"] = cls._generate_values(
            3000,
            10000,
            components["indices"],
            *components["index_names"]["demand_index"]
        )

        components["capacity"] = cls._generate_values(
            4000,
            10000,
            components["indices"],
            *components["index_names"]["capacity_index"]
        )
        return components

    @staticmethod
    def _generate_index(
        settings: Config_settings
    ):
        indices = defaultdict(set)

        # _generate_index -- setting up vectors (we could do this with an excel sheet or something)
        indices["hour"] = set(range(1,5)) # Four hours
        indices["season"] = set(range(1,5)) # Four seasons
        indices["day"] = set(range(1,3)) # Two representative days
        indices["tech"] = set(settings.techs)
        indices["region"] = set(settings.regions)
        indices["year"] = set(settings.years)
        return indices

    @staticmethod
    def _generate_values(
        min_value: int,
        max_value: int,
        indices: defaultdict[set],
        *index_names,
    ):
        # _generate_cost -- Create index set from inputs
        index_set = {key: indices[key] for key in index_names}
            
        # _generate_cost -- Creates cost vector indexed by indexes in set
        values = defaultdict()
        index_set = product(*index_set.values())
        for elem in index_set:
            values[elem] = random.uniform(min_value, max_value)
        return values
