"""This file contains the Build class which handles constructing models based on the instructions 
described by an instance of Config_settings.

Build instantiates all the modules required by the solve_algorithm in an instance of settings and handles 
the execution of the solve.

Currently, this is under development; main.py only directs Build to construct the model. Execution method still 
in progress
"""

# Import packages
import pyomo.environ as pyo
from typing import List, Dict
from pathlib import Path
import pandas as pd
import warnings

from src.common.config_setup import Config_settings
import src.interchange.improvedcrosswalk as IC
from src.common.tolerance import Tolerance
import src.models as Models

class Build():
    def __init__(
        self,
        settings: Config_settings,
        save: bool
    ):
        self.settings = settings
        self.save = save

    @classmethod
    def build(
        cls,
        settings: Config_settings,
        save: bool | None = False,
    ):
        """Build classmethod that creates a pyomo ConcreteModel instance that contains each of the pyomo
        modules specified in settings, as well as instantiates an Interchange object (which handles parameter
        exchange) and a Tolerance object (which checks for convergence after solve cycles)

        Parameters
        ----------
        settings : Config_settings
            Settings for a build constructed from configuration file
        save : bool | None, optional
            Pickle the built model, by default False

        Returns
        -------
        obj
            A Build instance
        """
        # Build: Initialize the Build object
        build = cls(settings, save)

        # Build: Initialize Tolerance instance
        build.instantiate_tolerance()

        # Build: Set up Dimension and Interchange objects
        build.instantiate_interchange()

        # Build: Using settings and the modules dictionary, create and build meta
        build.instantiate_model()

        # Build: Construct any necessary expressions to link variables (if unified)
        if settings.solve_method == "unified":
            build.generate_expressions(settings)
        
        if save:
            pass # pickle

        return build

    def execute(
        self
    ):
        """Executes the solve_algorithm contained in settings for the instance of 
        build.

        Currently under development; specifically, the 'exchange' method is under development
        """

        # # execute: using the solve_algorithm in settings, execute the solve
        # steps = [step for step in self.settings.solve_algorithm]

        # # execute: Find all components that need to be exchanged
        # #   - Check multiindex (I need to know which dimensions are time indices) OR check if the index has a _time_to_index set
        # #       - if time_to_index set, this is already a multiindex 
        # #   - If multi-time index, create (OR CHECK FOR EXISTENCE) a timestep w/ declare_ordered_time_set
        # #       - update_crosswalk((season,day):value, "season-day")
        # #   - Update/Reconstruct/Develop new crosswalk
        # #       - Populate a new column w/ hyphenated name of multi-index (with derived values)
        # #       - Take base column and new column; add unit name OR add the full new dataframe
        # # Output --> dictionary of the from/to components, from_multi-index

        # # execute: printing steps in each solve cycle
        # done = False
        # while not done:
        #     for step in steps:

        #         # execute: steps -- Get instructions for step
        #         ingredient = self.settings.solve_algorithm[step]

        #         # execute: steps -- get action
        #         action = getattr(self, ingredient["action"])

        #         kwargs = ingredient["options"]
        #         action(**kwargs)
        #     done = self.tolerance.check()
        pass

    def instantiate_tolerance(
        self
    ):
        """Create an instance of Tolerance class and assign to the build instance
        """
        self.tolerance = Tolerance(self.settings)

    def instantiate_interchange(
        self
    ):
        """Create an instance of Interchange and set up by creating UnitDimension objects for each dimension of
        crosswalk specified in the configuration file.
        """
        # Create an empty Interchange object    
        self.interchange = IC.Interchange()

        # for each of the dimensions listed in settings, add dimension to Interchange
        dimensions = self.settings.dimensions.__dict__
        for name in dimensions:
            dimension_options = getattr(self.settings.dimensions, name)
            unitdimension = self.instantiate_dimension(
                getattr(dimension_options, "path"),
                getattr(dimension_options, "dim_name"),
                getattr(dimension_options, "base_unit")
            )
            self.interchange.add_dimension(unitdimension)

    def instantiate_model(
        self,
        initial_solve: bool | None = True
    ):
        """Initialize each of the modules implied by settings.modules to execute the solve_algorithm.

        Solve each module on its own if initial_solve is True. If set False, certain first-cycle exchanges 
        may fail (e.g. variables with no values stored until a solve is conducted).

        Parameters
        ----------
        initial_solve : bool | None, optional
            Conduct initial solve on modules upon instantiation, by default True
        """
        # intantiate_model: Create an empty 'meta'
        self.meta = pyo.ConcreteModel()
        
        for module_name in self.settings.modules:
            module = self.settings.modules[module_name]
            class_name = module["class"]
            model_name = module["name"]

            # unpack any other keyword arguments
            kwargs = {}
            keys = [k for k in module if k not in ["class", "name"]]
            for key in keys:
                kwargs[key] = module[key]

            # build string to execute
            callable_class = getattr(Models, class_name)
            instance = callable_class.build(self.settings, **kwargs)

            # load the model to interchange
            self.interchange.add_model(instance)
            self.tolerance.add_model(model_name)

            # initial solve
            setattr(self.meta, model_name, instance)
            if initial_solve:
                self.solve(model_name)

    def instantiate_dimension(
        self,
        path: Path,
        dim_name: str,
        base_unit: str | None = None
    ):
        """Create a Dimension for the Interchange object

        Parameters
        ----------
        path : Path
            Path to the crosswalk file describing the Dimension
        dim_name : str
            Name of the Dimension
        base_unit : str | None, optional
            Base unit for the Dimension (e.g. the building block unit all others can be reduced to), by default None

        Returns
        -------
        obj
            A UnitDimension object
        """
        # load crosswalk based on Path
        crosswalk = pd.read_csv(path)

        unit_names = crosswalk.columns.to_list()
        if not base_unit or base_unit not in unit_names:
            warnings.warn(f"No base unit provided to instantiate_dimension for dimension {dim_name}: defaulting to row index")
            base_unit = "base"
            crosswalk.reset_index(inplace=True, names="base")
        
        # instantiate unit dimension
        return IC.UnitDimension(dim_name, crosswalk, base_unit)

    def generate_expressions(
        self,
        settings: Config_settings
    ):
        pass

    def exchange(
        self,
        from_module: str,
        from_component: str,
        to_module: str,
        to_component: str,
        **kwargs
    ):
        # # exchange: unpack str args into name objects
        # from_module_name = from_module
        # to_module_name = to_module
        # from_component_name = from_component
        # to_component_name = to_component

        # # exchange: Get hook to relevant modules in meta
        # from_module = getattr(self.meta, from_module_name)
        # to_module = getattr(self.meta, to_module_name)

        # # exchange: Check component for existence
        # from_component_exist = hasattr(from_module, from_component_name)
        # to_component_exist = hasattr(to_module, to_component_name)
        # if not (from_component_exist and to_component_exist):
        #     message = "Components provided for integration do not exist in class instances: "
        #     if not from_component_exist:
        #         message = message + f"from_component: {from_component_name}; "
        #     if not to_component_exist:
        #         message = message + f"to_component: {to_component_name}"
        #     raise ValueError(message)
        
        # # exchange: get the from values
        # from_values = self.create_exchange_series(from_module, from_component_name)

        # # exchange: get the to values
        # to_values = self.create_exchange_series(to_module, to_component_name)

        # # exchange: self.interchange.exchange(from, to, )


        pass

    def create_exchange_series(
        self,
        module: pyo.ConcreteModel | pyo.Block,
        component_name: str
    ):

        # # create_exchange_series: get the values
        # values = self.get_component_values(module, component_name)

        # # create_exchange_series: cross-reference the indices against the dimensions in interchange
        # index_dict, dimension_dict = self.get_index_dictionaries(module, component_name)

        # # create_exchange_series: initialize ExchangeSeries object
        # ex_series = IC.ExchangeSeries(values, index_dict, dimension_dict)
        # return ex_series
        pass

    def get_component_values(
        self,
        module: pyo.ConcreteModel | pyo.Block,
        component_name: str
    ) -> Dict:
        # # get_exchange_series: get a hook to the component of interest
        # component = getattr(module, component_name)

        # # # get_exchange_series: based on instance type, route to necessary pull
        # if isinstance(component, (pyo.Var, pyo.Param)):
        #     values = component.extract_values()
        # elif isinstance(component, pyo.Constraint):
        #     values = module.get_duals(component_name)
        # return values
        pass

    def get_index_dictionaries(
        self,
        module: pyo.ConcreteModel | pyo.Block,
        component_name: str
    ) -> List:
        # # get_dimension_labels: get index labels from cols_dict
        # index_labels = module.cols_dict[component_name]
        # index_dict = {}
        # for label in index_labels:
        #     index_dict[index_labels.index(label)] = label
        
        # # get_dimension_labels: get a hook to the interchange dimensions object
        # dimensions = self.interchange.dimensions

        # # # get_dimension_labels: for each dimension in interchange, create labels
        # dimension_labels = {}
        # for dim in dimensions:
        #     dimension_labels[dim] = []
        #     for label in index_labels:
        #         if label in dimensions[dim].units:
        #             dimension_labels[dim].append(index_labels.index(label))

        # # # get_dimension_labels: return dimension_labels list
        # return index_dict, dimension_labels        
        pass

    def solve(
        self,
        model_name: str,
        tee: bool | None = False,
        persistent: bool | None = False,
        **kwargs
    ):
        """Solve a pyo Model instance

        Parameters
        ----------
        model_name : str
            Name of model
        tee : bool | None, optional
            Print out full traceback, by default False
        persistent : bool | None, optional
            Set up persistent solver, by default False

        Raises
        ------
        RuntimeError
            Raise error if no optimal solution found
        """
        # solve: get a solver object
        opt = self.get_solver()

        # solve: hook to the model via the name unpacked from the args
        model = getattr(self.meta, model_name)

        # solve: solve the model
        res = opt.solve(model, tee = tee)

        # solve: if optimal, update tolerance isntance, else throw error
        if pyo.check_optimal_termination(res):
            self.tolerance.update_model(model_name, pyo.value(model.objective))
        else:
            raise RuntimeError(f"non-optimal termination for model {model_name} in cycle {self.tolerance.cycle}")

    @staticmethod
    def get_solver():
        """Returns a solver

        Returns
        -------
        pyo.SolverFactory
            The pyomo solver
        """
        # default = linear solver
        solver_name = 'appsi_highs'
        opt = pyo.SolverFactory(solver_name)
        return opt




