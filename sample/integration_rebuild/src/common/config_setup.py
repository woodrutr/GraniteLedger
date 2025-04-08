"""This script houses the Config_settings class; this unpacks the configuration 
contained within run_config.toml and handles the construction of the solve algorithm.

From the configuration file, Config_settings creates a list of modules to instantiate and 
actions to take to solve the model (e.g. exchange parameters, build an expression, etc.).

"""

# Import packages
import tomllib
from pathlib import Path
from collections import defaultdict
import types
import argparse
from itertools import product

class Config_settings:
    """Config_settings, upon initialization creates the following items needed for instantiating 
    a Build:

    settings.modules: A dicitonary of the modules to initialize, including the desired name, the class 
    used to created the instance, and other options.

    settings.crosswalks: Details on which crosswalks are needed to instantiate Interchange, the system of 
    value exchange used in the build.execute method.

    settings.classes: A dictionary of the classes (and the number of instances) needed to build the model 
    implied by the configuration file

    settings.dual_exchanges: All of the required exchanges of parameters between modules, based on solve method and
    the integration specified in the configuration.

    settings.solve_algorithm: The final "recipe" for solving the constructed model. This dictionary describes a 
    solve cycle based on the number of modules, the integration structure, and the solve method. Composed of 
    "actions" indexed by step (e.g. solve_algorithm[3]["action"] = "exchange"). Options contained in the dictionary
    describe how to conduct the "action".
    """
    def __init__(
        self,
        config_path: Path,
        args: argparse.Namespace | None = None
    ):

        # __INIT__: Find config
        config_path = config_path / "src" / "common" / "run_config.toml"

        # __INIT__: Pull args
        self.args = args
        if not args:
            self.args = types.SimpleNamespace()
            self.args.op_mode = None
            self.args.debug = False

        # __INIT__: Dump toml
        with open(config_path, 'rb') as src:
            config = tomllib.load(src)

        # __INIT__: Assign scales
        self.regions = config['scales']['regions']
        self.years = config['scales']['years']
        self.techs = config['scales']['techs']

        # __INIT__: Main.py options
        # self.return_model = config["main_options"]["return_model"]
        # self.save_tolerance = config["main_options"]["save_tolerance"]

        # __INIT__: Solve/Solver options
        self.solve_options = config["solve_options"]

        # __INIT__: Set up Interchange
        self.unpack_crosswalks(config)

        # __INIT__: Module control (creates compilation instructions for module builds)
        self.unpack_modules(config)
        self.unpack_duals(config)
        self.unpack_build(config)

    def unpack_crosswalks(self, config):
        """Unpack crosswalk settings from configuration.

        Parameters
        ----------
        config : dict
            Raw toml loaded from configuration
        """
        
        # unpack_crosswalks: Grab unique entries in TOML file
        dimensions = [dim for dim in config["integration"]["crosswalks"]]
        crosswalks = config["integration"]["crosswalks"]

        # unpack_crosswalks: unpack settings from TOML into dictionary entries used later in initializing UnitDimension objects
        self.dimensions = types.SimpleNamespace()
        for dimension in dimensions:
            # unpack_crosswalks: dimension loop -- create new namespace
            namespace = types.SimpleNamespace()

            # unpack_crosswalks: dimension loop -- create path and dim_name attributes
            namespace.path = Path() / "src" / "interchange" / "crosswalks" / crosswalks[dimension]["filename"]
            namespace.dim_name = dimension
            
            if "base_unit" in crosswalks[dimension]:
                namespace.base_unit = crosswalks[dimension]["base_unit"]
            else:
                namespace.base_unit = None
            setattr(self.dimensions, dimension, namespace)

    def unpack_modules(self, config):
        """Unpacks modules required for model build.

        Note that a LowRes instance is created for each technology in the configuration file

        Parameters
        ----------
        config : dict
            Raw toml loaded from configuration
        """

        # UNPACK_MODULES: Unpack modules turned on
        modules = [module for module in config["modules"] if config["modules"][module]]

        # UNPACK_MODULES: Create module build recipe w/ number of modules
        model_build = defaultdict()
        classes = defaultdict()

        # UNPACK_MODULES: For each module, update model_build dictionary with name, class, and tech (for LowRes)
        for module in modules:
            classes[module] = []
            match module: 
                case "HighRes":
                    mname = "highres"
                    model_build[mname] = {
                        "name": mname,
                        "class": module,
                        "tech": None
                    }
                    classes[module].append(mname)
                # UNPACK_MODULES: Create a lowres instance for each tech in techs
                case "LowRes":
                    for tech in self.techs:
                        mname = f"lowres{tech}"
                        model_build[mname] = {
                            "name": mname,
                            "class": module,
                            "tech": tech
                        }
                        classes[module].append(mname)
                case _:
                    raise ValueError(f"run_config includes settings for a non-existent module class: {module}")

        # UNPACK_MODULES: Assign modules and classes dictionaries to settings
        self.modules = model_build
        self.classes = classes

    def unpack_duals(self, config):
        """Unpacks the dual-integration described in configuration into instructions.

        Parameters
        ----------
        config : dict
            Raw toml loaded from configuration
        """
        duals = config["integration"]["dual"]
        dual_exchanges = defaultdict()

        # UNPACK_DUALS: For each exchange in configuration, unpack the instructions for each instance of classes
        for name in duals:
            dual_rule = duals[name]
            from_class_name = dual_rule["from_class"]
            to_class_name = dual_rule["to_class"]

            # UNPACK_DUALS: Get from and to component names from raw configuration
            from_class_component = dual_rule["from_component"]
            to_class_component = dual_rule["to_component"]

            # UNPACK_DUALS: get class instance counts from settings.classes
            from_class_instances = self.classes[from_class_name]
            to_class_instances = self.classes[to_class_name]

            # UNPACK_DUALS: For each of the exchanges, append to dual_exchanges
            set_of_model_exchanges = product(set(from_class_instances), set(to_class_instances))
            for from_model, to_model in set_of_model_exchanges:
                dual_exchanges[(from_model, to_model)] = {
                    "from_module": from_model,
                    "from_component": from_class_component,
                    "to_module": to_model,
                    "to_component": to_class_component
                }
        
        # UNPACK_DUALS: assign unpacked options to build_recipe
        self.dual_exchanges = dual_exchanges
    
    def unpack_build(self, config):
        # UNPACK_BUILD: Grab whether gs or unified solve algorithm
        self.solve_method = config["integration"]["options"]["solve_method"]

        # UNPACK_BUILD: Create simple object names for already-built portions of the recipe
        modules = self.modules
        duals = self.dual_exchanges

        # UNPACK_BUILD: Unpack duals stored by class and assemble build list
        solve_algorithm = defaultdict()

        # UNPACK_BUILD: Create solve algorithm instructions based on method in configuration
        match self.solve_method:
            case "gauss-seidel":
                elem = 0
                for module in modules:
                    # get any dual passing that exchanges *into* the module
                    duals_into_module = [index for index in duals if index[1] == module]
                    for index in duals_into_module:
                        solve_algorithm[elem] = {
                            "action":"exchange",
                            "options":duals[index]
                        }
                        elem += 1
                    
                    # Add solve element to solve algorithm
                    solve_algorithm[elem] = {
                        "action": "solve",
                        "options": {
                            "model_name": module,
                            "tee": config["solve_options"]["tee"],
                            "persistent": config["solve_options"]["persistent"]
                        }
                    }
                    elem += 1
                self.solve_algorithm = solve_algorithm          
            case _: # PLACEHOLDER FOR UNIFIED SOLVES
                pass
