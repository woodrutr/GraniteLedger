"""This script houses the Tolerance class, an object that flexibly handles checking solve
convergence by tracking objective values over time for each module instance constructed in
an instance of Build.
"""

from graniteledger.src.common.config_setup import Config_settings

class Tolerance():
    """Tolerance object handles checking convergence of any execution of an integrated model
    Build. Object keeps track of modules in an instance of Build, as well as the current cycle of
    the solve algorithm.

    Build instantiates Tolerance and adds models to it as they are constructed. Tolerance checks happen 
    using the objective values uploaded to Tolerance as each solve occurs (as specified in settings.solve_algorithm) 
    and executed in Build.
    """
    def __init__(
        self,
        settings: Config_settings
    ):
        # __INIT__: Create dictionary to hold models and list of objective values for each solve
        self.models = {}
        self.n_models = 0
        self.cycle = 0
        
        # __INIT__: Unpack solve options
        for option in settings.solve_options:
            setattr(self, option, settings.solve_options[option])

    def add_model(
        self,
        model_name: str
    ):
        """Adds a module to the Tolerance class instance and updates the count stored.

        Parameters
        ----------
        model_name : str
            Name of instance
        """
        self.models[model_name] = {}
        self.n_models += 1
    
    def update_model(
        self,
        model_name: str,
        obj_value: int | float
    ):
        """Updates model dictionary with objective value from a solve.

        Parameters
        ----------
        model_name : str
            Name of instance
        obj_value : int | float
            Value of objective from latest solve of instance
        """
        self.models[model_name][self.cycle] = obj_value

    def check(
        self,
        round_to: int | None = 4
    ) -> bool:
        """Checks whether tolerance met based on method in configuration:

        "average" method takes the average change in objectives from cycle to cycle and checks whether
        it is below "tol"

        "all" method ensures every model instance contained in meta demonstrates changes in objective values
        from cycle to cycle below "tol"

        Parameters
        ----------
        round_to : int | None, optional
            Decimal to round values to, by default 4

        Returns
        -------
        bool
            If True, tolerance met, finish execution, if False, continue
        """
        # check: if first cycle, return none
        if self.cycle == 0:
            return None
        
        # check: calcuate delta for each model in tolerance
        delta = {}
        for model_name in self.models:
            prev_obj = self.models[model_name][self.cycle - 1]
            cur_obj = self.models[model_name][self.cycle]

            delta[model_name] = round((cur_obj - prev_obj)/(prev_obj + 0.1), round_to)
        
        # check: based on tolerance method in settings, check tolerance
        match self.tol_method:
            case "average":
                sum_delta = 0
                for model_name in delta:
                    sum_delta += delta[model_name]
                check_value = round(sum_delta/self.n_models, round_to)
                below_tolerance = check_value <= self.tol
            case "all":
                below_tolerance = True
                for model_name in delta:
                    if delta[model_name] >= self.tol:
                        below_tolerance = False
        self.cycle += 1
    
        # check: with below_tolerance and settings, determine whether to proceed to next cycle
        finish = False
        # below_tolerance check
        if below_tolerance:
            finish = True

        # max iter check
        if (self.cycle >= self.max_iter):
            finish = True
        
        # force_10 check
        if self.force_10 and self.cycle < 10:
            finish = False
        return finish

