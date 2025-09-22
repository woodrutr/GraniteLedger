"""faster_sensitivity

This file contains the class SensitivityMatrix which  takes in sympy objects that have been converted from pyomo. It builds the matrix of partials to be used in sensitivity analysis.

It also contains class AutoSympy which takes in pyomo models and converts the objects into sympy.

Finally, it contains class toy_model, the sensitivity method in action and then runs toy_model with input n=5.

The file babymodel.py can be also use this method by importing this file instead of sensitivity_tools.py by:
    from faster_sensitivity import *
"""

import sympy as sp
import pyomo.environ as pyo
import numpy as np
from datetime import *

# TODO: redo datetime wildcard import to avoid sphinx issues
import random


class SensitivityMatrix:
    """This class takes in sympy objects that have been converted from pyomo.
    It builds the matrix of partials to be used in sensitivity analysis.
    """

    def __init__(self, sympification, duals, parameters_of_interest):
        self.sympification = sympification
        self.duals = duals
        self.variable_vector = list(sympification['variable_values'].keys())
        self.parameter_vector = list(parameters_of_interest['parameter_values'].keys())
        self.parameters_of_interest = parameters_of_interest
        self.parameter_map = sympification['parameter_map']
        self.variable_map = sympification['variable_map']

        self.components, self.subs_dict = self.generate_matrix()
        startAssembly = datetime.now()
        self.U, self.S = self.matrix_assembly(self.components, self.subs_dict)
        endAssembly = datetime.now()
        print(f'Assembly done in {endAssembly - startAssembly}')
        self.U_inv = sp.Matrix()
        self.partials = self.get_partials_matrix()
        print(f'Partials done in {datetime.now() - endAssembly}')

    def new_jacobian(self, f, values, map):
        """A function that returns the same result as Matrix.jacobian(values).
        This speeds up runtime by only taking derivatives of symbols that exist.
        The original function takes the derivative wrt everything in values.

        Parameters
        ----------
        f : sp.Matrix
            Matrix of equations
        values : list
            List of symbols that the function will take derivatives with respect to
        map : dict
            Dictionary of column locations for each symbol

        Returns
        -------
        sp.Matrix
            Returns jacobian of given f matrix
        """
        M = sp.zeros(len(f), len(values))

        for i in range(len(f)):
            for j in f[i].free_symbols:
                if j in values:
                    M[i, map[j]] = sp.diff(f[i], j)
        return M

    def matrix_sub(self, M, subs):
        """A function that substitutes values into a matrix.
        This is the same result as sp.Matrix().subs(subs).
        This speeds up runtime by only attempting to substitute values into symbols that actually exist in each cell.

        Parameters
        ----------
        M : sp.Matrix
            The matrix to have symbols substituted for values
        subs : dict
            Dictionary of values for sympy symbols

        Returns
        -------
        sp.Matrix
            The original matrix with all given values substituted into their symbols
        """
        i = 0
        for cell in M:
            for symbol in cell.free_symbols:
                cell = cell.subs({symbol: subs[symbol]})
            M[i] = cell
            i += 1
        return M

    def generate_matrix(self):
        """This creates all of the matrices that will be combined into the U and S matrices.

        Returns
        -------
        dict, dict
            Returns 2 dictionaries. The first is the dictionary of matrix components with their names as keys.
            The second dictionary is a map from the symbols to their values.
        """
        subs_dict = {}
        subs_dict.update(self.sympification['variable_values'])
        subs_dict.update(self.sympification['parameter_values'])
        now = datetime.now()
        f = sp.Matrix([self.sympification['objective']])
        fdone = datetime.now()
        print(f'f created in {fdone - now}')
        h = sp.Matrix([([h_k]) for h_k in self.sympification['equality_constraints'].values()])
        Lambda = sp.Matrix([tuple(dual for dual in self.duals['equality_duals'].values())])
        hdone = datetime.now()
        print(f'h and lambda created in {hdone - fdone}')
        Hx = self.new_jacobian(h, self.variable_vector, self.variable_map)
        Ha = self.new_jacobian(h, self.parameter_vector, self.parameter_map)
        hxdone = datetime.now()
        print(f'hx and ha done in {hdone - hxdone}')
        g = sp.Matrix(
            [
                self.sympification['inequality_constraints'][g_j]
                for g_j in self.duals['inequality_duals'].keys()
            ]
        )
        Mu = sp.Matrix([tuple(dual[0] for dual in self.duals['inequality_duals'].values())])
        gdone = datetime.now()
        print(f'g and mu done in {gdone - hxdone}')
        Gx = self.new_jacobian(g, self.variable_vector, self.variable_map)
        Ga = self.new_jacobian(g, self.parameter_vector, self.parameter_map)
        gxdone = datetime.now()
        print(f'gx and ga done in {gxdone - gdone}')
        Fx = f.jacobian(self.variable_vector)
        Fa = f.jacobian(self.parameter_vector)
        fxdone = datetime.now()
        print(f'fx and fa done in {fxdone - gxdone}')
        Fxx = self.new_jacobian(Fx, self.variable_vector, self.variable_map)
        if h:
            Fxx = Fxx + self.new_jacobian(Lambda * Hx, self.variable_vector, self.variable_map)
        if g:
            Fxx = Fxx + self.new_jacobian(Mu * Gx, self.variable_vector, self.variable_map)
        fxxdone = datetime.now()
        print(f'fxx done in {fxxdone - fxdone}')
        Fxa = self.new_jacobian(Fx, self.parameter_vector, self.parameter_map)
        if h:
            Fxa = Fxa + self.new_jacobian(Lambda * Hx, self.parameter_vector, self.parameter_map)
        if g:
            Fxa = Fxa + self.new_jacobian(Mu * Gx, self.parameter_vector, self.parameter_map)
        fxadone = datetime.now()
        print(f'fxa done in {fxadone - fxxdone}')
        matrix_components = {
            'Fx': Fx,
            'Fa': Fa,
            'Fxx': Fxx,
            'Fxa': Fxa,
            'Hx': Hx,
            'Ha': Ha,
            'Gx': Gx,
            'Ga': Ga,
        }

        return matrix_components, subs_dict

    def matrix_assembly(self, components, subs_dict):
        """Combines matrix components to create U and S matrices from the literature.

        Parameters
        ----------
        components : dict
            Dictionary of all precalculated matrix components
        subs_dict : dict
            Dictionary that maps symbols to their values

        Returns
        -------
        sp.Matrix, sp.Matrix
            Returns the U and S matrices respectively with all symbols replaced by corresponding values
        """
        now = datetime.now()
        U1 = sp.Matrix.vstack(
            components['Fx'], components['Fxx'], components['Hx'], components['Gx']
        )

        U2 = sp.Matrix.vstack(
            sp.Matrix.zeros(1, components['Hx'].transpose().cols),
            components['Hx'].transpose(),
            sp.Matrix.zeros(components['Hx'].rows),
            sp.Matrix.zeros(components['Gx'].rows, components['Hx'].transpose().cols),
        )

        U3 = sp.Matrix.vstack(
            sp.Matrix.zeros(1, components['Gx'].transpose().cols),
            components['Gx'].transpose(),
            sp.Matrix.zeros(components['Hx'].rows, components['Gx'].transpose().cols),
            sp.Matrix.zeros(components['Gx'].rows),
        )

        U4 = sp.Matrix.vstack(
            -1 * sp.Matrix.ones(1),
            sp.Matrix.zeros(
                components['Fxx'].rows + components['Hx'].rows + components['Gx'].rows, 1
            ),
        )
        # print(U1.shape,U2.shape,U3.shape,U4.shape)

        U = sp.Matrix.hstack(U1, U2, U3, U4)
        udone = datetime.now()
        print(f'U assembled in {udone - now}')
        S = sp.Matrix.vstack(
            components['Fa'], components['Fxa'], components['Ha'], components['Ga']
        )
        sdone = datetime.now()
        print(f'S assembled in {sdone - udone}')
        # return U.subs(subs_dict), S.subs(subs_dict)
        U = self.matrix_sub(U, subs_dict)
        uSubs = datetime.now()
        print(f'U substituted in {uSubs - sdone}')
        S = self.matrix_sub(S, subs_dict)
        sSubs = datetime.now()
        print(f'S substituted in {sSubs - uSubs}')
        return U, S

    def invert_U(self):
        """Calculates the inverse of the U matrix.
        The fastest method found for this so far has been to convert to numpy and use its inverse function

        Returns
        -------
        np.ndarray
            Calculated matrix for the inverse of U
        """
        now = datetime.now()
        print(f'starting U inv')
        new_M = np.array(self.U).astype(np.float64)
        inv = np.linalg.inv(new_M)
        # the inverse method comes back with lots of numbers that are 1e-16 and smaller away from the actual answer
        inv = inv.round(10)
        done = datetime.now()
        print(f'U inv calculated in {done - now}')
        return inv

    def get_partials_matrix(self):
        """Calculate the matrix of all partials as U^(-1) * S
        Thus far, this is found to run the fastest when U^(-1) and S are numpy arrays

        Returns
        -------
        np.ndarray
            Full partials matrix
        """
        if not self.U_inv:
            self.U_inv = self.invert_U()
        return np.matmul(self.U_inv, np.array(self.S).astype(np.float64)).round(10)

    def get_partial(self, x, a):
        """Retrieve the value of a particular partial derivative.
        The value retrieved will be dx/da.

        Parameters
        ----------
        x : sp.Symbol
            The symbol for the variable that you wish to know the change effect
        a : sp.Symbol
            The symbol for the parameter that you wish to change to cause an effect on a variable

        Returns
        -------
        float
            The value of the partial derivative dx/da
        """
        if not self.partials:
            self.partials = self.get_partials_matrix()
        return self.partials[self.variable_map[x], self.parameter_map[a]]

    def get_sensitivity_range(self, x, a, percent):
        """The estimated values for "x" if the parameter "a" changes by percent% (as number 0% to 100%).
        It will return values for an increase and decrease of the percent given.

        Parameters
        ----------
        x : sp.Symbol
            The symbol for the variable that you wish to know the change effect
        a : sp.Symbol
            The symbol for the parameter that you wish to change to cause an effect on a variable
        percent : float
            A number 0-100 for the percent change in "a"

        Returns
        -------
        float, float
            Returns the estimated value for "x" if "a" is increased by percent% and decreased by percent%
        """
        original_x = self.sympification['varible_values'][x]
        original_a = self.sympification['parameter_values'][a]
        derivative = self.get_partial(x, a)
        change = percent / 100
        high = original_x + derivative * change * original_a
        low = original_x - derivative * change * original_a
        return high, low


class AutoSympy:
    """This class take in pyomo models and converts the objects into sympy.
    This is useful for problems that needs methods such as derivatives to be calculated on the equations.
    We use these derivatives to calculate a sensitivity matrix that estimates the changes in variables due to changes in parameters
    """

    def __init__(self, model):
        self.model = model

        self.param_position_map = {}
        self.var_position_map = {}
        now = datetime.now()
        self.sets, self.set_values = self.get_sets()
        afterSets = datetime.now()
        print(f'Sets done in {afterSets - now}')
        self.parameters, self.parameter_values, self.parameter_index_sets, self.symbol_map = (
            self.get_parameters()
        )
        afterParams = datetime.now()
        print(f'Parameters done in {afterParams - afterSets}')
        self.variables, self.variable_values = self.get_variables()
        afterVars = datetime.now()
        print(f'Variables done in {afterVars - afterParams}')
        self.equality_constraints, self.inequality_constraints = self.get_constraints()
        afterCons = datetime.now()
        print(f'Constraints done in {afterCons - afterVars}')
        self.objective = self.get_objective()
        afterObj = datetime.now()
        print(f'Objective done in {afterObj - afterCons}')
        self.duals = self.generate_duals(
            self.model.component_objects(pyo.Constraint), self.model.dual
        )
        afterDual = datetime.now()
        print(f'Duals done in {afterDual - afterObj}')

    def get_sets(self):
        """Convert pyomo sets into sympy indexes

        Returns
        -------
        dict, dict
            The first dictionary has the pyomo objects' names as keys and newly created sympy indexes as values.
            The second dictionary has the new sympy indexes as keys and the pyomo sets' values as the dict values.
        """
        sets = {s.name: sp.Idx(s.name) for s in self.model.component_objects(pyo.Set)}

        set_values = {sets[s.name]: s.data() for s in self.model.component_objects(pyo.Set)}

        return sets, set_values

    def get_parameters(self):
        """Convert pyomo parameters into sympy objects.
        This procedure creates sympy IndexedBase objects and sympy Symbol objects of similar names.
        The IndexedBase datatype is necessary to parse the equations, but it does not work well with derivatives.
        We will substitute in Symbols when the equations are all created, so they need to map to each other.
        To keep the columns in order through all procedures, all parameters are given a unique column number by the variable "position"
        This position is stored in the class dict param_position_map

        Returns
        -------
        dict, dict, dict, dict
            Returns 4 dictionaries:
            parameters: keys are pyomo parameter names and values are sympy IndexedBase objects with the same name
            parameter_values: keys are sympy symbols and values are the numerical values of the pyomo objects
            parameter_index_sets: keys are pyomo parameter names and values are lists of that parameters indices
            symbol_map: keys are pyomo parameters with an index and values are sympy symbols with a similarly styled name and index
        """
        parameters = {}
        parameter_values = {}
        parameter_index_sets = {}
        symbol_map = {}

        # initializes first column position for a parameter
        position = 0
        for p in self.model.component_objects(pyo.Param):
            if p.is_indexed():
                # create IndexedBase object and all indices it will use
                parameters[p.name] = sp.IndexedBase(p.name)
                parameter_index_sets[p.name] = list(p.index_set())

                for index in p.index_set():
                    # create accompanying Symbol object with index in name
                    # map IndexedBase indexed object to Symbol object
                    # Pull numerical value from pyomo for final substitution from Symbol
                    # Assign unique column number to Symbol
                    token = sp.Symbol(f'{p.name}_({index})')
                    symbol_map[parameters[p.name][index]] = token
                    parameter_values[token] = p.extract_values()[index]
                    self.param_position_map[token] = position
                    position += 1

            else:
                # non-indexed parameters go directly to sympy Symbols
                # numerical value and column number are stored as with indexed parameters
                parameters[p.name] = sp.Symbol(p.name)
                parameter_values[parameters[p.name]] = p.value
                parameter_index_sets[p.name] = False
                self.param_position_map[sp.Symbol(p.name)] = position
                position += 1

        return parameters, parameter_values, parameter_index_sets, symbol_map

    def get_variables(self):
        """Convert pyomo variables into sympy objects.
        This procedure creates sympy IndexedBase objects and sympy Symbol objects of similar names.
        The IndexedBase datatype is necessary to parse the equations, but it does not work well with derivatives.
        We will substitute in Symbols when the equations are all created, so they need to map to each other.
        To keep the columns in order through all procedures, all parameters are given a unique column number by the variable "position"
        This position is stored in the class dict param_position_map

        Returns
        -------
        dict, dict
            Returns 2 dictionaries:
            variables: keys are pyomo variable names and values are sympy IndexedBase objects with the same name
            variable values: keys are sympy symbols and values are the numerical values of the pyomo objects
            It is also worth mentioning that this adds entries to the self.symbol_map in the same way parameters do.
            Symbol map entries have keys of IndexedBase objects and the values are their associated sympy Symbol
        """

        variables = {}
        variable_values = {}

        position = 0
        for v in self.model.component_objects(pyo.Var):
            if v.is_indexed():
                # creates IndexedBase object
                variables[v.name] = sp.IndexedBase(v.name)
                for index in v.index_set():
                    # create accompanying Symbol object with index in name
                    # map IndexedBase indexed object to Symbol object
                    # Pull numerical value from pyomo for final substitution from Symbol
                    # Assign unique column number to Symbol
                    token = sp.Symbol(f'{v.name}_({index})')
                    self.symbol_map[variables[v.name][index]] = token
                    variable_values[token] = v[index].value
                    self.var_position_map[token] = position
                    position += 1
            else:
                # non-indexed parameters go directly to sympy Symbols
                # numerical value and column number are stored as with indexed parameters
                variables[v.name] = sp.Symbol(v.name)
                variable_values[variables[v.name]] = v.value
                self.var_position_map[sp.Symbol(v.name)] = position
                position += 1

        return variables, variable_values

    def get_constraints(self):
        """This function converts all of the constraints in the pyomo object and converts the pyomo expressions into sympy expressions.

        Returns
        -------
        dict, dict
            Returns 2 dictionaries:
            equality_constraints: keys are tuples (constraint_name, index) and values are sympy expressions
            inequality_constraints: keys are tuples (constraint_name, index) and values are sympy expressions
        """

        # create a substitution dictionary with the sets, parameters, and variables
        # this will be fed to a parser to convert pyomo expressions into sympy expressions
        local_dict = self.sets.copy()
        local_dict.update(self.parameters.copy())
        local_dict.update(self.variables.copy())

        # keep track of which constraints are equality and which are inequality
        equality_constraints = {}
        inequality_constraints = {}

        for c in self.model.component_objects(pyo.Constraint):
            # Each conditional follows the same format:
            # Use conversion function parse_expr to convert from pyomo expression to sympy expression.
            # side note: pyo.sympy_tools.sympyify_expression would not read over mutable parameters.
            # side note: if sympyify begins working correctly, this should be sped up by going straight to symbols
            # check which objects are Indexed in the parsed expression and convert them to Symbols
            # add expression to equality or inequality constraints as defined
            if c.is_indexed():
                for index, expr in c.items():
                    # indexed constraints must have each index analyzed
                    if expr.equality:
                        temp = sp.parse_expr(expr.body.to_string(), local_dict=local_dict)
                        for free in temp.free_symbols:
                            if type(free) == sp.Indexed:
                                temp = temp.subs({free: self.symbol_map[free]})
                        equality_constraints[(c.name, index)] = temp
                    else:
                        temp = sp.parse_expr(expr.body.to_string(), local_dict=local_dict)
                        for free in temp.free_symbols:
                            if type(free) == sp.Indexed:
                                temp = temp.subs({free: self.symbol_map[free]})
                        inequality_constraints[(c.name, index)] = temp
            else:
                if c.equality:
                    temp = sp.parse_expr(c.body.to_string(), local_dict=local_dict)
                    for free in temp.free_symbols:
                        if type(free) == sp.Indexed:
                            temp = temp.subs({free: self.symbol_map[free]})
                    equality_constraints[(c.name, None)] = temp
                else:
                    temp = sp.parse_expr(c.body.to_string(), local_dict=local_dict)
                    for free in temp.free_symbols:
                        if type(free) == sp.Indexed:
                            temp = temp.subs({free: self.symbol_map[free]})
                    inequality_constraints[(c.name, None)] = temp

        return equality_constraints, inequality_constraints

    def get_objective(self):
        """This converts the pyomo objective function into a sympy function.

        Returns
        -------
        sympy equation
            The pyomo objective function converted into sympy
        """
        # create a substitution dictionary with the sets, parameters, and variables
        # this will be fed to a parser to convert pyomo expressions into sympy expressions
        local_dict = self.sets.copy()
        local_dict.update(self.parameters.copy())
        local_dict.update(self.variables.copy())

        # Use conversion function parse_expr to convert from pyomo expression to sympy expression.
        # side note: pyo.sympy_tools.sympyify_expression would not read over mutable parameters.
        # side note: if sympyify begins working correctly, this should be sped up by going straight to symbols
        # check which objects are Indexed in the parsed expression and convert them to Symbols
        # add expression to equality or inequality constraints as defined
        for obj in self.model.component_objects(pyo.Objective):
            if obj.active:
                temp = sp.parse_expr(obj.expr.to_string(), local_dict=local_dict)
                for free in temp.free_symbols:
                    if type(free) == sp.Indexed:
                        temp = temp.subs({free: self.symbol_map[free]})
                objective = temp
        return objective

    def generate_duals(self, constraints, duals):
        """Uses dual values and slack values to classify each constraint. It also stores the dual values for substitution later.

        Parameters
        ----------
        constraints : model.component_objects(pyo.Constraint)
            All of the constraint objects from the pyomo model
        duals : model.dual
            All of the dual (or Suffix) objects from the pyomo model

        Returns
        -------
        dict of lists and dicts
            _description_
        """

        deletions = []
        degenerates = []
        equality_duals = {}
        inequality_duals = {}

        i = 0
        for d in constraints:
            for index in d:
                if d[index].equality:
                    equality_duals[(d.name, index)] = duals[d[index]]
                elif d[index].slack() > 0:
                    deletions.append(((d.name, index), i))
                    i += 1

                elif duals[d[index]] != 0 and d[index].slack() == 0:
                    inequality_duals[(d.name, index)] = (
                        duals[d[index]],
                        d[index].slack(),
                        duals[d[index]] * d[index].slack() == 0
                        and not (duals[d[index]] == 0 and d[index].slack() == 0),
                    )
                    i += 1
                else:
                    inequality_duals[(d.name, index)] = (
                        duals[d[index]],
                        d[index].slack(),
                        duals[d[index]] * d[index].slack() == 0
                        and not (duals[d[index]] == 0 and d[index].slack() == 0),
                    )
                    degenerates.append(((d.name, index), i))
                    i += 1

        return {
            'deletions': deletions,
            'degenerates': degenerates,
            'equality_duals': equality_duals.copy(),
            'inequality_duals': inequality_duals.copy(),
        }

    def check_complimentarity_all(self):
        for value in self.inequality_duals.values():
            if value[0] * value[1] == 0 and not (value[0] == 0 and value[1] == 0):
                print('complimentarity slack')
            elif value[0] * value[1] != 0:
                print("something's wrong")
            else:
                print('active constraint with 0 dual')

    def get_sensitivity_matrix(self, parameters_of_interest=None):
        """This function gathers all of the new sympy objects and creates a SensitivityMatrix object.

        Parameters
        ----------
        parameters_of_interest : dict, optional
            Specified subset of the parameters if more information is known about needless parameters, by default None

        Returns
        -------
        SensitivityMatrix
            a SensitivityMatrix object that contains the sensitivity matrix and commands to use it.
        """

        if parameters_of_interest:
            params = {p: self.parameters[p] for p in parameters_of_interest}
            param_values = {}
            for p in parameters_of_interest:
                if self.parameter_index_sets[p]:
                    for index in self.parameter_index_sets[p]:
                        param_values[self.parameters[p][index]] = self.parameter_values[
                            self.parameters[p][index]
                        ]
                else:
                    param_values[self.parameters[p]] = self.parameter_values[self.parameters[p]]
        else:
            params = self.parameters.copy()
            param_values = self.parameter_values.copy()

        sympification = {
            'sets': self.sets.copy(),
            'set_values': self.set_values.copy(),
            'parameters': self.parameters.copy(),
            'parameter_values': self.parameter_values.copy(),
            'variables': self.variables.copy(),
            'variable_values': self.variable_values.copy(),
            'equality_constraints': self.equality_constraints.copy(),
            'inequality_constraints': self.inequality_constraints.copy(),
            'objective': self.objective,
            'duals': self.duals,
            'parameter_map': self.param_position_map,
            'variable_map': self.var_position_map,
        }
        of_interest = {'parameters': params.copy(), 'parameter_values': param_values.copy()}
        print('going to sensitivity matrix')
        return SensitivityMatrix(sympification, self.duals, of_interest)


class toy_model:
    """An example of the method in action that scales by the given 'n' value"""

    def __init__(self, n):
        self.n = n

        mod = self.create_model()

        sympy_model = AutoSympy(mod)

        sensitivity = sympy_model.get_sensitivity_matrix()

        np.savetxt('partials.csv', sensitivity.partials, '%d')
        # print(type(sensitivity.partials))
        # print(sensitivity.partials)

    def create_model(self):
        n = self.n

        mod = pyo.ConcreteModel()

        mod.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

        # Sets
        mod.a = pyo.Set(initialize=[f'a{i}' for i in range(n)])

        random.seed(12345)
        # Parameters
        mod.max = pyo.Param(
            mod.a, initialize={f'a{i}': random.randint(1, 100) for i in range(n)}, mutable=True
        )
        mod.x_cost = pyo.Param(
            mod.a, initialize={f'a{i}': random.randint(1, 10) for i in range(n)}, mutable=True
        )
        mod.demand = pyo.Param(
            mod.a, initialize={f'a{i}': random.randint(1, 50) for i in range(n)}, mutable=True
        )

        # Variables
        mod.x = pyo.Var(mod.a)

        # Constraints
        @mod.Constraint(mod.a)
        def constraint1(mod, a):
            return mod.x[a] - mod.max[a] <= 0

        @mod.Constraint()
        def constraint2(mod):
            return sum(mod.demand[i] for i in mod.a) - sum(mod.x[i] for i in mod.a) <= 0

        @mod.Constraint(mod.a)
        def constraint3(mod, a):
            return -1 * mod.x[a] <= 0

        @mod.Objective()
        def obj(mod):
            return sum(mod.x_cost[i] * mod.x[i] for i in mod.a)

        solution = pyo.SolverFactory('appsi_highs').solve(mod, tee=True)

        return mod


# toy_model(5)
