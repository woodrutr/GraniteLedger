"""Sensitivity Tools
This file contains the AutoSympy, SensitivityMatrix, CoordMap, and DifferentialMapping classes. These
classes serve as the data structures and containers of methods to get from a Pyomo ConcreteModel to easily
accessible sensitivities.

The flow goes:

Pyomo model -> AutoSympy -> SensitivityMatrix -> DifferentialMapping

sensitivities and sensitivity-based calculations can be done through the DifferentialMapping object.

"""
###################################################################################################
# Setup

# Import packages
from collections import defaultdict
from logging import getLogger
import pyomo.environ as pyo
from sympy import *

# TODO: redo sympy wildcard import to avoid sphinx issues
import time

# Establish logger
# logger = getLogger(__name__)

###################################################################################################


class AutoSympy:
    def __init__(self, model):
        """Extracts features from Pyomo model, converts them to Sympy, stores them in dictionaries.

        Parameters
        ----------
        model : obj
            a solved Pyomo ConcreteModel instance.
        """

        self.model = model
        print('sympifying')
        self.sets, self.set_values = self.get_sets()
        self.parameters, self.parameter_values, self.parameter_index_sets = self.get_parameters()
        self.variables, self.variable_values = self.get_variables()
        self.equality_constraints, self.inequality_constraints = self.get_constraints()
        self.objective = self.get_objective()
        self.duals = self.generate_duals(
            self.model.component_objects(pyo.Constraint), self.model.dual
        )

        print('sympification complete')

    def get_sets(self):
        """extracts sets from self.model and converts them to Sympy Idx objects

        Returns
        -------

        sets : dict
            dictionary of set names:Idx objects
        set_values : dict
            dictionary of Idx objects:corresponding element of the Pyomo set
        """
        sets = {s.name: Idx(s.name) for s in self.model.component_objects(pyo.Set)}
        set_values = {sets[s.name]: s.data() for s in self.model.component_objects(pyo.Set)}

        return sets, set_values

    def get_parameters(self):
        """extracts parameters from self.model and converts to Sympy Symbol or IndexedBase + index objects
        depending on whether they are indexed components or not.

        Returns
        -------

        parameters : dict
            dictionary of parameter names:IndexedBase objects with that name
        parameters_values : dict
            dictionary of Sympy objects:their numeric value
        parameter_index_sets :dict
            dictionary of parameter name:list of indices for the name
        """
        parameters = {}
        parameter_values = {}
        parameter_index_sets = {}

        for p in self.model.component_objects(pyo.Param):
            if p.is_indexed():
                parameters[p.name] = IndexedBase(p.name)
                parameter_index_sets[p.name] = [index for index in p.index_set()]

                for index in p.index_set():
                    parameter_values[parameters[p.name][index]] = p.extract_values()[index]

            else:
                parameters[p.name] = Symbol(p.name)
                parameter_values[parameters[p.name]] = p.value
                parameter_index_sets[p.name] = False

        return parameters, parameter_values, parameter_index_sets

    def get_variables(self):
        """extracts variables from self.model and converts them to Sympy Symbol or IndexedBase + index objects

        Returns
        -------
        variables : dict
            dictionary of variable names:corresponding Sympy object (Symbol or IndexedBase)
        variable_values : dict
            dictionary of Sympy objects (Symbol or IndexedBase):numeric value
        """
        variables = {}
        variable_values = {}

        for v in self.model.component_objects(pyo.Var):
            if v.is_indexed():
                variables[v.name] = IndexedBase(v.name)
                for index in v.index_set():
                    variable_values[variables[v.name][index]] = v.extract_values()[index]
            else:
                variables[v.name] = Symbol(v.name)
                variable_values[variable[v.name]] = v.value

        return variables, variable_values

    def get_constraints(self):
        """extract constraint expressions from self.model and convert to Sympy expressions
        in terms of the extracted Sympy Symbols for variables and parameters

        Returns
        -------

        equality_constraints : dict
            dictionary of names and indices of equality constraints:Sympy expression for lhs of constraint
        inequality_constraints : dict
            dictionary of names and indices of inequality constraints:Sympy expression for lhs of constraint

        """
        local_dict = self.sets.copy()
        local_dict.update(self.parameters.copy())
        local_dict.update(self.variables.copy())

        equality_constraints = {}
        inequality_constraints = {}

        for c in self.model.component_objects(pyo.Constraint):
            if c.is_indexed():
                for index, expr in c.items():
                    if expr.equality:
                        equality_constraints[(c.name, index)] = parse_expr(
                            expr.body.to_string(), local_dict=local_dict
                        )
                    else:
                        inequality_constraints[(c.name, index)] = parse_expr(
                            expr.body.to_string(), local_dict=local_dict
                        )
            else:
                if c.equality():
                    equality_constraints[c.name] = parse_expr(
                        c.expr().body.to_string(), local_dict=local_dict
                    )
                else:
                    inequality_constraints[c.name] = parse_expr(
                        c.expr().body.to_string(), local_dict=local_dict
                    )

        return equality_constraints, inequality_constraints

    def get_objective(self):
        """extract objective expression from self.model

        Returns
        -------
        expr
            objective expression
        """
        local_dict = self.sets.copy()
        local_dict.update(self.parameters.copy())
        local_dict.update(self.variables.copy())

        for obj in self.model.component_objects(pyo.Objective):
            if obj.active:
                objective = parse_expr(obj.expr.to_string(), local_dict=local_dict)
        return objective

    def generate_duals(self, constraints, duals):
        """cycles through constraints, extracts duals, categorizes by complementary slackness conditions

        Parameters
        ----------
        constraints : dict
            iterator of pyomo constraints
        duals : obj
            duals object from pyomo model

        Returns
        -------
        dict
            dictionary of duals sorts by type
        """

        deletions = []
        degenerates = []
        equality_duals = {}
        inequality_duals = {}

        i = 0

        for d in constraints:
            if d.is_indexed():
                for index in d:
                    if d[index].equality:
                        equality_duals[(d.name, index)] = duals[d[index]]

                    elif d[index].slack() != 0:
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
                        # equality_duals[(d.name, index)] = duals[d[index]]

                        inequality_duals[(d.name, index)] = (
                            duals[d[index]],
                            d[index].slack(),
                            duals[d[index]] * d[index].slack() == 0
                            and not (duals[d[index]] == 0 and d[index].slack() == 0),
                        )
                        degenerates.append(((d.name, index), i))
                        i += 1
            else:
                if d.equality:
                    equality_duals[(d.name, None)] = duals[d]
                elif d.slack() != 0:
                    deletions.append(((d.name, None), i))
                    i += 1
                elif duals[d] != 0 and d.slack() == 0:
                    inequality_duals[((d.name, None), i)] = (
                        duals[d],
                        d.slack(),
                        duals[d] * d.slack() == 0 and not (duals[d] == 0 and d.slack() == 0),
                    )
                    i += 1
                else:
                    # equality_duals[(d.name)] = duals[d]
                    inequality_duals[(d.name)] = (
                        duals[d],
                        d.slack(),
                        duals[d] * d.slack() == 0 and not (duals[d] == 0 and d.slack() == 0),
                    )
                    degenerates.append(((d.name, None), i))
                    i += 1

        print('number of deletions')
        print(len(deletions))
        print('number of degenerates')
        print(len(degenerates))
        print('number of equality duals')
        print(len(equality_duals))
        print('number of inequality duals kept')
        print(len(inequality_duals))

        return {
            'deletions': deletions,
            'degenerates': degenerates,
            'equality_duals': equality_duals.copy(),
            'inequality_duals': inequality_duals.copy(),
        }

    def check_complimentarity_all(self):
        """quick check of complementarity conditions"""
        for value in self.inequality_duals.values():
            if value[0] * value[1] == 0 and not (value[0] == 0 and value[1] == 0):
                print('complimenty slack')
            elif value[0] * value[1] != 0:
                print("something's wrong")
            else:
                print('active constraint with 0 dual')

    def substitute_values(self, substitution_dict):
        """subsitutes numeric values for sympy symbols according to substitution_dict

        Parameters
        ----------
        substitution_dict : dict
            dictionary of symbols:values to substitute
        """
        for c in self.equality_constraints.keys():
            self.equality_constraints[c] = self.equality_constraints[c].xreplace(substitution_dict)
        for c in self.inequality_constraints.keys():
            self.inequality_constraints[c] = self.inequality_constraints[c].xreplace(
                substitution_dict
            )
        self.objective.xreplace(substitution_dict)

    def get_sensitivity_matrix(self, parameters_of_interest=None):
        """generates a SensitivityMatrix object based on the sympy representation, keeping parameters in parameters_of_interest
        and substituting numeric values for the rest in all expressions.

        Parameters
        ----------
        parameters_of_interest : list, optional
            list of parameters to keep symbolic. These are the parameters to evaluate sensitivity for. Defaults to None.

        Returns
        -------
        obj
            SensitivityMatrix object for the model, with parameters_of_interest considered.
        """
        print('generating sensitivity matrix')

        params = {}
        param_values = {}
        params_to_sub = {}
        params_to_sub_values = {}

        if parameters_of_interest:
            for p in self.parameters.keys():
                if p in parameters_of_interest:
                    params[p] = self.parameters[p]
                else:
                    params_to_sub[p] = self.parameters[p]

            param_values = {}

            for p in parameters_of_interest:
                if self.parameter_index_sets[p]:
                    for index in self.parameter_index_sets[p]:
                        param_values[self.parameters[p][index]] = self.parameter_values[
                            self.parameters[p][index]
                        ]
                else:
                    param_values[self.parameters[p]] = self.parameter_values[self.parameters[p]]

            params_to_sub_values = {}

            for p in params_to_sub:
                if self.parameter_index_sets[p]:
                    for index in self.parameter_index_sets[p]:
                        params_to_sub_values[self.parameters[p][index]] = self.parameter_values[
                            self.parameters[p][index]
                        ]
                else:
                    params_to_sub_values[self.parameters[p]] = self.parameter_values[
                        self.parameters[p]
                    ]

            self.substitute_values(params_to_sub_values)

        else:
            params = self.parameters.copy()
            param_values = self.parameter_values.copy()

        sympification = {
            'sets': self.sets.copy(),
            'set_values': self.set_values.copy(),
            'parameters': params.copy(),
            'parameter_values': param_values.copy(),
            'variables': self.variables.copy(),
            'variable_values': self.variable_values.copy(),
            'equality_constraints': self.equality_constraints.copy(),
            'inequality_constraints': self.inequality_constraints.copy(),
            'objective': self.objective,
            'duals': self.duals,
        }
        of_interest = {
            'parameters': params.copy(),
            'parameter_values': param_values.copy(),
            'params_to_sub': params_to_sub.copy(),
            'params_to_sub_vals': params_to_sub_values.copy(),
        }

        return SensitivityMatrix(sympification, self.duals, of_interest)


class SensitivityMatrix:
    def __init__(self, sympification, duals, parameters_of_interest):
        """transforms dictionary of sympy representations of pyomo model into submatrix components of sensitivity matrix
        and assembles them into U^-1*S if U invertible, allowing all sensitivities to be pulled,

        Parameters
        ----------

        sympification : dict
            dictionary that stores names, indices and values of variables, duals parameters, and contstraint expressions
        duals : dict
            dictionary of duals:values
        parameters_of_interest : dict
            a dictionary to keep of parameters of interest for future substitutions
        """
        self.sympification = sympification
        self.duals = duals
        self.variable_vector = list(sympification['variable_values'].keys())
        self.parameter_vector = list(parameters_of_interest['parameter_values'].keys())
        self.parameters_of_interest = parameters_of_interest

        self.components = self.generate_matrix()
        self.subs_dict = self.create_substitution_dictionary()
        self.U, self.S = self.matrix_assembly(self.components)
        self.U = self.resolve_kronecker(self.U)
        self.S = self.resolve_kronecker(self.S)

    def create_substitution_dictionary(self, values_of_interest=None):
        """creates a single dictionary with all values to be substituted into the sensitivity matrix.

        Parameters
        ----------
        values_of_interest :dict, optional
            Subset of all possible substitutions you'd like to perform. Defaults to None.

        Returns
        -------
        dict
            dictionary of substitution values
        """
        # values of interest should be a dict of parameter or variable NAMES and list of index values
        # if index values are left blank, it should default to substituting the values for all indices

        subs_dict = {}

        if values_of_interest:
            for value_name in values_of_interest.keys():
                if values_of_interest[value_name]:
                    for index in values_of_interest[value_name]:
                        if value_name in self.sympification['variables'].keys():
                            subs_dict.update(
                                {
                                    self.sympification['variables'][value_name][
                                        index
                                    ]: self.sympification['variable_values'][
                                        self.sympification['variables'][value_name][index]
                                    ]
                                }
                            )
                        elif value_name in self.sympification['parameters'].keys():
                            subs_dict.update(
                                {
                                    self.sympification['parameters'][value_name][
                                        index
                                    ]: self.sympification['parameter_values'][
                                        self.sympification['variables'][value_name][index]
                                    ]
                                }
                            )
                else:
                    if value_name in self.sympification['variables'].keys():
                        subs_dict.update(
                            {
                                self.sympification['variables'][value_name]: self.sympification[
                                    'variable_values'
                                ][self.sympification['variables'][value_name]]
                            }
                        )
                    elif value_name in self.sympification['parameters']:
                        subs_dict.update(
                            {
                                self.sympification['parameters'][value_name]: self.sympification[
                                    'parameter_values'
                                ][self.sympification['variables'][value_name]]
                            }
                        )

        else:
            subs_dict.update(self.sympification['variable_values'])
            subs_dict.update(self.sympification['parameter_values'])
            subs_dict.update(self.parameters_of_interest['params_to_sub_vals'])

        return subs_dict

    def generate_matrix(self):
        """generates the submatrix components of the sensitivity matrix that will be used to express sensitivities and extrapolate from a given solution point

        Returns
        -------
        dict
            dictionary of submatrix name:Matrix object.
        """

        print('number of variables: ', len(self.variable_vector))
        print('number of parameters: ', len(self.parameter_vector))
        print(
            'number of active, non-degenerate equality constraints: ',
            len(self.sympification['equality_constraints']),
        )
        print('number of equality duals: ', len(self.duals['equality_duals']))
        print(
            'number of inequality_constraints: ', len(self.sympification['inequality_constraints'])
        )
        print(
            'number of duals for active, non-degenerate inequality constraints: ',
            len(self.duals['inequality_duals']),
        )

        f = Matrix([self.sympification['objective']])

        start = time.time()
        h = [Matrix([h_k]) for h_k in self.sympification['equality_constraints'].values()]
        h_lambda = [dual for dual in self.duals['equality_duals'].values()]
        Hx = Matrix([h[i] for i in range(len(h))]).jacobian(self.variable_vector)
        Ha = Matrix([h[i] for i in range(len(h))]).jacobian(self.parameter_vector)
        stop = time.time()

        print('Hx, Ha obtained: ', stop - start)
        print('Hx dimensions: ', Hx.shape, ' Ha dimensions: ', Ha.shape)

        start = time.time()
        g = [
            Matrix([self.sympification['inequality_constraints'][g_j]])
            for g_j in self.duals['inequality_duals'].keys()
        ]
        g_mu = [dual[0] for dual in self.duals['inequality_duals'].values()]
        Gx = Matrix([g[i] for i in range(len(g))]).jacobian(self.variable_vector)
        Ga = Matrix([g[i] for i in range(len(g))]).jacobian(self.parameter_vector)
        stop = time.time()

        print('Gx,Ga obtained: ', stop - start)
        print('Gx dimensions: ', Gx.shape, ' Ga dimensions: ', Ga.shape)

        print('calculating Fxx,Fxa the fast way:')

        start = time.time()
        Fx = f.jacobian(self.variable_vector)
        Fa = f.jacobian(self.parameter_vector)
        stop = time.time()

        print('Fx,Fa: ', stop - start)

        start = time.time()
        hxx = [Hx.row(i).jacobian(self.variable_vector).transpose() for i in range(Hx.rows)]
        hxa = [Ha.row(i).jacobian(self.variable_vector).transpose() for i in range(Ha.rows)]
        gxx = [Gx.row(i).jacobian(self.variable_vector).transpose() for i in range(Gx.rows)]
        gxa = [Ga.row(i).jacobian(self.variable_vector).transpose() for i in range(Ga.rows)]
        stop = time.time()

        print('Hxx,Hxa,Gxx,Gxa: ', stop - start)

        start = time.time()
        fxx = Fx.jacobian(self.variable_vector).transpose()
        fxa = Fa.jacobian(self.variable_vector).transpose()
        stop = time.time()

        print('fxx,fxa: ', stop - start)

        start = time.time()
        Fxx = (
            fxx
            + sum([h_lambda[i] * hxx[i] for i in range(len(h_lambda))], zeros(fxx.rows, fxx.cols))
            + sum([g_mu[i] * gxx[i] for i in range(len(g_mu))], zeros(fxx.rows, fxx.cols))
        )
        Fxa = (
            fxa
            + sum([h_lambda[i] * hxa[i] for i in range(len(h_lambda))], zeros(fxa.rows, fxa.cols))
            + sum([g_mu[i] * gxa[i] for i in range(len(g_mu))], zeros(fxa.rows, fxa.cols))
        )
        stop = time.time()

        print('Fxx,Fxa obtained. time: ', stop - start)
        print('Fxx dimensions: ', Fxx.shape, ' Fxa dimensions: ', Fxa.shape)

        matrix_components = {
            'Fx': Fx,
            'Fa': Fa,
            'Fxx': Fxx,
            'Fxa': Fxa,
            'Hx': Hx,
            'Ha': Ha,
            'h_lambda': h_lambda,
            'Gx': Gx,
            'Ga': Ga,
            'g_mu': g_mu,
        }

        return matrix_components

    def matrix_assembly(self, components):
        """assemble the submatrix components into U,S

        Parameters
        ----------
        components : dict
            dictionary of submatrix names:Matrix objects

        Returns
        -------
        obj
            Matrix object U
        obj
            Matrix object S
        """
        print('assembling matrix')

        U1 = Matrix.vstack(components['Fx'], components['Fxx'], components['Hx'], components['Gx'])
        U2 = Matrix.vstack(
            Matrix.zeros(1, components['Hx'].transpose().cols),
            components['Hx'].transpose(),
            Matrix.zeros(components['Hx'].rows),
            Matrix.zeros(components['Gx'].rows, components['Hx'].transpose().cols),
        )
        U3 = Matrix.vstack(
            Matrix.zeros(1, components['Gx'].transpose().cols),
            components['Gx'].transpose(),
            Matrix.zeros(components['Hx'].rows, components['Gx'].transpose().cols),
            Matrix.zeros(components['Gx'].rows),
        )
        U4 = Matrix.vstack(
            Matrix.ones(1),
            Matrix.zeros(components['Fxx'].rows + components['Hx'].rows + components['Gx'].rows, 1),
        )

        print('U block dimensions:')
        print(U1.shape, U2.shape, U3.shape, U4.shape)

        U = Matrix.hstack(U1, U2, U3, U4)

        print('final U matrix dimensions: ', U.shape)
        print(
            components['Fa'].shape,
            components['Fxa'].shape,
            components['Ha'].shape,
            components['Ga'].shape,
        )

        S = Matrix.vstack(components['Fa'], components['Fxa'], components['Ha'], components['Ga'])
        print('dimensions of S: ', S.shape)

        return U, S

    def substitute_values(self, values_dict=None):
        """substitute values into self.U and self.S according to a given substitution dictionary, or a substition dictionary for all values

        Parameters
        ----------
        values_dict : dict, optional
            dictionary of symbols:values

        Returns
        -------
        obj
            the stored Matrix object U after substituting numeric values
        obj
            the stored matrix S after substituting numeric values
        """
        if values_dict:
            return self.U.xreplace(values_dict), self.S.xreplace(values_dict)
        else:
            return self.U.xreplace(self.subs_dict), self.S.xreplace(self.subs_dict)

    def get_sensitivities(self):
        """creates a set of dictionaries to keep track of the relationships between symbol names, symbol objects,
        and their position in the respective vectors of parameters and variable quantities. Calculates U^-1*S and Creates a DifferentialMapping object.
        Sensitivities of quantities with respect to parameters will be corresponding entries in the matrix generated,
        which can be queried directly in the DifferentialMapping object

        Returns
        -------
        obj
            an DifferentialMapping object that stores the sensitivity matrix and dictionaries matching coordinates to symbols
        """
        start = time.time()
        U, S = self.substitute_values()
        stop = time.time()

        print('substitute values into U,S: ', stop - start)

        start = time.time()
        coord2item = {}
        item2coord = {}
        for i in range(len(self.variable_vector)):
            coord2item[i] = self.variable_vector[i]
            item2coord[self.variable_vector[i]] = i
        for i in range(len(self.sympification['duals']['equality_duals'].keys())):
            coord2item[len(coord2item.keys())] = list(
                self.sympification['duals']['equality_duals'].keys()
            )[i]
            item2coord[list(self.sympification['duals']['equality_duals'].keys())[i]] = len(
                item2coord.keys()
            )

        for i in range(len(self.sympification['duals']['inequality_duals'].keys())):
            coord2item[len(coord2item.keys())] = list(
                self.sympification['duals']['inequality_duals'].keys()
            )[i]
            item2coord[list(self.sympification['duals']['inequality_duals'].keys())[i]] = len(
                item2coord.keys()
            )

        coord2item.update({len(coord2item): 'dz'})
        item2coord.update({'dz': len(item2coord)})
        param2coord = {self.parameter_vector[i]: i for i in range(len(self.parameter_vector))}
        coord2param = {i: self.parameter_vector[i] for i in range(len(self.parameter_vector))}
        stop = time.time()

        print('creating bimap dictionaries between row, col values and symbols: ', stop - start)

        start = time.time()
        US = U.inv() * S * eye(len(self.parameter_vector))
        stop = time.time()

        print('inverting U matrix and multiplying: ', stop - start)

        return DifferentialMapping(US, coord2item, item2coord, param2coord, coord2param)

    def resolve_kronecker(self, expr):
        """evaluates KroneckerDelta terms

        Parameters
        ----------
        obj
            sympy expression

        Returns
        -------
        obj
            sympy expr with KroneckerDelta terms evaluated to 0 or 1
        """
        return expr.replace(KroneckerDelta, lambda a, b: 1 if a == b else 0)


class CoordMap:
    def __init__(self, var_vector, eq_duals, ineq_duals, params):
        """An object that can keep track of all the symbol types, and their coordinates all at once.
            the types in the list is open and could be names or could be symbol objects. It's left open.

        Parameters
        ----------
        var_vector : list
            list of variables
        eq_duals : list
            list of equality duals
        ineq_duals : list
            list of inequality duals
        params : list
            list of parameters
        """
        self.coord2item = {}
        self.item2coord = {}

        for i in range(len(var_vector)):
            coord2item[i] = var_vector[i]
            item2coord[var_vector[i]] = i
        for i in range(len(eq_duals)):
            coord2item[len(coord2item.keys())] = eq_duals[i]
            item2coord[eq_duals[i]] = len(item2coord.keys())
        for i in range(len(ineq_duals)):
            coord2item[len(coord2item.keys())] = ineq_duals[i]
            item2coord[ineq_duals[i]] = len(item2coord.keys())
        coord2item.update({len(coord2item): 'dz'})
        item2coord.update({'dz': len(item2coord)})

        self.param2coord = {params[i]: i for i in range(len(params))}
        self.coord2param = {i: params[i] for i in range(len(params))}


class DifferentialMapping:
    def __init__(self, US, coord2item, item2coord, param2coord, coord2param):
        """class that holds the sensitivity matrix in a ready-to-use form, along with dictionaries that map
            the correspondence between the entries' coordinates and the corresponding symbol or value. Can query
            for sensitivities or perform an extrapolation.

        Parameters
        ----------

        US : Matrix
            Matrix of sensitivities
        coord2item : dict
            dictionary of coordinates:items (symbols, variable names, etc)
        item2coord : dict
            dictionary of items (variable, dual, objective):coordinates
        param2coord : dict
            dictionary of parameters:coordinates
        coord2param : dict
            dictionary of coordinates:parameters
        """
        self.US = US
        self.coord2item = coord2item
        self.item2coord = item2coord
        self.param2coord = param2coord
        self.coord2param = coord2param

    def sensitivity(self, item, parameter):
        """picks out the sensitivity of item with respect to parameters

        Parameters
        ----------
        item : symbol, str
            however the particular item was stored, usually symbol
        parameter : symbol, str
            however the particular parameter was stored, usually symbol

        Returns
        -------
        float
            the sensitivity of item with respect to parameter.
        """
        return self.US.row(self.item2coord[item]).col(self.param2coord[parameter])

    def extrapolate(self, current_values, param_delta):
        """given the current value of all varying quantities (variables, duals, objective) as a vector,
           and a delta of parameter values as a vector, calculates the effect of perturbing the parameters
           by the parameter delta.

        Parameters
        ----------

        current_values : Matrix
            vector of [variables values, dual values, objective value]
        param_delta : Matrix
            vector of [parameter values]

        Returns
        -------
        item_delta : obj
            column Matrix of items
        item_delta_map : dict
            map between items and their changes
        item_new_value : Matrix
            column Matrix of new values after change
        item_new_value_map : dict
            map between items and their new values
        """
        item_delta = self.US * param_delta
        item_delta_map = {}
        item_new_value = current_values + item_delta
        item_new_value_map = {}
        for i in range(len(self.coord2item.keys())):
            item_delta_map[self.coord2item[i]] = item_delta.row(i).col(0)
            item_new_value_map[self.coord2item[i]] = item_new_value.row(i).col(0)

        return item_delta, item_delta_map, item_new_value, item_new_value_map
