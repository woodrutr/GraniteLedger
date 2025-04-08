# src.common.model

Establish a base model class for the sectoral modules to inherit.

### Functions

| `getLogger`([name])   | Return a logger with the specified name, creating it if necessary.   |
|-----------------------|----------------------------------------------------------------------|

### Classes

| [`Model`](#src.common.model.Model)(\*args, \*\*kwds)   | This is the base model class for the models.                              |
|--------------------------------------------------------|---------------------------------------------------------------------------|
| `defaultdict`                                          | defaultdict(default_factory=None, /, [...]) --> dict with default factory |

### *class* src.common.model.Model(\*args, \*\*kwds)

This is the base model class for the models.

This class contains methods for declaring pyomo components, extracting duals, and
decorating expressions. The model class methods and attributes provide functionality
for keeping track of index labels and ordering for all pyomo components; this is
essential for integration tasks without the use of hard-coded indices and allows for
easy post-processing tasks.

#### *class* ConstraintExpression(\*args, \*\*kwargs)

Constraint Expression decorator that works the same as pyomo decorators, while keeping
column dictionary updated for any indexed parameters given.

#### *class* DefaultDecorator(model, \*args, \*\*kwargs)

Default decorator class that handles assignment of model scope/pointer in order to use
pyomo-style parameter and constraint decorators.

Upon initialization, the decorator handles model assignment at class level to ensure
inheriting classes have access to the models within local scope.

#### *classmethod* assign_model(model)

Class-method that assigns a model instance to DefaultDecorator

* **Parameters:**
  **model** (*pyo.ConcreteModel*) – A pyo model instance

#### *class* ParameterExpression(\*args, \*\*kwargs)

Parameter Expression decorator that works the same as pyomo decorators, while keeping
column dictionary updated for any indexed parameters given.

#### \_active

#### \_declare_set_with_dict(sname: str, sdata: Dict | DefaultDict, scols: MutableSequence, return_set: bool | None = False, switch: bool | None = True, create_indexed_set: bool | None = True, use_values: bool | None = False) → Set

Declares a pyomo Set object named ‘sname’ using input index values and labels.

Function takes a dictionary argument and creates pyomo set object from keys, values, or both.

If an indexed set is desired, set create_indexed_set to True; the function will create an
indexed set with its own indices set as keys. Otherwise, an Ordered Scalar Set will be
created, either from the keys or the values of ‘sdata’ depending on the value for
‘use_values’ (False for keys, True for values).

Names for the indices handled by scols; user must provide.

* **Parameters:**
  * **sname** (*str*) – Name of set
  * **sdata** (*Dict*) – Data object that contains set values
  * **scols** (*Sequence* *|* *None* *,* *optional*) – List of column names corresponding to index labels and position, by default None
  * **return_set** (*bool* *|* *None* *,* *optional*) – Return the set rather than assign within function, by default False
  * **switch** (*bool* *|* *None* *,* *optional*) – Return None if False, by default True
  * **create_indexed_set** (*bool* *|* *None* *,* *optional*) – Indicator for whether output set should include values as well as new index (IndexedSets), by default True
  * **use_values** (*bool* *|* *None* *,* *optional*) – If create_indexed_set is False, use the values of sdata rather than keys for pyo Set members, by default False
* **Returns:**
  Pyomo Set Object
* **Return type:**
  pyo.Set

#### \_declare_set_with_iterable(sname: str, sdata: Sequence | Set | array, scols: Sequence[str] | None = None, return_set: bool | None = False, switch: bool | None = True) → Set

Declares a pyomo Set object named ‘sname’ using input index values and labels.

Function can take iterable objects such as tuples, lists, etc as data inputs. Note that if
the dimension of the index is larger than 1, user needs to provide a list of names for each
set dimension.

* **Parameters:**
  * **sname** (*str*) – Name of set
  * **sdata** (*Sequence*) – Data object that contains set values
  * **scols** (*Sequence* *|* *None* *,* *optional*) – List of column names corresponding to index labels and position, by default None
  * **return_set** (*bool* *|* *None* *,* *optional*) – Return the set rather than assign within function, by default False
  * **switch** (*bool* *|* *None* *,* *optional*) – Return None if False, by default True
* **Returns:**
  Pyomo Set Object
* **Return type:**
  pyo.Set

#### \_declare_set_with_pandas(sname: str, sdata: DataFrame | Series, return_set: bool | None = False, switch: bool | None = True, use_columns: bool | None = False)

Declares a pyomo Set object named ‘sname’ using input index values and labels from a
Pandas object.

Function assumes that the index values are the desired data to construct set object. User
can specify whether to create set with column values instead

* **Parameters:**
  * **sname** (*str*) – Name of set
  * **sdata** (*MutableSequence* *|* *dict*) – Data object that contains set indices
  * **return_set** (*bool* *|* *None* *,* *optional*) – Return the set rather than assign within function, by default False
  * **switch** (*bool* *|* *None* *,* *optional*) – Return None if False, by default True
  * **use_columns** (*bool* *|* *None* *,* *optional*) – Use columns as indices for pyo set rather than row index, by default False
* **Returns:**
  Pyomo Set Object
* **Return type:**
  pyo.Set

#### *classmethod* build()

Default build command; class-level build to create and return an instance of Model.

This will work for any class inheriting the method, but it is recommended to replace this
with model-specific build instructions if this functionality is desired.

* **Returns:**
  Instance of Model object
* **Return type:**
  Object

#### declare_ordered_time_set(sname: str, \*sets: Set, return_set: bool | None = False)

Unnest the time sets into a single, unnested ordered, synchronous time set, an IndexedSet
object keyed by the values in the time set, and an IndexedSet object keyed by the combined,
original input sets.

These three set outputs are directly set as attributes of the model instance:

sname:               (1,) , (2, ), … ,(N)
sname_time_to_index: (1,):[set1, set2, set3] , (2,):[set1, set2, set3]
sname_index_to_time: (set1, set2, set3): [1] , (set1, set2, set3): [2]

In summary, this function creates three sets, creating a unique, ordered set from input sets
with the assumption that they are given to the function in hierarchical order. For example,
for a desired time set that orders Year, Month, Day values, the args for the function
should be provided as:

m.Year, m.Month, m.Day

Pyomo set products are used to unpack and create new set values that are ordered by the
hierarchy provided:

(year1, month1, day1) , (year1, month1, day2) , … , (year2, month1, day1) , … (yearY, monthM, dayD)

* **Parameters:**
  * **sname** (*str*) – Desired root name for the new sets
  * **sets** (*pyo.Set*) – A series of unnamed arguments assumed to contain pyo.Set in order of temporal hierarchy
  * **return_set** (*bool* *|* *None* *,* *optional*) – Return the set rather than assign within function, by default False
* **Returns:**
  No return object; all sets assigned to model internally
* **Return type:**
  None
* **Raises:**
  **ValueError** – “No sets provided in args; provide pyo.Set objects to use this function”

#### declare_param(pname: str, p_set: Set | None, data: dict | DataFrame | Series | int | float, return_param: bool | None = False, default: int | None = 0, mutable: bool | None = False) → Param

Declares a pyo Parameter component named ‘pname’ with the input data and index set.

Unpacks column dictionary of index set for param instance and creates pyo.Param; either
assigns the value internally or returns the object based on return_param.

* **Parameters:**
  * **pname** (*str*) – Desired name of new pyo.Param instance
  * **p_set** (*pyo.Set*) – Pyomo Set instance to index new Param
  * **data** (*dict* *|* *pd.DataFrame* *|* *pd.Series*) – Data to initialize Param instance
  * **return_param** (*bool* *|* *None* *,* *optional*) – Return the param after function call rather than assign to self, by default False
  * **default** (*int* *|* *None* *,* *optional*) – pyo.Param keyword argument, by default 0
  * **mutable** (*bool* *|* *None* *,* *optional*) – pyo.Param keyword argument, by default False
* **Returns:**
  A pyomo Parameter instance
* **Return type:**
  pyo.Param
* **Raises:**
  **ValueError** – Raises error if input data not in format supported by function

#### declare_set(sname: str, sdata: MutableSequence | DataFrame | Series | Dict, scols: MutableSequence | None = None, return_set: bool | None = False, switch: bool | None = True, create_indexed_set: bool | None = True, use_values: bool | None = False, use_columns: bool | None = False)

Declares a pyomo Set object named ‘sname’ using input index values and labels.

Function handles input values and calls appropriate declare_set methods based on data
type of sdata

* **Parameters:**
  * **sname** (*str*) – Name of set
  * **sdata** (*Dict*) – Data object that contains set values
  * **scols** (*Sequence* *|* *None* *,* *optional*) – List of column names corresponding to index labels and position, by default None
  * **return_set** (*bool* *|* *None* *,* *optional*) – Return the set rather than assign within function, by default False
  * **switch** (*bool* *|* *None* *,* *optional*) – Return None if False, by default True
  * **create_indexed_set** (*bool* *|* *None* *,* *optional*) – If dict, indicator for whether output set should include values as well as new index (IndexedSets), by default True
  * **use_values** (*bool* *|* *None* *,* *optional*) – If dict and create_indexed_set is False, use the values of sdata rather than keys for pyo Set members, by default False
  * **use_columns** (*bool* *|* *None* *,* *optional*) – If Pandas, use columns as indices for pyo set rather than row index, by default False
* **Returns:**
  Pyomo Set Object
* **Return type:**
  pyo.Set

#### declare_set_with_sets(sname: str, \*sets: Set, return_set: bool | None = False, switch: bool | None = True) → Set

Declares a new set object using input sets as arguments.

Function creates a set product with set arguments to create a new set. This is how pyomo
handles set creation with multiple existing sets as arguments.

However, this function finds each pyomo set in column dictionary and unpacks the names,
so that the new set can be logged in the column dictionary too.

* **Parameters:**
  * **sname** (*str*) – Desired name of new set
  * **\*sets** (*tuple* *of* *pyo.Set*) – Unnamed arguments assumed to be pyomo sets
  * **return_set** (*bool* *|* *None* *,* *optional*) – Return the set rather than assign within function, by default False
  * **switch** (*bool* *|* *None* *,* *optional*) – Return None if False, by default True
* **Returns:**
  Pyomo Set Object
* **Return type:**
  pyo.Set

#### declare_shifted_time_set(sname: str, shift_size: int, shift_type: Literal['lag', 'lead'], \*sets: Set, return_set: bool | None = False, shift_sets: List | None = None)

A generalize shifting function that creates sets compatible with leads or lags in pyomo
components.

For example, with a storage constraint where the current value is contrained to be equal to
the value of storage in the previous period:

model.storage[t] == model.storage[t-1] + …

The indexing set must be consistent with the storage variable, but not include elements that
are undefined for this constraint. In this example, the set containing values for t must not
include t = 1 (e.g. the lagged value must be defined). This function creates a shifted time
set by removing values from the input sets to comply with the lags or leads.

Function inputs require a shift size (in the example above, this would be 1), a shift type
(lead or lag), and the sets used to construct the new, shifted set (model.timestep). If a
lag or lead is required on a single dimension of the new set, the ‘shift_sets’ argument can
include a list of pyo.set names (included in the arguments) to shift by the other args.

For example…

model.storage[hub, season] == model.storage[hub, season - 1]

In this case, season = 1 is always invalid due to the lag; so index (1, 2) or the value for
hub = 1 and season = 2 is valid, but (2, 1) remains an invalid argument as there is no
season = 0. A new set composed of hub and season, with shift_sets = [“season”] and
sets = model.hub, model.season, is created to lag on one index value while leaving others
unchanged.

Default is to create set product of all input sets and lag/lead w/ resulting elements.

* **Parameters:**
  * **sname** (*str*) – Desired name for new set
  * **shift_size** (*int*) – Size of shift in set
  * **shift_type** (*str in* *[* *"lag"* *,*  *"lead"* *]*) – Type of shift (e.g. t-1 or t+1)
  * **\*sets** (*Unnamed arguments*) – A series of unnamed arguments assumed to contain pyo.Set in order of temporal hierarchy
  * **return_set** (*bool* *|* *None* *,* *optional*) – Return the set rather than assign within function, by default False
  * **shift_sets** (*List* *|* *None* *,* *optional*) – List of pyo.Set (by name) in 

    ```
    *
    ```

    sets to shift, by default None
* **Returns:**
  A pyomo Set
* **Return type:**
  pyo.Set
* **Raises:**
  * **ValueError** – Shift sets don’t align with 

    ```
    *
    ```

    sets names
  * **ValueError** – Type argument is neither lead nor lag

#### declare_var(vname: str, v_set: Set, return_var: bool | None = False, within: Literal['NonNegativeReals', 'Binary', 'Reals', 'NonNegativeIntegers'] | None = 'NonNegativeReals', bound: tuple | None = (0, 1000000000), switch: bool | None = True) → Var

Declares a pyo Variable component named ‘vname’ with index set ‘v_set’.

Creates variable indexed by previously defined pyo Set instance ‘v_set’ and assigns to self;
function will return the component if return_var is set to True. Other keywords passed to
pyo.Var are within and bound.

* **Parameters:**
  * **vname** (*str*) – Desired name of new pyo Variable
  * **v_set** (*pyo.Set*) – Index set for new pyo Variable
  * **return_var** (*bool* *|* *None* *,* *optional*) – Return component rather than assign internally, by default False
  * **within** (*str in* *[* *"NonNegativeReals"* *,*  *"Binary"* *,*  *"Reals"* *,*  *"NonNegativeIntegers"* *]*  *|* *None* *,* *optional*) – pyo.Var keyword argument, by default “NonNegativeReals”
  * **bound** (*tuple* *|* *None* *,* *optional*) – pyo.Var keyword argument, by default (0, 1000000000)
* **Returns:**
  A pyomo Variable instance
* **Return type:**
  pyo.Var

#### get_duals(component_name: str) → defaultdict

Extract duals from a solved model instance

* **Parameters:**
  **component_name** (*str*) – Name of constraint
* **Returns:**
  Dual values w/ index values
* **Return type:**
  defaultdict

#### populate_sets_rule(sname, set_base_name=None, set_base2=None) → Set

Generic function to create a new re-indexed set for a pyomo ConcreteModel instance which
should speed up build time. Must pass non-empty (either) set_base_name or set_base2

* **Parameters:**
  * **m1** (*pyo.ConcreteModel*) – pyomo model instance
  * **sname** (*str*) – name of input pyomo set to base reindexing
  * **set_base_name** (*str* *,* *optional*) – the name of the set to be the base of the reindexing, if left blank, uses set_base2, by default ‘’
  * **set_base2** (*list* *,* *optional*) – the list of names of set columns to be the base of the reindexing, if left blank, should
    use set_base_name, by default [] these will form the index set of the indexed set structure
* **Returns:**
  reindexed set to be added to model
* **Return type:**
  pyomo set

#### reorganize_index_set(sname: str, new_sname: str, return_set: bool | None = False, create_indexed_set: bool | None = False, reorg_set_cols: List[str] | None = None, reorg_set_sname: str | None = None)

Creates new pyomo sets based on an input set and a desired set of indices for an output
set. User should provide either names of columns desired for reorganized output set OR the
name of a set that mirrors the desired indexing.

For instance, an input set indexed by (yr, region, month, day) can be reorganized into an
output set:

(yr, region):[(month,day), (month,day), (month,day)]

when [“yr”, “region”] is provided for reorg_set_cols.

If only the set keys are desired, without creating an indexed set object as illustrated
above, the user can set ‘create_indexed_set’ to false. If true, the output is a
pyo.IndexedSet, with each element of the IndexedSet containing the values of other indices

* **Parameters:**
  * **sname** (*str*) – Name of input set
  * **new_sname** (*str*) – Name of output set or IndexedSet
  * **create_indexed_set** (*bool* *|* *None* *,* *optional*) – Indicator for whether output set should include values as well as new index (IndexedSets), by default False
  * **return_set** (*bool* *|* *None* *,* *optional*) – Indicator for whether to return the constructed set
  * **reorg_set_cols** (*List* *[**str* *]*  *|* *None* *,* *optional*) – List of columns to index output set contained in ‘sname’, by default None
  * **reorg_set_sname** (*str* *|* *None* *,* *optional*) – Name of set to use for identifying output set indices, by default None
* **Returns:**
  Pyomo Set or IndexedSet object reorganized based on input set
* **Return type:**
  Pyo.Set
* **Raises:**
  * **ValueError** – Populate function is either-or for reorg_set_cols and reorg_set_sname, received both
  * **ValueError** – Populate function is either-or for reorg_set_cols and reorg_set_sname, received neither
  * **ValueError** – Elements missing from input set desired in new set

#### unpack_set_arguments(sname: str, sets: Tuple[Set], return_set_product: bool | None = True) → Set

Handles unnamed pyo.Set arguments for multiple declaration functions.

For an arbitrarily large number of set inputs, this function unpacks the names for each set
stored in the column dictionary, creates a new list of the index labels and ordering, and
then provides the pyo.Set product result as an output.

* **Parameters:**
  * **sname** (*str*) – Name of new set
  * **sets** (*tuple* *of* *pyo.Set*) – Tuple of pyo.Set arguments to be used to generate new set
  * **return_set_product** (*bool*) – If True, return the unpacked set product
* **Returns:**
  **new_set** – Set product result from input sets, by order of sets arguments
* **Return type:**
  pyo.Set
