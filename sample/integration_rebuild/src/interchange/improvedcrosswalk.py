# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:14:15 2025

@author: JNI
"""

import pandas as pd
import sympy as sp

    
class ToyModel:
    
    def __init__(self, name):

        self.name = name
        self.params = {}
        
    def load_param(self, name, data, indices, dimensions):
        
        self.params[name] = {'data':data,'indices':indices,'dimensions':dimensions}
        
    def export_param(self, param, series = True):
        
        if series:
            return ExchangeSeries(self.params[param]['data'].copy(),self.params[param]['indices'].copy(),self.params[param]['dimensions'].copy())
        else:
            return {'indices':self.params[param]['indices'].copy(),'dimensions':self.params[param]['dimensions'].copy()}
    
    def import_param(self,param, series):
        
        self.params[param] = {'data': series.data, 'indices': series.indices, 'dimensions': series.dimensions}
        
        
   

class DimensionData:
    
    def __init__(self, name: str, base: str, crosswalk:pd.DataFrame):
        """_summary_

        Parameters
        ----------
        name : str
            _description_
        base : str
            _description_
        crosswalk : pd.DataFrame
            _description_
        """          
        self.name = name
        self.base = base
        self.crosswalk = crosswalk
        
            

class ExchangeSeries:
    
    def __init__(self, data: dict, indices:dict, dimensions:dict, years = None):
        """Stores an indexed series of values representing an exchange 
        parameter, along with index labels and dimensional quantity labels

        Parameters
        ----------
        data : dict
            the raw data series in the form dict[tuple:value]
            ex: {(1,1): 2.4, (1,2):1.9, ...}
            
        indices : dict
            mapping of index positions to index name in form dict[int:str]
            ex: {0:'day', 1: 'year', 2: 'region'}
        dimensions : dict
            mapping of dimensions to the index position they correspond to in form dict[str:int]
            ex: {'time': 0, 'space': 2}
            
        years : list, optional
            list of years, by default None
        """        
        self.data = data # { tuple of indices : value}
        self.indices = indices # { index position : index name}
        self.dimensions = dimensions # {name of dimension : index that has it}
        self.index_names = {self.indices[key]:key for key in self.indices.keys() } # {index name : index position}
    
    def unit_from_dim(self, dim:str):
        """get the unit name for a given dimension

        Parameters
        ----------
        dim : str
            name of dimension

        Returns
        -------
        str
            name of unit for that dimension
        """        
        return self.indices[self.dimensions[dim]]
        
    def present_dim(self, dimension:str):
        """reorganizes the data to present the values by the dimension index for each of 
        the other indices 

        Parameters
        ----------
        dimension : str
            name of dimension to present

        Returns
        -------
        dict
            series in the form dict[tuple:dict[index:value]]
            ex: 
            data of the form: 
            {(1,1):5.1, (1,2):3.5, (2,1):3.2, (2,2):4.0} 
            presenting the first index becomes the return value:
            {(1,): {1: 5.1, 2: 3.2}, (2,): {1: 3.5, 2: 4.0}}
        """        
        new_data = {tuple(index[i] for i in range(len(index)) if i != self.dimensions[dimension]): {} \
                    for index in self.data.keys()}
        
        for index in self.data.keys():
            
            new_index = tuple(index[i] for i in range(len(index)) if i != self.dimensions[dimension])
            dim_index = index[self.dimensions[dimension]]
            new_data[new_index][dim_index] = self.data[index]
            
        return new_data

    def rebuild_series(self,dimension:str, series:dict, inplace = True):
        """takes series data that has been reorganized to present a given index
        and rebuilds it in the standard format by inserting the presented dimensional 
        index back into the original index tuple

        Parameters
        ----------
        dimension : str
            name of dimension to reinsert
        series : dict
            the raw series data
        inplace : bool, optional
            add the return series data back to self.data, by default True

        Returns
        -------
        dict
            standard ExchangeSeries format of dict[tuple:value]
        """        
        new_series = {}
        for i in series.keys():

            for j in series[i].keys():
                

                temp = list(i)

                temp.insert(self.dimensions[dimension], j)

                new_index = tuple(temp)
                new_series[new_index] = series[i][j]

            
        return new_series

    def reset(self,new_data: dict, dim:str, new_unit:str):   
        """resets the ExchangeSeries data field and dimensional unit

        Parameters
        ----------
        new_data : dict
            new series data in standard form
        dim : str
            name of dimension to change
        new_unit : str
            the name of the new unit for that dimension
        """        
        self.data = new_data
        self.indices[self.dimensions[dim]] = new_unit

    def aggregate_to_dimensions(self, dimensions):
        """_summary_

        Parameters
        ----------
        dimensions : _type_
            _description_
        """        
        
        pass
        
    def reorg(self,new_indices, new_dims):

        permute = {}
        for dim in new_dims.keys():
            permute[new_dims[dim]] = self.dimensions[dim]
        for index in new_indices:
            if index not in permute.keys():
                permute[index] = self.index_names[new_indices[index]]
        
        if all(key == permute[key] for key in permute.keys()): 
            pass
        else:
            new_data = {tuple([tuple_index[permute[index]] for index in range(len(tuple_index))]):self.data[tuple_index] for tuple_index in self.data.keys()}
            self.data = new_data
            self.indices = new_indices
            self.dimensions = new_dims
            


class Interchange:
    
    def __init__(self, dimensions = None, units = None):
        """Handles exchanging parameters between models and converting ExchangeSeries

        Parameters
        ----------
        dimensions : list
            list of dimension objects
        units : list, optional
            list of units to load, by default None
        """        
        self.dimensions = {}
        self.models = {}
        
        if units:
            pass
        elif dimensions:            
            for dim in dimensions:
                self.add_dimension(dim)
       
    def add_dimension(self, dimension):
        
        self.dimensions[dimension.name] = dimension
        
    def add_model(self,model: ToyModel):
        """adds a model to the Interchange

        Parameters
        ----------
        model : Model
            Model object
        """        
        self.models[model.name] = model
        
    
    
    def convert_series(self, series: ExchangeSeries, dimension:str, destination_unit:str):
        """convert the data of an ExchangeSeries object along a dimension to a new unit

        Parameters
        ----------
        series : ExchangeSeries
            ExchangeSeries object to convert
        dimension : str
            name of dimension to convert
        destination_unit : str
            name of unit you are converting to

        Returns
        -------
        dict
            series data organized to present the given dimension
        """        
        s = series.present_dim(dimension)
        unit = series.unit_from_dim(dimension)
        
        self.dimensions[dimension].add_conversion_weights(unit, destination_unit)
        
        for key in s.keys():
            
            d = self.dimensions[dimension]
            s[key] = d.convert_series(d.units[unit],d.units[destination_unit], s[key])
                    
        return s
    
    def change_series(self, series: ExchangeSeries, dimension:str, destination_unit:str, inplace = True):
        """convert ExchangeSeries data along a given dimension to a new unit and change it

        Parameters
        ----------
        series : ExchangeSeries
            the ExchangeSeries object to convert
        dimension : str
            name of dimension being converted
        destination_unit : str
            name of the unit converted to
        inplace : bool, optional
            change the series itself if True, by default True
        """        
        if inplace: 
            series.reset(series.rebuild_series(dimension, self.convert_series(series, dimension, destination_unit)), dimension, destination_unit)
        else:
            return ExchangeSeries(series.reset(series.rebuild_series(dimension, self.convert_series(series, dimension, destination_unit)), dimension, destination_unit))
        
        
    def exchange(self, param,source_model = None,dest_model = None, series = None, new_units = None, method = 'aggregate'):
        
        if source_model and dest_model:
            series = self.models[source_model].export_param(param)
            new_units = self.models[dest_model].export_param(param,series=False)
            series.reorg(new_units['indices'],new_units['dimensions'])
            for dim in new_units['dimensions'].keys():
                self.change_series(series,dim,new_units['indices'][new_units['dimensions'][dim]])
            self.models[dest_model].import_param(param,series)

        elif series and new_units:
            series.reorg(new_units['indices'],new_units['dimensions'])
            for dim in new_units['dimensions'].keys():
                self.change_series(series,dim,new_units['indices'][new_units['dimensions'][dim]])
        
        else: 
            raise ValueError("need either source and destination model or ExchangeSeries object and destination format")
            

    def fake_exchange(self,source_model,dest_model, param):
        
        s = self.models[source_model].export_param(param) # exchange series for from_component
        new_units = self.models[dest_model].export_param(param,series=False) # exchange series for to_compoment
        for dim in new_units['dimensions'].keys(): # 
            self.change_series(s,dim,new_units['indices'][new_units['dimensions'][dim]]) # get info from exchange series for new_units (e.g. to component)
        self.models[dest_model].import_param(param,s) # update values in the destination module (direct call to the method in module in this case)
    
    
    
class UnitDimension:
    
    def __init__(self, name:str, crosswalk:pd.DataFrame, baseunit:str, all_units = True):
        """Stores data and methods for a single dimension and the units it contains

        Parameters
        ----------
        name : str
            name of dimension
        crosswalk : pd.DataFrame
            crosswalk dataframe with units as column names and indices in the rows
        baseunit : str
            name of base unit
        """        
        self.name = name
        self.crosswalk = crosswalk
        self.base = baseunit
        self.baseunits = self.crosswalk[baseunit].unique()
        self.units = {}
        self.conversionweights = {}
        
        if all_units:
            for unit in self.crosswalk.columns:
                if unit != self.base:
                    self.add_unit(unit)
        
    def add_unit(self, unitname):
        """add a new unit to the dimension from the crosswalk

        Parameters
        ----------
        unitname : str
            name of unit
        """        
        self.units[unitname] = Unit(self.crosswalk, unitname, self.base)
        
        
    def add_conversion_weights(self, source_unit:str, destination_unit:str):
        """creates a set of conversion weights between source_unit and destination_unit 
        to allow faster calculations, and adds them to self.conversionweights
        
        This is a dict of form: ...

        Parameters
        ----------
        source_unit : str
            name of source unit
        destination_unit : str
            name of destination unit
        """        
        if type(source_unit) == str:
            source_unit = self.units[source_unit]
        if type(destination_unit) == str:
            destination_unit = self.units[destination_unit]
        
        if (source_unit,destination_unit) not in self.conversionweights.keys():
            unit2unit_weights = {dest_Idx: [source_unit.baseweights[base_Idx] for base_Idx in destination_unit.self2base[dest_Idx]] for dest_Idx in destination_unit.units}
            self.conversionweights[(source_unit.name,destination_unit.name)] = unit2unit_weights

    def convert_series(self,source_unit: str, destination_unit: str, series: dict):
        """convert a series of values indexed by source_unit to one indexed by 
        destination_unit

        Parameters
        ----------
        source_unit : str
            name of source unit
        destination_unit : str
            name of destination unit
        series : dict
            dictionary of values to convert in the form: {index:value}

        Returns
        -------
        dict
            dictionary of values after conversion in the form: {index:value}
        """

        if (source_unit.name,destination_unit.name) not in self.conversionweights.keys(): 
            self.add_conversion_weights(source_unit, destination_unit)
        conversion_dict = self.conversionweights[(source_unit.name,destination_unit.name)]
        converted_series = {}
        
        for index in destination_unit.units:
            
            value = sum(series[origin[0]] * origin[1] for origin in conversion_dict[index])
            converted_series[index] = value
            
        return converted_series
    
    
class Unit:
    
    def __init__(self,crosswalk: pd.DataFrame, unitname: str,baseunit:str, info = None):
        """initialized basic data for a single unit and it's conversions to the base unit

        Parameters
        ----------
        crosswalk : pd.DataFrame
            crosswalk file to use
        unitname : str
            name of unit
        baseunit : str
            name of base unit
        info : _type_, optional
            whatever you want, by default None
        """        
        self.name = unitname
        self.base = baseunit

        self.crosswalk = crosswalk[[baseunit, unitname]]
        
        self.units = self.crosswalk[self.name].unique()
        self.baseunits = self.crosswalk[self.base].unique()
        
        self.self2base = self.get_self2base() # {unit Idx: [base indices in unit Idx]}
        self.base2self = self.get_base2self() # {base Idx: unit Idx that base Idx is in}
        
        self.counts = {Idx: len(self.self2base[Idx]) for Idx in self.self2base.keys()}
        self.baseweights = self.get_base_weights() # {}
        
    def get_base2self(self):
        """get the mapping of base unit indices to the unit indices they
        correspond to

        Returns
        -------
        dict
            dictionary of the form: {base unit index: unit index}
        """        
        base2self = {}
        
        for index, row in self.crosswalk.iterrows():
            base2self[row[self.base]] = row[self.name]
        return base2self
        
    def get_self2base(self):
        """mapping of the unit indices to the list of base unit indices they correspond to

        Returns
        -------
        dict
            dictionary of the form {unit index: list(base indices)}
        """        
        self2base = {Idx:[] for Idx in self.units}
        
        for index, row in self.crosswalk.iterrows():            
            self2base[row[self.name]].append(row[self.base])
        return self2base
    
    def get_base_weights(self):
        """get the weights of base units as their proportion of the unit index 
        they correspond to, along with that unit. Used to create conversion weights
        
        ex: if base indices 1,2,3 lie in unit index 5 then 
        
        baseweights[1] = (5, .333)
        baseweights[2] = (5, .333)
        baseweights[3] = (5, .333)

        Returns
        -------
        dict
            dictionary of baseweights of the form: {base index: (unit index, baseweight for that unit)}
        """        
        baseweights = {}
        for Idx in self.base2self.keys():
            baseweights[Idx] = (self.base2self[Idx], 1/self.counts[self.base2self[Idx]])
        return baseweights
