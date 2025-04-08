import pandas as pd
import sympy as sp

import improvedcrosswalk as IC

"""
testing for interchange functionality
"""

### set up data and dimensions

# load crosswalks
crosswalk = pd.read_csv('sample/integration_rebuild/src/interchange/crosswalks/time_crosswalk.csv')
crosswalk2 = pd.read_csv('sample/integration_rebuild/src/interchange/crosswalks/space_crosswalk.csv')

# instantiate two UnitDimension objects from the crosswalks corresponding to space and time units
D = IC.UnitDimension('time', crosswalk, 'hour')
D2 = IC.UnitDimension('space', crosswalk2, 'county')

#test types
type(D2.crosswalk)
type(D)

# add units to UnitDimensions
D.add_unit('day')
D.add_unit('weekday')
D.add_unit('day_night')

D2.add_unit('region')
D2.add_unit('subregion')

# create test data

test_series = {(1,1):1,(2,1):1,(3,1):2,(4,1):3,(5,1):4,(6,1):5,(7,1):2, \
               (1,2):1,(2,2):1,(3,2):2,(4,2):3,(5,2):4,(6,2):5,(7,2):2}
    
sym_series = {(1,1):sp.Symbol('a'),(2,1):sp.Symbol('b'),(3,1):sp.Symbol('c'),(4,1):sp.Symbol('d'),(5,1):sp.Symbol('e'),(6,1):sp.Symbol('f'),(7,1):sp.Symbol('g'), \
               (1,2):sp.Symbol('h'),(2,2):sp.Symbol('i'),(3,2):sp.Symbol('j'),(4,2):sp.Symbol('k'),(5,2):sp.Symbol('l'),(6,2):sp.Symbol('m'),(7,2):sp.Symbol('m')}
t_series = {}
for n in range(4):
    for key in test_series.keys():
        temp = list(key)
        temp = temp + [n+1]
        t_series[tuple(temp)] = test_series[key]

s_series = {}
for n in range(4):
    for key in sym_series.keys():
        temp = list(key)
        temp = temp + [n+1]
        s_series[tuple(temp)] = sym_series[key]
        
t_indices = {0:'day', 1:'year', 2: 'region'}
t_dimension = {'time': 0, 'space': 2}

# initialize Interchange with UnitDimensions
I = IC.Interchange([D, D2])

# initialize toy models
M1 = IC.ToyModel('M1')
M2 = IC.ToyModel('M2')
M3 = IC.ToyModel('M3')

# load test parameters into toy models
M1.load_param('unobtainium',t_series, t_indices, t_dimension)
M2.load_param('unobtainium',s_series, t_indices, t_dimension)

# load dummy data into M3 with different units
M3.load_param('unobtainium',{1:1},{0:'weekday',1:'year',2:'subregion'},{'time':0,'space':2})

# add toy models to Interchange
I.add_model(M1)
I.add_model(M2)
I.add_model(M3)
# load test parameters into toy models
M1.load_param('unobtainium',t_series, t_indices, t_dimension)
M2.load_param('unobtainium',s_series, t_indices, t_dimension)
M3.load_param('unobtainium',{1:1},{0:'weekday',1:'year',2:'subregion'},{'time':0,'space':2})
# add toy models to Interchange
I.add_model(M1)
I.add_model(M2)
I.add_model(M3)

# exchange parameter from model M1 to M3
I.exchange( 'unobtainium',source_model='M1',dest_model='M3')

# test series reorg
s = M1.export_param('unobtainium')

print('original form')
print(s.indices)
print(s.data)
s.reorg({0:'region',1:'day',2:'year'},{'space':0,'time':1})    
print('reorganized form')
print(s.indices)
print(s.data)     