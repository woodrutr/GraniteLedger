from improvedcrosswalk import *
import pandas as pd
import sympy as sp
   
"""
Testing
"""

crosswalk = pd.read_csv('crosswalk.csv')
crosswalk2 = pd.read_csv('space_crosswalk.csv')
D = UnitDimension('time', crosswalk, 'hour')
D2 = UnitDimension('space', crosswalk2, 'county')
D.add_unit('day')
D.add_unit('weekday')

D.add_unit('day')
D.add_unit('weekday')
D.add_unit('day_night')
#D.add_conversion_weights(D.units['day'], D.units['weekday'])

D2.add_unit('region')
D2.add_unit('subregion')
#D2.add_conversion_weights(D2.units['region'], D2.units['subregion'])
day_vals = {1:1,2:1,3:2,4:3,5:4,6:5,7:2}
symbolic_day_vals = {1: sp.Symbol('Sun'), 2:sp.Symbol('Mon'), 3: sp.Symbol('Tue'), 4: sp.Symbol('Wed'), 5: sp.Symbol('Thurs'), 6: sp.Symbol('Fri'), 7: sp.Symbol('Sat')}

D.convert_series(D.units['day'],D.units['weekday'],day_vals)
s = D.convert_series(D.units['day'],D.units['weekday'],symbolic_day_vals)
#U = Unit(crosswalk, 'day', 'hour')

test_series = {(1,1):1,(2,1):1,(3,1):2,(4,1):3,(5,1):4,(6,1):5,(7,1):2, \
               (1,2):1,(2,2):1,(3,2):2,(4,2):3,(5,2):4,(6,2):5,(7,2):2}
test_indices = {0: 'day', 1: 'year'}
test_dimension = {'time': 0}

t_series = {}
for n in range(4):
    for key in test_series.keys():
        temp = list(key)
        temp = temp + [n+1]
        t_series[tuple(temp)] = test_series[key]

t_indices = {0:'day', 1:'year', 2: 'region'}
t_dimension = {'time': 0, 'space': 2}




S = ExchangeSeries(test_series,test_indices,test_dimension)
S1 = ExchangeSeries(t_series,t_indices,t_dimension)
I = Interchange([D, D2])

print('data')
print(S.data)
print('present time dimension')
print(S.present_dim('time'))
print('rebuild time dimension in standard form')
print(S.rebuild_series('time', S.present_dim('time')))
print('convert time dimension from to weekday')
print(I.convert_series(S,'time','weekday'))
print('rebuild converted time dimension series in standard form')
print(S.rebuild_series('time', I.convert_series(S,'time','weekday')))
print('testing multiple conversion: before')
print(S1.data)
print(S1.indices)
print(S1.dimensions)
print('change space from region to subregion:')
I.change_series(S1, 'space', 'subregion')
print(S1.data)
print(S1.indices)
print(S1.dimensions)
print('now change time from day to weekday:')
I.change_series(S1, 'time', 'weekday')
print(S1.data)
print(S1.indices)
print(S1.dimensions)
