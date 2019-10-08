import pandas as pd

#test_make_single_prediction()

df = pd.DataFrame([['a', 'b'], ['c', 'd']],
                 index=['row 1', 'row 2'],
                  columns=['col 1', 'col 2'])


#print(df)

val = {'Unnamed: 0': 0, 'age': 0.301369863, 'education-num': 0.8, 'capital-gain': 0.667491852, 'hours-per-week': 0.397959184,
 'marital-status_ Divorced': 0, 'marital-status_ Married-AF-spouse': 0, 'marital-status_ Married-civ-spouse': 0,
 'marital-status_ Married-spouse-absent': 0, 'marital-status_ Never-married': 1, 'marital-status_ Separated': 0,
 'marital-status_ Widowed': 0, 'relationship_ Husband': 0, 'relationship_ Not-in-family': 1,
 'relationship_ Other-relative': 0, 'relationship_ Own-child': 0, 'relationship_ Unmarried': 0, 'relationship_ Wife': 0}


val2 = {"Product":{"0":"Desktop Computer","1":"Tablet","2":"iPhone","3":"Laptop"},"Price":{"0":700,"1":250,"2":800,"3":1200}}


val3 = {"Product":"0","Price":"700"}


dfrev = pd.read_json(val3)

print(dfrev)