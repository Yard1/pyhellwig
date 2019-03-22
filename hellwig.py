#!/usr/bin/python3
import itertools
import math
import argparse
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from joblib import Parallel, delayed
from operator import itemgetter

#############################
###
### Hellwig's method/method of the capacity of information bearers for Python, by Antoni Baum
### Written in Python 3.6.7
### Requires numpy, pandas, tqdm, joblib packages
###
### Copyright (c) 2019 Antoni Baum (Yard1)
### Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
### The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###
### usage: hellwig.py [-h] [-d DELIMETER] [-s DECIMAL_SEPARATOR] [-min MIN]
###                   [-max MAX]
###                   [-id INDEPENDENT_VARIABLES [INDEPENDENT_VARIABLES ...]]
###                   file dependent_variable
### 
### Give the best combination of variables for modelling using Hellwig's method.
### 
### positional arguments:
###   file                  Path to the data in .csv format
###   dependent_variable    Dependent variable
###
### optional arguments:
###   -h, --help            show this help message and exit
###   -d DELIMETER, --delimeter DELIMETER
###                         csv delimeter (Default: ;)
###   -s DECIMAL_SEPARATOR, --decimal_separator DECIMAL_SEPARATOR
###                         csv decimal separator (Default: ,)
###   -min MIN, --min MIN   The smallest number of items in a combination
###                         (Default: 1)
###   -max MAX, --max MAX   The largest number of items in a combination (Default:
###                         number of variables)
###   -id INDEPENDENT_VARIABLES [INDEPENDENT_VARIABLES ...], --independent_variables INDEPENDENT_VARIABLES [INDEPENDENT_VARIABLES ...]
###                         Independent variables to check. If not given, will
###                         check all independent variables in the dataset
###
#############################

def main(fname, dependent_var, independent_vars, delimeter, decimal, min, max):
    if min < 1:
        raise ValueError("min cannot be smaller than 1")
    df = pd.read_csv(fname, delimiter=str(delimeter), decimal=str(decimal))
    df.apply(pd.to_numeric, errors='coerce')
    dependent_df = df[dependent_var]
    independent_df = df.drop(dependent_var, axis=1)
    if not independent_vars or len(independent_vars) < 1:
        independent_vars = independent_df.keys()

    yx_corrs = collections.OrderedDict()
    for var in independent_vars:
        yx_corrs[var] = df[dependent_var].corr(df[var])
    combinations = []
    combinations = Parallel(n_jobs=-1)(delayed(combination)(independent_vars, i) for i in range(min, len(independent_vars)+1 if max < 1 else max))

    best_info = hellwig(independent_df.corr(), yx_corrs, combinations)
    print(list(best_info[0]), best_info[1])

def hellwig(correlation_matrix, dependent_var_correlation_matrix, var_combinations):
    best_info = []
    for combination in tqdm(var_combinations):
        h = Parallel(n_jobs=-1)(delayed(hellwig_singular)(correlation_matrix, dependent_var_correlation_matrix, c) for c in combination)
        h = max(h,key=itemgetter(1))
        best_info.append(h)
    best_info = max(best_info,key=itemgetter(1))
    return best_info

def hellwig_singular(correlation_matrix, dependent_var_correlation_matrix, var_combination):
    h = 0
    var_combination = list(var_combination)
    denominator = 0
    for var in var_combination:
        denominator += abs(correlation_matrix[var_combination[0]][var])
    for var in var_combination:
        h += (dependent_var_correlation_matrix[var]**2)/denominator
    return (var_combination, h)

def combination(iterable, n):
    return itertools.combinations(iterable, n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Give the best combination of variables for modelling using Hellwig\'s method.')
    parser.add_argument( 'file', help='Path to the data in .csv format')
    parser.add_argument( 'dependent_variable', help='Dependent variable')
    parser.add_argument('-d', '--delimeter', required=False, default=";",
                    help='csv delimeter (Default: ;)')
    parser.add_argument('-s', '--decimal_separator', required=False, default=",",
                    help='csv decimal separator (Default: ,)')
    parser.add_argument('-min','--min', type=int, required=False, default=1,
                        help='The smallest number of items in a combination (Default: 1)')
    parser.add_argument('-max','--max', type=int, required=False, default=-1,
                        help='The largest number of items in a combination (Default: number of variables)')
    parser.add_argument('-id','--independent_variables', type=str, nargs='+', required=False,
                        help='Independent variables to check. If not given, will check all independent variables in the dataset')
    args = parser.parse_args()
    main(args.file, args.dependent_variable, args.independent_variables, args.delimeter, args.decimal_separator, args.min, args.max)