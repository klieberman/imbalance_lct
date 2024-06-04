import torch
from scipy.stats import loguniform
from numpy.random import uniform, triangular
import numpy as np

_available_lambda_dists = ['loguniform', 
                           'uniform',
                           'fixed',
                           'triangular',
                           'linear']

def available_lambda_dists():
    return _available_lambda_dists


def linear(a, b, hb, size):
    '''
    Custom linear distribution which has domain [a, b]
    and hb is the height of the pdf at x=b
    @a: lower bound of domain
    @b: upper bound of domain
    @hb: height of pdf at x=b
    '''
    ha = 2 / (b - a) - hb
    slope = (hb - ha) / (b - a)
    y_int = ha - slope * a
    if slope == 0:
        return uniform(a, b, size)
    p = uniform(0., 1.0, size=size)
    return (np.sqrt(slope*(a**2*slope+2*p) + 2*a*y_int*slope + y_int**2) - y_int) / slope


def get_lambdas_one(n, **kwargs):
    '''
    Draws n lambda values for one hyperparameter.
    '''
    if kwargs['dist'] == 'loguniform':
        if kwargs['a'] == 0.0:
            a, loc = 0.01, -0.01
        else:
            a, loc = kwargs['a'], 0
        lambdas = loguniform.rvs(a, kwargs['b'], size=n, loc=loc)
    elif kwargs['dist'] == 'uniform':
        lambdas = uniform(kwargs['a'], kwargs['b'], size=n)
    elif kwargs['dist'] == 'fixed':
        lambdas = [kwargs['a']] * n
    elif kwargs['dist'] == 'triangular':
        lambdas = triangular(kwargs['a'], kwargs['mode'], kwargs['b'], size=n)
    elif kwargs['dist'] == 'linear':
        lambdas = linear(kwargs['a'], kwargs['b'], kwargs['hb'], size=n)
    else:
        print(f"Invalid lambda distribution {kwargs['dist']}.")
        return None
    lambdas = torch.Tensor(lambdas).unsqueeze(1)
    return lambdas


def get_lambdas_all(lambda_vars, n):
    '''
    Draws n lambda values for all hyperparameters in lambda_vars
    '''
    lambdas = []
    for _, info in lambda_vars.items():
        lambdas.append(get_lambdas_one(n=n, **info))
    return torch.cat(lambdas, dim=1)
