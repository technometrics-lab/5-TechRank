import os
import requests
import arrow
import json
import math
import geopandas

import pandas as pd
import networkx as nx
import scipy.stats as ss
import matplotlib.pyplot as plt

import numpy as np
import geopy.geocoders as geocoders

from geopy.geocoders import Nominatim
from typing import Dict
from importlib import reload

import functions.fun_meth_reflections
from functions.fun_meth_reflections import w_star_analytic

geolocator = Nominatim(user_agent='myapplication')


def rank_comparison(a_ranks_sorted: Dict[str, float], 
                    b_ranks_sorted: Dict[str, float], 
                    do_plot=False):
    """ Returns the Spearman correlation of two ranks (in a Dict form) already sorted
    
    Args:
        - a_ranks_sorted: first rank to compare
        - b_ranks_sorted: second rank to compare

    Return:
        - sper_corr: Spearman correlation
    """
    
    a_list = list()
    b_list = list()
    
    for a_name, a_rank in a_ranks_sorted.items():
        
        #find this element in list b
        for b_name, b_rank in b_ranks_sorted.items():
            
            if a_name == b_name: # if same element in a and b
                a_list.append(a_rank)
                b_list.append(b_rank)
                break
    
    # after the loop we have wo lists, a_list and b_list, for which, in each position, we have the rank of the same element. 
    # This is needed to find the correlation
    
    # plot        
    if do_plot:    
        plt.figure(figsize=(10,20))
        plot([1,2], [a_list, b_list], '-o')
        plt.show()

    sper_corr = ss.spearmanr(a_list, b_list)

    return sper_corr


def calibrate_analytic(M, 
                       ua, 
                       dict_class, 
                       exogenous_rank, 
                       index_function, 
                       title, 
                       do_plot=False):
    """ Returns the top parameters after the greed search
    
    Args:
        - M: triangular matrix
        - ua: defines if we are working with 'Technologies' or 'Companies'
        - dict_class: dictionary of technologies of companies according to ua
        - exogenous_rank: basetruth
        - groundtruth_dict: dict to use as groundtruth
        - index_function: describe how mapping the area for the greed search
        - title: title of the plot (in case we have one)
        - do_plot: indicate if plotting
        
        
    Return:
        - top_spearman: dict including 
                                'spearman': ..
                                'alpha': best alpha
                                'beta': best beta
                                'ua'
    """
    
    # ua
    if ua == 'Companies':
        w_star_type = 'w_star_c'
    elif ua == 'Technologies':
        w_star_type = 'w_star_p'
    
    # Create the map:
    squarelen = np.arange(10) # range in which iterate
    # map the position in the grid with the parameter value
    alpha_range = [index_function(x) for x in squarelen]
    beta_range = [index_function(x) for x in squarelen]
    
    landscape = np.zeros(shape=(len(list(alpha_range)),len(list(beta_range))))

    # Structure of the result
    top_spearman = {'spearman':None,'alpha':None, 'beta':None, 'ua':ua}

    # Greed search among all alphas and all betas
    for alpha_index, alpha in enumerate(alpha_range): #_index is the count of the current iteration
        for beta_index, beta in enumerate(beta_range):
            
            # find the analytic weights with that specific alpha and beta:
            [w_rank, dict] = w_star_analytic(M, alpha, beta, ua, dict_class) # it already returns a dictionary
            
            # the next step was to create the dictout of the weights, but we already create it in the function w_star_analytic
            #w_ranks = {name: w_converged[pos] for name, pos in comp_or_tech_dict.iteritems() }
            
            # We need to sort to use rank_comparison
            # note: operator.itemgetter(n) constructs a callable that assumes an iterable object 
            #       (e.g. list, tuple, set) as input, and fetches the n-th element out of it.
            #w_ranks_sorted = sorted(w_converged.iteritems(), key=operator.itemgetter(1))
            w_ranks_sorted = w_rank #not sorting for now

            
            # Spearman correlation between the created rank and the benchmark
            spearman = rank_comparison(w_ranks_sorted, exogenous_rank)
            
            # the spearman variable contains both the correlation and the pvalue:
            sper_corr = spearman[0]
            sper_pvalue = spearman[1]
            print(f"alpha:{alpha}, beta:{beta} --> corr:{sper_corr} pvalue:{sper_pvalue}")

            if sper_pvalue < 0.1: # statistically significant
                landscape[alpha_index][beta_index] = sper_corr
                
                if (not top_spearman['spearman']) or (sper_corr > top_spearman['spearman']):
                    top_spearman['spearman'] = sper_corr
                    top_spearman['alpha'] = alpha
                    top_spearman['beta'] = beta
            else:
                landscape[alpha_index][beta_index] = np.nan

    if np.isnan(landscape).all(): # for all cases teh pvalue was too high 
        print("No significant results")
        return 

    # plot
    if do_plot:
        plt.figure(figsize=(10,10))
        heatmap = plt.imshow(landscape, interpolation='nearest', vmin=-1, vmax=1)
        #heatmap = plt.pcolor(landscape)
        colorbar = plt.colorbar(heatmap)
        plt.xlabel(r'$ \beta $')
        plt.xticks(squarelen, beta_range, rotation=90)
        plt.ylabel(r'$ \alpha $')
        plt.yticks(squarelen, alpha_range)
        plt.title(title)
        
        # to add: save files 
        
    return top_spearman

    
def create_exogenous_rank(ua, dict_class, preferences: Dict[str, float]):
    """Create the exogenous rank used as groundtruth for the model calibration. It accounts for the investors' preferences 
    including exogenous factors, such as previous investments, geographical distance ecc.

    Arg:
        - ua: defines if we are working with 'Technologies' or 'Companies'
        - dict_class: dictionary of technologies of companies according to ua
        - preferences: investors' preferences expressed as Dict(name:percentage). The sum must be 100.
    Return:
        - exogenous_rank
    """

    if sum(preferences.items)!=100: # the sum of all the preferences is not 100
        print("Error: the investors' preferences percentage must sum to 100%")
        return

    # check which factors are included in the preferences
    # work on them if they are included and their preference percentage is >0

    # previous investments 
    if "previous_investments" in preferences.keys() and preferences["previous_investments"]>0:
        if ua=='Technologies': # this factors makes sense only for companies
            print("The previous invetsors factor can be applied only for companies, not technologies")
            return 
        
        
        
    exogenous_rank = 0

    return exogenous_rank


"""
def extract_coordinates_location(location_company):
    Extracts the address, the latiutude and the longitude from the dict of the address
    
    
    str_place = location_company['city'] + ', ' + location_company['region'] #+ ', ' +  row['country_code']

    address = geolocator.geocode(str_place) # coversion to conventional address (valid for the next command)
    
    lat = address.latitude
    lon = address.longitude
    
    return lat, lon 


def haversine_distance(lat1, lon1, lat2, lon2):
    Calculate the haversine distance (in km) between two points on heart
    

    loc1=(lat1, lon1)
    loc2=(lat2, lon2)
    hs.haversine(loc1,loc2)

    return distance # in km
"""