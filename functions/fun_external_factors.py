import os
import requests
import arrow
import json
import math
import geopandas

import pandas as pd
import networkx as nx
import scipy.stats as ss
import numpy as np
import geopy.geocoders as geocoders

from geopy.geocoders import Nominatim
from typing import Dict

geolocator = Nominatim(user_agent='myapplication')

#---------------------------------------------------------------------------------------------------------------------------------------
def rank_comparison(a_ranks_sorted: Dict[str, float], 
                    b_ranks_sorted: Dict[str, float], 
                    do_plot=False):
    """ Returns the spearman correlation of two already sorted ranks
    
    Args:
        - a_ranks_sorted: first rank to compare
        - b_ranks_sorted: second rank to compare
    """
    
    a_list = list()
    b_list = list()
    
    for atup in a_ranks_sorted:
        aiden = atup[0] # name of element in a
        apos = atup[1]  # rank of element in a
        
        #find this element in list b
        for btup in b_ranks_sorted:
            biden = btup[0] # name of element in b
            bpos = btup[1]  # rank of element in b
            
            if aiden == biden: # if same element in a and b
                a_list.append(apos)
                b_list.append(bpos)
    
    # after the loop we have wo lists, a_list and b_list, for which, in each position, we have the rank of the same element. 
    # This is needed to find the correlation
    
    # plot        
    if do_plot:    
        plt.figure(figsize=(10,20))
        plot([1,2], [a_list, b_list], '-o')
        plt.show()
    
    return ss.spearmanr(a_list, b_list)

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