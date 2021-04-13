import pandas as pd
import networkx as nx
import os
import requests
import arrow
import json
import matplotlib.pyplot as plt
import math

from typing import List, Dict
from classes import Company, Investor, Technology

# import os.path

from pandas import json_normalize 
# from dotenv import load_dotenv
# load_dotenv(verbose=True)

from networkx.algorithms import bipartite

def df_from_api_CB (query, cb_table):
    """Creates the DataFrame of the data from the CrunchBase (CB) API

    Args:
        - query: CB query that specifies what fields we are needed 
        - cb_table: typology of CB table (organizations or investors or people...)

    Return:
        - df: DataFrame from CB
    """

    api_key = os.getenv("CRUNCHBASE_API_KEY") # extract API key 
    base_url = "https://api.crunchbase.com/api/"
    params = {"user_key" : api_key}

    url_table = "https://api.crunchbase.com/api/v4/searches/" + cb_table

    r = requests.post(url_table, json=query, params=params)

    json_data = json.loads(r.text)

    df = pd.json_normalize(json_data['entities'])

    return df


def CB_data_cleaning (
    df: pd.DataFrame, 
    to_drop: List[str], 
    to_rename: Dict[str, str], 
    to_check_double: Dict[str, str],
    sort_by: str = ""):
    """Performs the Data Cleaning part of the CB dataset

    Args:
        - df: dataset to clean
        - to_drop: columns to drop
        - to_rename: columns to rename and new name
        - to_check_double: columns to check. If they bring additional value and,
                           in case they don't, drop them
        - sort_by: column by which sort values

    Return:
        - df: cleaned dataset
    """

    df = df.drop(to_drop, axis=1)
    df = df.rename(columns = to_rename)

    for key, item in to_check_double.items():
        # item does not bring new info:
        if (df[key] == df[item]).all() == True: 
            df = df.drop([item], axis=1)

    if len(sort_by)>0:
        df = df.sort_values(sort_by)

    sort_by

    return df


def extract_data_from_column(column, get_what:str):
    """Extracts data from complex columns

    Args:
        - column: data to be changed
        - get_what: value or values we are interested in

    Return:
        - column: cleaned data
    """

    if isinstance(column, pd.core.series.Series):
        column = column.values

    i = 0
    for line in column:
        
        if issubclass(type(line), float) == False:
            column[i] = [ x.get(get_what) for x in line]
        
        i = i+1

    return column


def extract_classes_company_tech(df):
    """Extracts the dictionaries of Companies and Technologies 
    from the dataset 
    
    Args:
        - df: dataset

    Return:
        - dict_companies: dictionary of companies
        - dict_tech: dictionary of technologies
    """

    df = df.dropna(subset=['location_comp'])
    
    # dictionary of companies: name company: class Company
    dict_companies = {}
    # dictionary of technologies: name technology: class Technology
    dict_tech = {}
    # initialization bipartite graph:
    B = nx.Graph()
    
    for index, row in df.iterrows():   
        
        location_df = row['location_comp']
        location_company = {x.get('location_type'):x.get('value') for x in location_df}

        """location_company = {}

        for x in location_df:
            #print(x)
            loc_type = x['location_type']
            value = x['value']
            location_company[loc_type] = value"""
    
        # Companies:
        comp_name = row['name']

        c = Company(
            id=row['uuid'],
            name=comp_name,
            location = location_company
                   )
        
        dict_companies[comp_name] = c

        B.add_node(comp_name, bipartite=0)
        
        # Technologies:
        if issubclass(type(row['category_groups']), List):
            for tech in row['category_groups']:
                t = Technology(name=tech)
                dict_tech[tech] = t

                B.add_node(tech, bipartite=1)
                B.add_edge(comp_name, tech)
        else:
            t = Technology(name=tech)
            dict_tech[tech] = t   

            B.add_node(tech, bipartite=1)
            B.add_edge(comp_name, tech)

    return dict_companies, dict_tech, B


def nx_dip_graph_from_pandas(df):
    """Creates the bipartite graph from the dataset

    bipartite = 0 => company
    bipartite = 1 => other value

    Args:
        - df: Datafame

    Return:
        - B: bipartite graph 
    """

    df_columns = list(df.columns)
    df_columns = df_columns[0]

    B = nx.Graph()
    
    for name, value in df.iterrows():
    
        value = value[df_columns]
        
        if issubclass(type(value), float) == False:
        
            B.add_node(name, bipartite=0)

            if issubclass(type(value), str): # if only one value
                B.add_node(value, bipartite=1)
                B.add_edge(name, value)

            else: # if value has more values (list)
                for x in value:
                    B.add_node(x, bipartite=1)
                    B.add_edge(name, x)

    return B


def extract_nodes(G, bipartite_set) -> List:
    """Extract nodes from the nodes of one of the bipartite sets

    Args:
        - G: graph
        - bipartite_set: select one of the two sets (0 or 1)

    Return:
        - nodes: list of nodes of that set
    """

    nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == bipartite_set]

    return nodes


def filter_dict(G, percentage, set1, set2):
    """Selects values to delete from the graph G according to ...

    Args:
        - G: graph 
        - 
        -

    Return:
        - to_delete: list of values to delete
    """
    
    threshold_companies = math.ceil( len(set2)/percentage )
    
    dict_nodes = nx.degree(G, set1) 
    
    to_delete= []
    
    # Iterate over all the items in dictionary
    for (key, value) in dict(dict_nodes).items():
        
        if value <= threshold_companies:
            to_delete.append(key)
    
    return to_delete


def plot_bipartite_graph(G, small_degree=True, percentage=10, circular=False):
    """Plots the bipartite network ...

    Args:
        - G: graph 
        - small_degree
        - percentage
        - circular
    """

    set1 = [node for node in G.nodes() if G.nodes[node]['bipartite']==0]
    set2 = [node for node in G.nodes() if G.nodes[node]['bipartite']==1]

    if circular==False:
        pos=nx.spring_layout(G) # positions for all nodes
    else:
        pos=nx.circular_layout(G)

    if small_degree == False: # don't plot nodes with low number of edges
        to_delete = filter_dict(G, percentage, set1, set2)

        G_filtered = G.copy()
        G_filtered.remove_nodes_from(to_delete)
        G_filtered.remove_nodes_from(list(nx.isolates(G_filtered)))

        plot_bipartite_graph(G_filtered, small_degree=True, percentage=percentage, circular=circular)
        
        return 

    # company, value = bipartite.sets(G)
    # subsituted with:
    company = [node for node in G.nodes() if G.nodes[node]['bipartite']==0]
    value = [node for node in G.nodes() if G.nodes[node]['bipartite']==1]

    # calculate degree centrality
    companyDegree = nx.degree(G, company) 
    valueDegree = nx.degree(G, value)

    if len(set1)>30:
        plt.figure(1,figsize=(25,15))
    else: 
        plt.figure(1,figsize=(15,10)) 
    
    plt.axis('off')

    # nodes
    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=company,
                           node_color='r',
                           node_size=[v * 100 for v in dict(companyDegree).values()],
                           alpha=0.3,
                           label=company)

    nx.draw_networkx_nodes(G,
                           pos,
                           nodelist=value,
                           node_color='b',
                           node_size=[v * 200 for v in dict(valueDegree).values()],
                           alpha=0.3,
                           label=value)

    nx.draw_networkx_labels(G, pos, {n: n for n in company}, font_size=10)
    nx.draw_networkx_labels(G, pos, {n: n for n in value}, font_size=10)

    # edges
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)

    return 


def degree_bip(G):
    """Gives the degree of each node of the two bipartite graphs.

    degree: number of links

    Args:
        - G: graph

    Return:
        - degree_set0(1): degree of the bipartite set 0(1)
    """

    set0 = extract_nodes(G, 0)
    set1 = extract_nodes(G, 1)

    # calculate degree centrality
    degree_set0 = nx.degree(G, set0) 
    degree_set1 = nx.degree(G, set1)

    return dict(degree_set0), dict(degree_set1)


def insert_data_classes(dict_class: Dict, new_data: Dict, feature: str):

    """

    Arg:
        - dict_class: (name, class)
        - new_data: dict or DegreeView with the name and the value of the feature
        - feature: name of the feature in class to update 

    Return:
        - dict_class: (name, class) with updated class
    """
    
    for name, value in new_data.items():
        
        if name not in dict_class.keys():
            print(f"Error: try to add a feature for {name}, which is not in class")
            return
        
        c = dict_class[name]
        
        setattr(c, feature, value)
    
    return dict_class


