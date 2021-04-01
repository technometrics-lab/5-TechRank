import pandas as pd
import networkx as nx
import os
import requests
import arrow
import json
import matplotlib.pyplot as plt
import math

from typing import List, Dict 

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

    return df


def extract_nodes(G, bipartite_set):
    """Extract nodes from the nodes of one of the bipartite sets

    Args:
        - df: Datafame
        - bipartite_set: select one of the two sets (0 or 1)

    Return:
        - B: bipartite graph 
    """

    nodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == bipartite_set}

    return nodes


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

def plot_bipartite_graph(G, small_degree=True, percentage=10):
    """Plots the bipartite network ...

    Args:
        - G: graph 
        - small_degree
        - percentage
    """

    set1 = [node for node in G.nodes() if G.nodes[node]['bipartite']==0]
    set2 = [node for node in G.nodes() if G.nodes[node]['bipartite']==1]

    pos=nx.spring_layout(G) # positions for all nodes
    #pos=nx.circular_layout(G)


    if small_degree == False: # don't plot nodes with low number of edges
        to_delete = filter_dict(G, percentage, set1, set2)
        G.remove_nodes_from(to_delete)
        G.remove_nodes_from(list(nx.isolates(G)))

        plot_bipartite_graph(G, small_degree=True)
        
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
    nx.draw_networkx_nodes(G,pos,
                        nodelist=company,
                        node_color='r',
                        node_size=[v * 100 for v in dict(companyDegree).values()],
                        alpha=0.3,
                        label=company)

    nx.draw_networkx_nodes(G,pos,
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