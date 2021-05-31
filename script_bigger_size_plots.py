import math
import arrow

import ipynb 
import os.path
import json
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from dotenv import load_dotenv
from networkx.algorithms import bipartite
from importlib import reload

from typing import List



# import functions from py file 

import function
reload(function)
from function import CB_data_cleaning, df_from_api_CB, extract_nodes, extract_data_from_column
from function import nx_dip_graph_from_pandas, plot_bipartite_graph, filter_dict
from function import extract_classes_company_tech, degree_bip, insert_data_classes


# In[96]:


# import classes 

import classes
reload(classes)
from classes import Company, Investor, Technology


# ### Download data from CSV

df_start = pd.read_csv("data/data_cb/organizations.csv")


df_start.columns


# ### Data Cleaning


to_drop = [
    'type',
    'permalink',
    'cb_url',   
    'created_at',
    'domain',
    'address',
    'state_code',
    'updated_at',
    'legal_name',
    'roles',
    'postal_code',
    'homepage_url',
    'num_funding_rounds',
    'total_funding_currency_code',
    'phone',
    'email',
    'num_exits',
    'alias2',
    'alias3',
    'num_exits',
    'logo_url',
    'alias1',
    'last_funding_on',
    'twitter_url',
    'facebook_url'
]

# to_rename = { 'category_groups_list': 'category_groups' }
to_rename = { 'category_list': 'category_groups' }

drop_if_nan = [
    'category_groups',
    'rank',
    'short_description'
]

to_check_double = {}

sort_by = "rank"

df = CB_data_cleaning(df_start, to_drop, to_rename, to_check_double, drop_if_nan, sort_by)



# convert category_groups to list

def convert_to_list(string):
    li = list(string.split(","))
    return li
  
if type(df["category_groups"][df.index[0]]) != list:
    df["category_groups"] = [convert_to_list(x) for x in df["category_groups"]]


# ### Select cybersecurity


cybersecurity_words = [
    "cybersecurity"
    "confidentiality",
    "integrity",
    "availability",
    "secure",
    "security",
    "safe",
    "reliability",
    "dependability",
    "confidential",
    "confidentiality",
    "integrity",
    "availability",
    "defense",
    "defence",
    "defensive",
    "privacy"
]

def check_desc(line, words):
    if isinstance(line, float):
        return False
    
    #print(sum(w in line for w in words))
    
    if sum(w in line for w in words)>1:
    #if any(w in line for w in words):
        return True
    return False


df = df.loc[df["short_description"].apply(lambda x: check_desc(x, cybersecurity_words))]

# ### Create Companies and Technologies classes

# #### Ranking

# ### Set limits

#df = df[:100]

# ### Create graph

[dict_companies, dict_tech, B] = extract_classes_company_tech(df)
print(f"We have {len(dict_companies)} companies and {len(dict_tech)} technologies")

[company_degree, tech_degree] = degree_bip(B)

# sort by value
company_degree_sorted = dict(sorted(company_degree.items(), key=lambda item: item[1], reverse=True))
tech_degree_sorted = dict(sorted(tech_degree.items(), key=lambda item: item[1], reverse=True))

# only maximum 
num_max = 10


# check we don't go out of range
if len(company_degree)<num_max or len(tech_degree)<num_max:
    minn = min(len(company_degree), len(tech_degree))
    num_max = minn-1

def limit_value(x, num_max_perc):
    return list(x.values())[num_max_perc]

company_degree_max = {k: company_degree_sorted[k] for k in list(company_degree_sorted.keys())[:num_max]}
tech_degree_max = {k: tech_degree_sorted[k] for k in list(tech_degree_sorted.keys())[:num_max]}


dict_companies = insert_data_classes(dict_companies, dict(company_degree), 'degree')


# technologies' degree
dict_tech = insert_data_classes(dict_tech, dict(tech_degree), 'degree')


a = list(company_degree.values())
b = list(tech_degree.values())
degrees_a = range(len(a))
degrees_b = range(len(b))


from networkx.algorithms.bipartite.matrix import biadjacency_matrix



set0 = extract_nodes(B, 0)
set1 = extract_nodes(B, 1)

# adjacency matrix of bipartite graph
adj_matrix = biadjacency_matrix(B, set0, set1)


adj_matrix_dense = adj_matrix.todense()

a = np.squeeze(np.asarray(adj_matrix_dense))


import functions_method_of_reflections1
reload(functions_method_of_reflections1)
from functions_method_of_reflections1 import zero_order_score, Gct_beta, Gtc_alpha, make_G_hat, next_order_score, generator_order_w
from functions_method_of_reflections1 import M_test_triangular, w_stream, find_convergence, rank_df_class


# M is the array version of the matrix adj_matrix_dense:
M = np.squeeze(np.asarray(adj_matrix_dense))


# ### Triangularize matrix


# M_test_triangular(adj_matrix_dense)


# ### Zero order score


k_c, k_t = zero_order_score(M)

# COMPANY
convergence_comp = find_convergence(M, alpha=0.5, beta=0.5, fit_or_ubiq='fitness', do_plot=True)
plt.savefig(f'plots/CSV_company_rank_evolution_{str(len(df))}_bigger.pdf')
plt.savefig(f'plots/CSV_company_rank_evolution_{str(len(df))}_bigger.png')

#f_final_companies, dict_companies = rank_df_class(convergence_comp, dict_companies)

# relative rank
#df_final_companies['final_rank_normlized'] = df_final_companies['final_rank']/np.max(list(df_final_companies['final_rank']))*10
#n = np.max(df_final_companies['rank_CB']) + 1
#df_final_companies['rank_CB_normlized'] = n - df_final_companies['rank_CB']

#df_final_companies['TeckRank_int'] = df_final_companies.index + 1.0

#df_spearman = df_final_companies[["TeckRank_int", "rank_CB_normlized"]]
#df_spearman = df_spearman.astype(float)
#df_spearman["name"] = df_final_companies['final_configuration']
#df_spearman.set_index("name")


# save df
#name = "csv_results/complete_companies_" + str(len(df)) + ".csv"
#df_final_companies.to_csv(name, index = False, header=True)


# **Sperman correlation**


#sns.lmplot(x="rank_CB_normlized", y="TeckRank_int", data=df_spearman)
#plt.show()



#print(df_spearman.corr(method='spearman'))

# TECHNOLOGY
convergence_tech = find_convergence(M, alpha=0.5, beta=0.5, fit_or_ubiq='ubiquity', do_plot=True)

plt.savefig(f'plots/CSV_tech_rank_evolution_{str(len(df))}_bigger.pdf')
plt.savefig(f'plots/CSV_tech_rank_evolution_{str(len(df))}_bigger.png')

df_final_tech, dict_tech = rank_df_class(convergence_tech, dict_tech)

df_final_tech



# save df
name = "csv_results/complete_tech_" + str(len(df)) + ".csv"

df_final_tech.to_csv(name, index = False, header=True)


