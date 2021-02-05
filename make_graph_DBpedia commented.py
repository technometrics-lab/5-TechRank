import json
import os
import argparse
from tqdm.notebook  import tqdm
import glob
import networkx as nx
from networkx.readwrite import json_graph


def prepare_data(args):

    # url = "http://dbpedia.org/data/{}.jsod"

    company_path = args.company
    indeed_path = args.indeed
    mag_path = args.mag
    patentsview_path = args.patentsview

    # glob.glob( company_path + "/part*" ) returns a list of string of all the elements in company_path which begins with part
    # I guess it is a list of companies...

    company_id_list = list() # initialise empty list of companies id
    # for each line of each company element, identify the id and add it in the company_id_list 
    for result_path in glob.glob( company_path + "/part*" ):

         
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                json_line = json.loads(line)
                company_id = json_line['uid']
                company_id_list.append(company_id)

    wiki_terms = list()

    indeed_annotation_dict = dict() # dict where key=indeed_id and items=

    for result_path in glob.glob(indeed_path+"/annotation/**/part*", recursive=True):

        #     print(result_path)

        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                json_line = json.loads(line)
                indeed_id = json_line['uid']

                if indeed_id not in indeed_annotation_dict: # if the key is not already present in the dict
                    indeed_annotation_dict[indeed_id] = [] # empty item

                if 'annotations' in json_line: # what is annotations??
                    annotations = json_line['annotations']
                    for anno in annotations:
                        term = anno['uri'].split("/")[-1].lower()
                        indeed_annotation_dict[indeed_id].append(term)
                        wiki_terms.append(term)
                else:
                    term = json_line['uri'].split("/")[-1].lower()
                    indeed_annotation_dict[indeed_id].append(term)
                    wiki_terms.append(term)

    indeed_company_dict = dict()

    for result_path in glob.glob(
            indeed_path+"/linked/dataset.json/**/part*",
            recursive=True):
        #     print(result_path)
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                json_line = json.loads(line)
                indeed_id = json_line['id']
                if indeed_id not in indeed_company_dict:
                    indeed_company_dict[indeed_id] = []
                company_id = json_line['uid']
                indeed_company_dict[indeed_id].append(company_id)
                company_id_list.append(company_id)

    mag_annotation_dict = dict()
    for result_path in glob.glob(
            mag_path+"/annotation/**/part*",
            recursive=True):
        #     print(result_path)
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                json_line = json.loads(line)
                indeed_id = json_line['uid']
                if indeed_id not in mag_annotation_dict:
                    mag_annotation_dict[indeed_id] = []
                if 'annotations' in json_line:
                    annotations = json_line['annotations']
                    for anno in annotations:
                        term = anno['uri'].split("/")[-1].lower()
                        mag_annotation_dict[indeed_id].append(term)
                        wiki_terms.append(term)
                else:
                    term = json_line['uri'].split("/")[-1].lower()
                    mag_annotation_dict[indeed_id].append(term)
                    wiki_terms.append(term)

    # wiki_terms = list()
    patent_annotation_dict = dict()
    for result_path in glob.glob(
            patentsview_path + "/annotation/**/part*",
            recursive=True):
        #     print(result_path)
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                json_line = json.loads(line)
                indeed_id = json_line['uid']
                if indeed_id not in patent_annotation_dict:
                    patent_annotation_dict[indeed_id] = []
                if 'annotations' in json_line:
                    annotations = json_line['annotations']
                    for anno in annotations:
                        term = anno['uri'].split("/")[-1].lower()
                        patent_annotation_dict[indeed_id].append(term)
                        wiki_terms.append(term)
                else:
                    term = json_line['uri'].split("/")[-1].lower()
                    patent_annotation_dict[indeed_id].append(term)
                    wiki_terms.append(term)

    patent_company_dict = dict()
    for result_path in glob.glob(
            patentsview_path +"/linked/dataset.json/**/part*",
            recursive=True):
        #     print(result_path)
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                json_line = json.loads(line)
                indeed_id = json_line['patent_id']
                if indeed_id not in patent_company_dict:
                    patent_company_dict[indeed_id] = []
                company_id = json_line['uid']
                patent_company_dict[indeed_id].append(company_id)
                company_id_list.append(company_id)

    return company_id_list,wiki_terms,indeed_annotation_dict,indeed_company_dict,mag_annotation_dict,patent_annotation_dict,patent_company_dict

# function that creates the undirected graph 
def make_graph(company_id_list, 
                wiki_terms,
                indeed_annotation_dict,
                indeed_company_dict,
                mag_annotation_dict,
                patent_annotation_dict,
                patent_company_dict):
    
    G = nx.Graph()
    node_id = 0
    
    company_type, term_type, indeed_type, mag_type, patent_type = list(range(5))
    
    company_map, term_map, indeed_map, mag_map, patent_map = {}, {}, {}, {}, {}

    # add the nodes of different types. In particular we have 5 type:
    # company, term, indeed, mag, patent

    for company in tqdm(company_id_list):
        G.add_node(node_id, feature=[company_type], label=[company_type], content=[company])
        company_map[company] = node_id
        node_id += 1
    print(node_id)

    for term in tqdm(wiki_terms):
        G.add_node(node_id, feature=[term_type], label=[term_type], content=[term])
        term_map[term] = node_id
        node_id += 1
    print(node_id)

    # set_all_kw2 = list(set_all_kw2)
    for indeed in tqdm(set(list(indeed_annotation_dict.keys()) + list(indeed_company_dict.keys()))):
        G.add_node(node_id, feature=[indeed_type], label=[indeed_type], content=[indeed])
        indeed_map[indeed] = node_id
        node_id += 1
    print(node_id)

    for mag in tqdm(mag_annotation_dict):
        G.add_node(node_id, feature=[mag_type], label=[mag_type], content=[mag])
        mag_map[mag] = node_id
        node_id += 1
    print(node_id)

    for patent in tqdm(set(list(patent_annotation_dict.keys()) + list(patent_company_dict.keys()))):
        G.add_node(node_id, feature=[patent_type], label=[patent_type], content=[patent])
        patent_map[patent] = node_id
        node_id += 1
    print(node_id)
    for indeed, term_list in tqdm(indeed_annotation_dict.items()):
        for term in term_list:
            try:
                G.add_edge(indeed_map[indeed], term_map[term])
            except:
                #             print("Keo")
                continue

    
    # create edges:

    for indeed, comp_list in tqdm(indeed_company_dict.items()):
        for comp in comp_list:
            try:
                G.add_edge(indeed_map[indeed], company_map[comp])
            except:
                #             print("Keo")
                # what is Keo??
                continue

    for mag, term_list in tqdm(mag_annotation_dict.items()):
        for term in term_list:
            try:
                G.add_edge(mag_map[mag], term_map[term])
            except:
                #             print("Keo")
                continue
    for patent, term_list in tqdm(patent_annotation_dict.items()):
        for term in term_list:
            try:
                G.add_edge(patent_map[patent], term_map[term])
            except:
                #             print("Keo")
                continue

    for patent, comp_list in tqdm(patent_company_dict.items()):
        for comp in comp_list:
            try:
                G.add_edge(patent_map[patent], company_map[comp])
            except:
                #             print("Keo")
                continue

    return G


if __name__ == "__main__":
 
    # argparse is the recommended command-line parsing module in the Python standard library.
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--company',
                        default="/hdd/tam/entity-fishing-custom/data_crawl/data_swiss/tmm_data/mnt/tmm_share/data/20200401T000000/company.json/",
                        type=str,
                        help='path to folder of spacy-indeed result')
    parser.add_argument('-i', '--indeed',
                        default="/hdd/tam/entity-fishing-custom/data_crawl/data_swiss/tmm_data/mnt/tmm_share/data/20200401T000000/indeed/",
                        type=str,
                        help='path to folder of spacy-patent result')
    parser.add_argument('-m', '--mag',
                        default="/hdd/tam/entity-fishing-custom/data_crawl/data_swiss/tmm_data/mnt/tmm_share/data/20200401T000000/mag/",
                        type=str,
                        help='path to folder of spacy-register result')
    parser.add_argument('-pv', '--patentsview',
                        default="/hdd/tam/entity-fishing-custom/data_crawl/data_swiss/tmm_data/mnt/tmm_share/data/20200401T000000/patentsview/",
                        type=str,
                        help='path to folder of indeed result')
    parser.add_argument('-od', '--output_data',
                        default='DBpedia',
                        type=str,
                        help='output path ', 
                        ) 
    parser.add_argument('-o', '--output_path',
                        default='result_pale',
                        type=str,
                        help='output path ',
                        )

    args = parser.parse_args()

    # execute functions:
    company_id_list, wiki_terms, indeed_annotation_dict, indeed_company_dict, mag_annotation_dict, patent_annotation_dict, patent_company_dict \
        = prepare_data(args)

    G = make_graph(company_id_list,wiki_terms,indeed_annotation_dict,indeed_company_dict,mag_annotation_dict,patent_annotation_dict,patent_company_dict)

    # outputs:
    output_data = args.output_data
    if not os.path.exists(output_data): # create the path only if it is not already there.
        os.makedirs(output_data)
    
        
    with open(output_data+"/company_id_list.json", 'w', encoding='utf8') as outfile:
        json.dump(company_id_list, outfile, ensure_ascii=False)
    with open(output_data+"/wiki_terms.json", 'w', encoding='utf8') as outfile:
        json.dump(wiki_terms, outfile, ensure_ascii=False)
    with open(output_data+"/indeed_annotation_dict.json", 'w', encoding='utf8') as outfile:
        json.dump(indeed_annotation_dict, outfile, ensure_ascii=False)
    with open(output_data+"/indeed_company_dict.json", 'w', encoding='utf8') as outfile:
        json.dump(indeed_company_dict, outfile, ensure_ascii=False)
    with open(output_data+"/mag_annotation_dict.json", 'w', encoding='utf8') as outfile:
        json.dump(mag_annotation_dict, outfile, ensure_ascii=False)
    with open(output_data+"/patent_annotation_dict.json", 'w', encoding='utf8') as outfile:
        json.dump(patent_annotation_dict, outfile, ensure_ascii=False)
    with open(output_data+"/patent_company_dict.json", 'w', encoding='utf8') as outfile:
        json.dump(patent_company_dict, outfile, ensure_ascii=False)


    dataset = args.output_path
    dir = "graph/{}/".format(dataset)
    if not os.path.exists(dir):
        os.makedirs(dir)
    res = json_graph.node_link_data(G)

    if not os.path.exists(dir):
        os.makedirs(dir)

    res['nodes'] = [
        {
            'id': node['id'],
            'label': node['label'],
            'content': node['content'],
            'train': True,
            'test': False
        }
        for node in res['nodes']]
    res['links'] = [
        {
            'source': link['source'],
            'target': link['target'],
            'test_removed': False,
            'train_removed': False
        }
        for link in res['links']]

    with open(dir + dataset + "-G.json", 'w') as outfile:
        json.dump(res, outfile)
    print("====================================================================")
    print("==================== MAKE GRAPH DONE ===============================")
    print("====================================================================")
