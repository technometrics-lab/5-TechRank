import json
import os
import argparse
from tqdm.notebook  import tqdm
import glob
import networkx as nx
from networkx.readwrite import json_graph


def prepare_data(args):
    company_id_list = list()
    # url = "http://dbpedia.org/data/{}.jsod"
    company_path = args.company
    indeed_path = args.indeed
    mag_path = args.mag
    patentsview_path = args.patentsview
    for result_path in glob.glob(
            company_path + "/part*"):
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                json_line = json.loads(line)
                company_id = json_line['uid']
                company_id_list.append(company_id)
    wiki_terms = list()
    indeed_annotaion_dict = dict()
    for result_path in glob.glob(
            indeed_path+"/annotation/**/part*",
            recursive=True):
        #     print(result_path)
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                json_line = json.loads(line)
                indeed_id = json_line['uid']
                if indeed_id not in indeed_annotaion_dict:
                    indeed_annotaion_dict[indeed_id] = []
                if 'annotations' in json_line:
                    annotations = json_line['annotations']
                    for anno in annotations:
                        term = anno['uri'].split("/")[-1].lower()
                        indeed_annotaion_dict[indeed_id].append(term)
                        wiki_terms.append(term)
                else:
                    term = json_line['uri'].split("/")[-1].lower()
                    indeed_annotaion_dict[indeed_id].append(term)
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

    mag_annotaion_dict = dict()
    for result_path in glob.glob(
            mag_path+"/annotation/**/part*",
            recursive=True):
        #     print(result_path)
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                json_line = json.loads(line)
                indeed_id = json_line['uid']
                if indeed_id not in mag_annotaion_dict:
                    mag_annotaion_dict[indeed_id] = []
                if 'annotations' in json_line:
                    annotations = json_line['annotations']
                    for anno in annotations:
                        term = anno['uri'].split("/")[-1].lower()
                        mag_annotaion_dict[indeed_id].append(term)
                        wiki_terms.append(term)
                else:
                    term = json_line['uri'].split("/")[-1].lower()
                    mag_annotaion_dict[indeed_id].append(term)
                    wiki_terms.append(term)
    # wiki_terms = list()
    patent_annotaion_dict = dict()
    for result_path in glob.glob(
            patentsview_path + "/annotation/**/part*",
            recursive=True):
        #     print(result_path)
        with open(result_path, 'r', encoding='utf-8') as file:
            for line in tqdm(file):
                json_line = json.loads(line)
                indeed_id = json_line['uid']
                if indeed_id not in patent_annotaion_dict:
                    patent_annotaion_dict[indeed_id] = []
                if 'annotations' in json_line:
                    annotations = json_line['annotations']
                    for anno in annotations:
                        term = anno['uri'].split("/")[-1].lower()
                        patent_annotaion_dict[indeed_id].append(term)
                        wiki_terms.append(term)
                else:
                    term = json_line['uri'].split("/")[-1].lower()
                    patent_annotaion_dict[indeed_id].append(term)
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

    return company_id_list,wiki_terms,indeed_annotaion_dict,indeed_company_dict,mag_annotaion_dict,patent_annotaion_dict,patent_company_dict

def make_graph(company_id_list,wiki_terms,indeed_annotaion_dict,indeed_company_dict,mag_annotaion_dict,patent_annotaion_dict,patent_company_dict):
    G = nx.Graph()
    node_id = 0
    company_type, term_type, indeed_type, mag_type, patent_type = list(range(5))
    company_map, term_map, indeed_map, mag_map, patent_map = {}, {}, {}, {}, {}

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
    for indeed in tqdm(set(list(indeed_annotaion_dict.keys()) + list(indeed_company_dict.keys()))):
        G.add_node(node_id, feature=[indeed_type], label=[indeed_type], content=[indeed])
        indeed_map[indeed] = node_id
        node_id += 1
    print(node_id)

    for mag in tqdm(mag_annotaion_dict):
        G.add_node(node_id, feature=[mag_type], label=[mag_type], content=[mag])
        mag_map[mag] = node_id
        node_id += 1
    print(node_id)

    for patent in tqdm(set(list(patent_annotaion_dict.keys()) + list(patent_company_dict.keys()))):
        G.add_node(node_id, feature=[patent_type], label=[patent_type], content=[patent])
        patent_map[patent] = node_id
        node_id += 1
    print(node_id)
    for indeed, term_list in tqdm(indeed_annotaion_dict.items()):
        for term in term_list:
            try:
                G.add_edge(indeed_map[indeed], term_map[term])
            except:
                #             print("Keo")
                continue

    for indeed, comp_list in tqdm(indeed_company_dict.items()):
        for comp in comp_list:
            try:
                G.add_edge(indeed_map[indeed], company_map[comp])
            except:
                #             print("Keo")
                continue

    for mag, term_list in tqdm(mag_annotaion_dict.items()):
        for term in term_list:
            try:
                G.add_edge(mag_map[mag], term_map[term])
            except:
                #             print("Keo")
                continue
    for patent, term_list in tqdm(patent_annotaion_dict.items()):
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

    company_id_list, wiki_terms, indeed_annotaion_dict, indeed_company_dict, mag_annotaion_dict, patent_annotaion_dict, patent_company_dict = prepare_data(args)

    G = make_graph(company_id_list,wiki_terms,indeed_annotaion_dict,indeed_company_dict,mag_annotaion_dict,patent_annotaion_dict,patent_company_dict)

    output_data = args.output_data
    if not os.path.exists(output_data):
        os.makedirs(output_data)
    with open(output_data+"/company_id_list.json", 'w', encoding='utf8') as outfile:
        json.dump(company_id_list, outfile, ensure_ascii=False)
    with open(output_data+"/wiki_terms.json", 'w', encoding='utf8') as outfile:
        json.dump(wiki_terms, outfile, ensure_ascii=False)
    with open(output_data+"/indeed_annotaion_dict.json", 'w', encoding='utf8') as outfile:
        json.dump(indeed_annotaion_dict, outfile, ensure_ascii=False)
    with open(output_data+"/indeed_company_dict.json", 'w', encoding='utf8') as outfile:
        json.dump(indeed_company_dict, outfile, ensure_ascii=False)
    with open(output_data+"/mag_annotaion_dict.json", 'w', encoding='utf8') as outfile:
        json.dump(mag_annotaion_dict, outfile, ensure_ascii=False)
    with open(output_data+"/patent_annotaion_dict.json", 'w', encoding='utf8') as outfile:
        json.dump(patent_annotaion_dict, outfile, ensure_ascii=False)
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
