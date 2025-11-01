import networkx as nx
import matplotlib.pyplot as plt
import os
import json
from pyvis.network import Network

node_counter = 0

def iterate_extraction(path):
    node = {
        'node_name': os.path.basename(path) or path
    }

    if os.path.isdir(path):
        node['is_directory'] = True
        node['children'] = []
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                
                child_node = iterate_extraction(item_path)
                
                node['children'].append(child_node)
        except PermissionError:
            pass
            
    else:
        node['is_directory'] = False

    return node

def add_nodes_and_edges(graph, node_data, parent_id=None):
    global node_counter
    current_id = node_counter
    node_counter += 1
    
    attributes = node_data.copy()
    if 'children' in attributes:
        del attributes['children']
    graph.add_node(current_id, **attributes)
    
    if parent_id is not None:
        graph.add_edge(parent_id, current_id)
    
    if 'children' in node_data:
        for child in node_data['children']:
            add_nodes_and_edges(graph, child, parent_id=current_id) 

def get_site_raw_routes(src_root_path, output_path):
    output_file = output_path
    
    EXTRACTED_ROUTES = iterate_extraction(src_root_path)
    jsonFile = json.dumps(EXTRACTED_ROUTES, indent=4)

    with open(output_file, 'w') as f:
        f.write(jsonFile)

    data_json = json.loads(jsonFile)
    G = nx.DiGraph()

    add_nodes_and_edges(G, data_json)
    nx.write_gexf(G, "graph.gexf")

    net = Network(height='900px', width='100%', directed=True, notebook=False, cdn_resources='remote')

    net.force_atlas_2based(
        gravity=-60,
        central_gravity=0.015,
        spring_length=250,
        spring_strength=0.08,
        damping=0.7,
        overlap=0
    )

    for node, attrs in G.nodes(data=True):
        is_fixed = True if node == 0 else False
        net.add_node(node, label=attrs.get('node_name', str(node)), fixed=is_fixed)

    net.add_edges(G.edges())
    net.show_buttons(filter_=['physics'])
    net.show('graph.html', notebook=False)

    print("Generated interactive graph: graph.html")
    

get_site_raw_routes("osu_parse", "osu.json")
