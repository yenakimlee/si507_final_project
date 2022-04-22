import json
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import networkx as nx


f = '/Users/yena_kim/Documents/UM/Winter 2022/SI 507/Final Project/graph_structure.json'


def read_json_file(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)

G = read_json_file(f)

for path in nx.all_simple_paths(G, source= 1999, target= 2020):
    for node in path:
        print (G, G.nodes[node])

nx.draw(G, with_labels = True)
plt.show()