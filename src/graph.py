import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def plot_graph(G):
    labels = dict()
    pos = []

    for i, node in enumerate(G.nodes):
        labels[i] = G.nodes[i]['label']
        pos.append(G.nodes[i]['pos'])

    pos = np.array(pos)

    nx.draw(G, labels=labels, font_weight='bold', pos=pos[:,:2])
    plt.show()
    nx.draw(G, labels=labels, font_weight='bold', pos=pos[:,1:])
    plt.show()
    nx.draw(G, labels=labels, font_weight='bold', pos=pos[:,[0, 2]])
    plt.show()


def get_delaunay_graph(eloc, picture=False):
    positions = eloc.get_positions()['ch_pos']
    points = np.stack(list(positions.values()))
    ch_names = np.stack(list(positions.keys()))

    tri = Delaunay(points)

    G = nx.Graph()
    for ind, (pos, label) in enumerate(zip(points, ch_names)):
        G.add_node(ind, pos=pos, label=label)

    for path in tri.simplices[:, 1:]:
        nx.add_cycle(G, path)

    if picture:
        plot_graph(G)

    return nx.adjacency_matrix(G)


def get_neighbors_graph(eloc, n_neighbors=9, picture=False):
    positions = eloc.get_positions()['ch_pos']
    points = np.stack(list(positions.values()))
    ch_names = np.stack(list(positions.keys()))

    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(points)

    neighbors_list = neigh.kneighbors(points, return_distance=True)[1]

    G = nx.Graph()
    for ind, (pos, label) in enumerate(zip(points, ch_names)):
        G.add_node(ind, pos=pos, label=label)

    for path in neighbors_list:
        for neighbor in path[1:]:
            G.add_edge(path[0], neighbor)

    if picture:
        plot_graph(G)

    return nx.adjacency_matrix(G)


def get_pos_init_graph(eloc, delta=0.0025):
    positions = eloc.get_positions()['ch_pos']
    points = np.stack(list(positions.values()))

    matrix = np.zeros((len(points), len(points)))

    for i, point_i in enumerate(points):
        for j, point_j in enumerate(points):
            matrix[i,j] = np.clip(delta / np.power(point_i - point_j + 1e-5, 2).sum(), 0.1, 1)

    return matrix
