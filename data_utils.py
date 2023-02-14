
#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import Counter
import os
import numpy as np

def _read_nodes(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if ':' in line:
                line = line.split(':')[0]
            data.extend(line.replace('\n', '').split(','))
        return data

def read_graph(filename, node_to_id):
    N = len(node_to_id)
    A = np.zeros((N,N), dtype=np.float32)
    with open(filename, 'r') as f:
        for line in f:
            edge = line.strip().split()
            if edge[0] in node_to_id and edge[1] in node_to_id:
                source_id = node_to_id[edge[0]]
                target_id = node_to_id[edge[1]]
                # if len(edge) >= 3:
                #     A[source_id,target_id] = float(edge[2])
                # else:
                A[source_id,target_id] = 1.0
    return A

def _build_vocab(filename):
    data = _read_nodes(filename)

    counter = Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    nodes, _ = list(zip(*count_pairs))
    nodes = list(nodes)
    nodes.insert(0,'-1') # index for mask
    node_to_id = dict(zip(nodes, range(len(nodes))))

    return nodes, node_to_id

def _file_to_node_ids(filename, node_to_id):
    data = []
    len_list = []