import numpy as np

def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def str_list_to_int(str_list):
    return [int(item) for item in str_list]

def read_edges(train_filename, test_filename):
    """read data from files 
    Args:
        train_filename: training file name
        test_filename: test file name
    Returns:
        node_num: int, number of nodes in the graph
        pos/neg_graph: dict, node_id -> list of neighbors in the graph
    """
    def add_node_to_graph(node_1, node_2, graph, positive=True, train=True):
        if positive:
            pos_nodes.add(node_1)
            pos_nodes.add(node_2)
        else:
            neg_nodes.add(node_1)
            neg_nodes.add(node_2)

        if graph.get(node_1) is None:
            graph[node_1] = []
        if graph.get(node_2) is None:
            graph[node_2] = []

        if train:  # yc: remove
            graph[node_1].append(node_2)
            graph[node_2].append(node_1)


    pos_graph = {}; neg_graph = {}
    pos_nodes = set()  # roots for the positive BFS tree
    neg_nodes = set()  # roots for the negative BFS tree

    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename) if test_filename != "" else []

    print("\t Reading edges in training file...")
    for edge in train_edges:
        if edge[2] == 1:
            add_node_to_graph(edge[0], edge[1], pos_graph)
        elif edge[2] == -1:
            add_node_to_graph(edge[0], edge[1], neg_graph, positive=False)

    print("\t Reading edges in test file...") # yc: remove
    for edge in test_edges:
        if edge[2] == 1:
            add_node_to_graph(edge[0], edge[1], pos_graph, train=False)
        elif edge[2] == -1:
            add_node_to_graph(edge[0], edge[1], neg_graph, positive=False, train=False)

    n_nodes = pos_nodes | neg_nodes  # set union
    pos_nodes = sorted(list(pos_nodes))
    neg_nodes = sorted(list(neg_nodes))

    return pos_graph, neg_graph, max(n_nodes)+1, pos_nodes, neg_nodes 


def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges


def read_embeddings(filename, n_node, n_embed):
    """read pretrained node embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        embedding_matrix = np.random.rand(n_node, n_embed)
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        return embedding_matrix


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()


def divide_chunks(iter, n_parts):
    for i in range(0, len(iter), n_parts):
        yield iter[i:i + n_parts]


def aggregate_link_emb(link_method, emb_1, emb_2):
    if link_method == "weight_l1":
        link_emb = np.absolute(emb_1 - emb_2)

    elif link_method == "weight_l2": 
        link_emb = (emb_1 - emb_2) ** 2

    elif link_method == "concatenation":
        link_emb = np.zeros(shape=len(emb_1) * 2)
        link_emb[:len(emb_1)] = emb_1
        link_emb[len(emb_1):] = emb_2

    elif link_method == "average":
        link_emb = (emb_1 + emb_2) / 2

    elif link_method == "Hadamard": 
        link_emb = emb_1 * emb_2

    elif link_method == "addition":
        link_emb = emb_1 + emb_2

    return link_emb