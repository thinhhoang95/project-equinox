import networkx as nx

def enumerate_nodes(graph: nx.Graph):
    """
    Given a networkx graph, enumerate each node and return a mapping
    from node name to node id (integer).
    
    Args:
        graph (nx.Graph): The input graph.
        
    Returns:
        dict: A dictionary mapping node name to node id.
    """
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    return node_mapping

def node_names_to_ids(graph: nx.Graph, node_names, node_mapping=None):
    """
    Given a list of node names, return the list of node ids (integers).
    If node_mapping is None, generate it using enumerate_nodes.
    
    Args:
        graph (nx.Graph): The input graph.
        node_names (list): List of node names.
        node_mapping (dict, optional): Mapping from node name to node id.
        
    Returns:
        list: List of node ids corresponding to the node names.
    """
    if node_mapping is None:
        node_mapping = enumerate_nodes(graph)
    return [node_mapping[name] for name in node_names]

def node_ids_to_names(graph: nx.Graph, node_ids, node_mapping=None):
    """
    Given a list of node ids (integers), return the list of node names.
    If node_mapping is None, generate it using enumerate_nodes.
    
    Args:
        graph (nx.Graph): The input graph.
        node_ids (list): List of node ids (integers).
        node_mapping (dict, optional): Mapping from node name to node id.
        
    Returns:
        list: List of node names corresponding to the node ids.
    """
    if node_mapping is None:
        node_mapping = enumerate_nodes(graph)
    # Reverse the mapping: id -> name
    id_to_name = {idx: name for name, idx in node_mapping.items()}
    return [id_to_name[node_id] for node_id in node_ids]

def get_adjacency_matrix(graph: nx.Graph):
    """
    Return the adjacency matrix of a given route graph (networkx.Graph).

    Args:
        graph (nx.Graph): The input graph.

    Returns:
        np.ndarray: The adjacency matrix as a numpy array.
    """
    # Use networkx to get the adjacency matrix as a numpy array
    # The order of nodes is the order in graph.nodes()
    return nx.to_numpy_array(graph)
