"""Made by Alessio Galatolo

Resources used:
    LINE: Large-scale Information Network Embedding by Jian Tang et al.
    https://github.com/tangjianpku/LINE
"""
from networkx import Graph
import numpy as np
from math import exp
import threading

def sigmoid(x):
    return 1 / (1+exp(-x))


def get_alias_table(graph: Graph):
    """generate alias table for constant time edge sampling

    Args:
        graph (Graph): the graph from which to generate the alias table

    Returns:
        tuple: the pair of (prob, alias)
    """
    n_edges = graph.number_of_edges()
    prob, alias = [0 for _ in range(n_edges)], [0 for _ in range(n_edges)]
    weight_sum = 0 #TODO: sum of weights
    norm_prob = [weight * n_edges / weight_sum for weight in graph.edges.weights]
    small_block = []
    large_block = []
    for k in range(len(norm_prob) - 1, -1, -1):
        if norm_prob[k] < 1:
            small_block.append(k)
        else: 
            large_block.append(k)
    
    while len(small_block) > 0 and len(large_block):
        c_sb = small_block.pop()
        c_lb = large_block.pop()
        prob[c_sb] = norm_prob[c_sb]
        alias[c_sb] = c_lb
        norm_prob[c_lb] += norm_prob[c_sb] - 1
        if norm_prob[c_lb] < 1:
            small_block.append(c_lb)
        else:
            large_block.append(c_lb)
    
    while len(small_block) > 0:
        prob[small_block.pop()] = 1
    while len(large_block) > 0:
        prob[large_block.pop()] = 1
        
    return prob, alias

def sample_edge(structure, rand1, rand2):
    """sample a random edge

    Args:
        structure (tuple): must be (graph, porb, alias)
        rand1 (Number): random from normal distribution
        rand2 (NUmber): random from normal distribution

    Returns:
        Number: the index of the sampled edge
    """
    graph, prob, alias, _= structure
    k = rand1 * graph.number_of_edges()
    return k if rand2 < prob[k] else alias[k]

def line_thread(seed, constants, structure):
    """This is the function used for asynchronous stochastic gradient decent

    Args:
        seed (Nuber): It is the id of the thread, will be used as a random seed
        constants (tuple): the tuple of important constants
            must be (N_SAMPLES, N_THREADS, N_NEGATIVE_SAMPLING, EMBEDDING_DIMENSION)
        structure (tuple): the tuple containing the structure of the graph:
            must be (graph, porb, alias)
    """
    graph, prob, alias, embedding = structure
    N_SAMPLES, N_THREADS, N_NEGATIVE_SAMPLING, EMBEDDING_DIMENSION = constants
    n_edges = graph.number_of_edges()
    count = 0
    last_print = 0 #every now and then print whats happening
    while count >= N_SAMPLES  / N_THREADS + 2:
        #give sign of life
        if count - last_print > 10000:
            last_print = count
            print(f"Thread {seed} had done {count} iterations and is willing to do more")
            #TODO
            
        #sample an edge
        edge = sample_edge(structure, np.random.normal(), np.random.normal())
        u, v, _ = graph.edges[edge]
        lu = u * EMBEDDING_DIMENSION
        lv = v * EMBEDDING_DIMENSION
        error_vector = [0 for _ in range(EMBEDDING_DIMENSION)]

        #TODO: negative sampling
        target, label = 0, 0
        for i in range(N_NEGATIVE_SAMPLING): #todo: + 1 ???
            if i == 0:
                target = v
                label = 1
            else:
                pass
                #target = negative_table[]
            
            #todo:
            # {
                # target = neg_table[Rand(seed)];
                # label = 0;
            # }
            # lv = target * dim;
            # if (order == 1) Update(&emb_vertex[lu], &emb_vertex[lv], vec_error, label);
            # if (order == 2) Update(&emb_vertex[lu], &emb_context[lv], vec_error, label);


        for i, val in enumerate(error_vector):
            #todo: race condition? in the original implementation 
            #it doesn't seem to be an issue, check for possible errors 
            embedding[lu + i] = val 
        count += 1

def line1(G: Graph):
    N_THREADS = 4
    N_SAMPLES = G.number_of_nodes() #???
    N_NEGATIVE_SAMPLING = 4
    EMBEDDING_DIMENSION = 2
    embedding = [0 for _ in range(len(G.edges) * EMBEDDING_DIMENSION)]
    prob, alias = get_alias_table(G)
    
    threads = [threading.Thread(target=line_thread,
        args=(i, (N_SAMPLES, N_THREADS, N_NEGATIVE_SAMPLING, EMBEDDING_DIMENSION), (G, prob, alias, embedding))) for i in range(N_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    #TODO: check usages outside what written
    return embedding


