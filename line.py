from networkx import Graph, DiGraph
from networkx.classes.function import is_weighted, set_edge_attributes
import numpy as np
from math import exp
import threading
from typing import Union


class LINE1():
    def __init__(self, n_threads=1, n_negative_sampling=4, embedding_dim=4,
                 neg_sampling_power=0.75, neg_table_size=100000,
                 sigmoid_table_size=1000, sigmoid_bound=6, initial_lr=0.025):
        """
        Init LINE1 embedding method

        Args:
            n_threads (int, optional): number of thread for asynchronous gradient descent. Defaults to 1.
            n_negative_sampling (int, optional): Defaults to 4.
            embedding_dim (int, optional): the dimension of the embedding. Defaults to 4.
            neg_sampling_power (float, optional): the power to elevate the negative samples. Defaults to 0.75.
            neg_table_size (int, optional): Defaults to 100000.
            sigmoid_table_size (int, optional): Defaults to 1000.
            sigmoid_bound (int, optional): Defaults to 6.
            initial_lr (float, optional): max and min value for sigmoid. Defaults to 0.025.
        """
        self.N_THREADS = n_threads
        self.N_NEGATIVE_SAMPLING = n_negative_sampling
        self.EMBEDDING_DIMENSION = embedding_dim
        self.NEG_SAMPLING_POWER = neg_sampling_power
        self.NEG_TABLE_SIZE = neg_table_size
        self.SIGMOID_TABLE_SIZE = sigmoid_table_size
        self.SIGMOID_BOUND = sigmoid_bound
        self.INITIAL_RHO = initial_lr

    # generate various tables for faster computation
    def generate_sigmoid_table(self):
        """Compute and store common sigmoid values
        """
        for i in range(self.SIGMOID_TABLE_SIZE):
            x = 2 * self.SIGMOID_BOUND * i / self.SIGMOID_TABLE_SIZE - self.SIGMOID_BOUND
            self.sigmoid_table[i] = 1 / (1 + exp(-x))

    def generate_negative_table(self, graph: Graph):
        """Get negative vertex samples according to vertex degrees

        Args:
            graph (Graph): The original graph
        """
        sum = 0
        n_nodes = graph.number_of_nodes()
        por, cur_sum, vid = 0, 0, 0
        for (_, d) in graph.degree:
            sum += d ** self.NEG_SAMPLING_POWER
        for i in range(self.NEG_TABLE_SIZE):
            if (i + 1) / self.NEG_TABLE_SIZE > por:
                cur_sum += graph.degree[list(graph.nodes)[vid % n_nodes]] ** self.NEG_SAMPLING_POWER
                por = cur_sum / sum
                vid += 1
            self.negative_table[i] = list(graph.nodes)[vid - 1 % n_nodes]

    def generate_alias_table(self, graph: Graph):
        """ generate alias table for constant time edge sampling.
            Alias table method from: Reducing the sampling complexity of topic models.
            by A. Q. Li et al.
        Args:
            graph (Graph): the graph from which to generate the alias table
        """
        n_edges = graph.number_of_edges()
        self.prob, self.alias = [0 for _ in range(n_edges)], [0 for _ in range(n_edges)]

        # total sum of weights in graph
        weight_sum = 0
        # the graph has weights
        for _, _, w in graph.edges.data("weight"):
            weight_sum += w
        norm_prob = [weight * n_edges / weight_sum for _, _, weight in graph.edges.data("weight")]

        small_block = []
        large_block = []
        for i in range(len(norm_prob) - 1, -1, -1):
            if norm_prob[i] < 1:
                small_block.append(i)
            else:
                large_block.append(i)

        while len(small_block) > 0 and len(large_block) > 0:
            c_sb = small_block.pop()
            c_lb = large_block.pop()
            self.prob[c_sb] = norm_prob[c_sb]
            self.alias[c_sb] = c_lb
            norm_prob[c_lb] += norm_prob[c_sb] - 1
            if norm_prob[c_lb] < 1:
                small_block.append(c_lb)
            else:
                large_block.append(c_lb)

        while len(small_block) > 0:
            self.prob[small_block.pop()] = 1
        while len(large_block) > 0:
            self.prob[large_block.pop()] = 1

    # utility functions
    def fast_sigmoid(self, x):
        """Compute sigmoid of x reusing store values

        Args:
            x (Number): A number (expected to be between between the sigmoid bound)

        Returns:
            Number: Value in [0, 1]
        """
        if x > self.SIGMOID_BOUND:
            return 1
        elif x < -self.SIGMOID_BOUND:
            return 0
        k = int((x + self.SIGMOID_BOUND) * self.SIGMOID_TABLE_SIZE / self.SIGMOID_BOUND / 2)
        return self.sigmoid_table[k]

    def rand(self, seed):
        """Fastly generate a random integer. Code from original source.

        Args:
            seed (Number): the seed for the random generator

        Returns:
            (Number, Number): An updated seed and a (pseudo)random number
        """
        seed = seed * 25214903917 + 11
        return seed, (seed >> 16) % self.NEG_TABLE_SIZE

    # graph related functions
    def sample_edge(self, graph, rand1, rand2):
        """sample a random edge

        Args:
            graph (Graph): the original graph
            rand1 (Number): random from normal distribution
            rand2 (NUmber): random from normal distribution

        Returns:
            Number: the index of the sampled edge
        """
        k = int(rand1 * graph.number_of_edges())
        return k if rand2 < self.prob[k] else self.alias[k]

    def update(self, u, v, error_vector, label):
        """Update embedding

        Args:
            u (Number): The index of the source node
            v (Number): The index of the context node
            error_vector (array-like of Number): The
            label ([type]): [description]
        """
        x, g = 0, 0

        for i in range(self.EMBEDDING_DIMENSION):
            x += self.embedding[u][i] * self.embedding[v][i]
        g = (label - self.fast_sigmoid(x)) * self.rho
        for i in range(self.EMBEDDING_DIMENSION):
            error_vector[i] += g * self.embedding[v][i]
            self.embedding[v][i] += g * self.embedding[u][i]

    def line_thread(self, seed, graph):
        """This is the function used for asynchronous stochastic gradient decent

        Args:
            graph (Graph): the original graph
        """
        count = 0
        last_print = 0  # every now and then print whats happening
        random_generator = np.random.default_rng()
        while count <= self.n_samples / self.N_THREADS + 2:
            # give sign of life and update rho
            if count - last_print > 1e3:
                self.current_sample_count += count - last_print
                last_print = count
                rho = self.INITIAL_RHO * (1 - self.current_sample_count / (self.n_samples + 1))
                if rho < self.INITIAL_RHO * 1e-4:
                    rho = self.INITIAL_RHO * 1e-4

            # sample an edge
            edge = self.sample_edge(graph, random_generator.random(),
                                    random_generator.random())
            u, v = list(graph.edges)[edge]
            error_vector = [0 for _ in range(self.EMBEDDING_DIMENSION)]

            target, label = 0, 0
            for i in range(self.N_NEGATIVE_SAMPLING + 1):
                if i == 0:
                    target = v
                    label = 1
                else:
                    seed, rand_number = self.rand(seed)
                    target = self.negative_table[rand_number]
                    label = 0
                self.update(u, target, error_vector, label)

            for i, val in enumerate(error_vector):
                self.embedding[u][i] += val
            count += 1

    def embed(self, graph: Union[Graph, DiGraph]):
        """Executed LINE1 method for embedding
        Will save the embedding on a file "line1_" + graph_name if given
        If not given a hashing of the graph will be used (SLOW)
        Note: Line needs to be done on a directed graph
        if an undirected graph is given, a directed graph is generated
        where each undirected edge is represented by 2 directed ones

        Args:
            graph (Graph or DiGraph): The graph for which to do the embedding
        Returns:
            dict: contains the pairs (node_id, embedding)
        """
        if type(graph) == Graph:
            graph = graph.to_directed()
        self.prob, self.alias = None, None
        self.negative_table = [0 for _ in range(self.NEG_TABLE_SIZE)]
        self.sigmoid_table = [0 for _ in range(self.SIGMOID_TABLE_SIZE)]
        # the result of the embedding of the vertices
        self.embedding = None
        # for learning rate
        self.rho = self.INITIAL_RHO
        self.current_sample_count = 0
        self.n_samples = min(graph.number_of_edges(), graph.number_of_nodes())

        # check if graph has weights
        if not is_weighted(graph):
            set_edge_attributes(graph, values=1, name='weight')

        self.embedding = {u: [(np.random.random() - 0.5) / self.EMBEDDING_DIMENSION
                              for _ in range(self.EMBEDDING_DIMENSION)]
                          for u in graph.nodes}

        self.generate_alias_table(graph)
        self.generate_negative_table(graph)
        self.generate_sigmoid_table()

        threads = [threading.Thread(target=self.line_thread, args=(i, graph))
                   for i in range(self.N_THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return self.embedding
