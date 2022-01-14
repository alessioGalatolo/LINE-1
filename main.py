# Example of usage
from networkx import read_edgelist, DiGraph
from line import LINE1

# test this method
if __name__ == "__main__":
    graph = read_edgelist("WikiVote.txt",
                          create_using=DiGraph(),
                          nodetype=int,
                          data=(('weight', float),))
    print("Starting embedding...")
    embedding = LINE1().embed(graph)
    print(embedding)
