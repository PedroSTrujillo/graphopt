# graphopt

`graphopt` is a unified Python interface for graph combinatorial optimization problems. The goal is to use different methods to solve or approximate NP-hard graph theoretical problems, including:

- Max-Cut
- Colorings
- Cliques
- Hamiltonian Cycles

Pass a graph, pick a problem and a method (ILP, SAT, local search, or an SDP relaxation with randomized rounding), and get back a solution together with a certificate: a dual bound or proof object you can check independently.


