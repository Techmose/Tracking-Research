import heapq
import numpy as np
from numpy import random
from scipy.optimize import linear_sum_assignment
from timer import Timer
from OneD_Consept import Hypothesis

cache_hits,cache_misses = 0,0

def enumerator(cost_matrix,k,N,M):
    global cache_hits, cache_misses

    atimer = Timer(verbose=False)
    n = len(cost_matrix)
    all_rows = list(range(n)) # list of all row indices
    all_cols = set(range(n))  # set of all column indices

    # initial node
    depth = 0
    path = []
    with atimer:
        rows,cols = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[rows,cols].sum()
    node = (cost,path)

    # initialize data structures
    top_k = []
    pq = []  # priority queue
    heapq.heappush(pq,node)
    cache = { (): 0 }

    while pq and len(top_k) < k:
        current_cost,current_path = heapq.heappop(pq)
        next_row = len(current_path)
        ###Start of EDIT###
        if next_row == N:
            remaining_cols = sorted(all_cols - set(current_path))
            current_path.extend([-1] * len(remaining_cols))
            for row, col in enumerate(current_path[:N]):
                remaining_cols = sorted(all_cols - set(current_path))
                if col<N:
                        current_path[col+N]=(row+M)
            for row, col in enumerate(current_path[N:],start=N):
                remaining_cols = sorted(all_cols - set(current_path))
                if current_path[row]==(-1):
                        current_path[row] = remaining_cols[0]  
            top_k.append((current_cost,current_path))
            continue                  
        ###END of EDIT###
        """ if next_row == n:
            top_k.append((current_cost,current_path))
            continue """
        remaining_cols = all_cols.difference(current_path)
        new_path_rows = all_rows[:next_row+1]
        sub_rows = all_rows[next_row+1:] # for sub-matrix indices
        for col in remaining_cols:
            new_path = current_path + [col]
            sub_cols = tuple(sorted(remaining_cols.difference([col])))
            if sub_cols in cache:
                cache_hits += 1
                sub_cost = cache[sub_cols]
            else:
                cache_misses += 1
                sub_indices = np.ix_(sub_rows,sub_cols)
                sub_cost_matrix = cost_matrix[sub_indices]
                with atimer:
                    rows,cols = linear_sum_assignment(sub_cost_matrix)
                sub_cost = sub_cost_matrix[rows,cols].sum()
                cache[sub_cols] = sub_cost
            prefix_cost = cost_matrix[new_path_rows,new_path].sum()
            new_cost = prefix_cost + sub_cost
            depth = -len(new_path)
            node = (new_cost,new_path)
            heapq.heappush(pq,node)

    print("timer: %.4f (%d calls)" % (atimer.total_time,atimer.total_calls))
    return top_k

def brute_force_enumerator(cost_matrix):
    from itertools import permutations

    n = len(cost_matrix)
    all_rows = list(range(n))

    all_assignments = []
    for pi in permutations(all_rows):
        cost = cost_matrix[all_rows,pi].sum()
        all_assignments.append((cost,list(pi)))

    all_assignments.sort()
    return all_assignments

if __name__ == '__main__':
    seed=0
    n=10
    #k=numpy.math.factorial(n)
    k=15

    tracks = [3, 7, 20]
    measurements = [2.5, 7.5, 10]
    sigma = 1
    lam_birth = 0.1
    lam_clutter = 0.1
    prob_detection = 0.9093095 #.9093 is a birth

    hyp_class = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)

    N=hyp_class.N
    M=hyp_class.M
    print("---COST_MATRIX:")

    with Timer("enumerator"):
        top_k = enumerator(hyp_class.cost_matrix,k,N,M)
        print(top_k)
        #top_k.sort() # sort assignments by lexicographic order

    print("cache hits: %d" % cache_hits)
    print("cache miss: %d" % cache_misses)

    with Timer("brute force"):
        all_k = brute_force_enumerator(hyp_class.cost_matrix)

    """ check_pass = True
    for i in range(k):
        if top_k[i][0] != all_k[i][0]:
            check_pass = False
            break

    if check_pass:
        print("check ok")
    else:
        print("NOT OK")
        import pdb; pdb.set_trace()
        pass """

