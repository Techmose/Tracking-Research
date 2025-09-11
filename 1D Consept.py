import itertools
import math
import numpy as np

#Tracks
T = [5, 9]

#Readings
Z = [4, 8, 15]

#Poisson function for births or clutter given lambda: (lam)mean expected birth/clutter and k: number of events
def poisson_prob(k, lam):
    return math.exp(-lam) * (lam**k) / math.factorial(k)

def poisson_log_prob(k, lam):
    return -lam + k * math.log(lam) - math.lgamma(k + 1)

#Tn = New track / Tf = false alarm
#Matrix represntation of Tracks 
#   Tn T1 T2 Tf
#Z1 [1, 1, 0, 1]
#Z2 [1, 1, 1, 1]
#Z3 [1, 0, 1, 1]

#Likelyhood matrix is the probability of reading z_i given track t_i
def create_likelyhood_matrix(tracks, readings, sigma = 1):
    tracks = np.array(tracks, dtype=float).reshape(-1, 1)   # shape (N,1)
    readings = np.array(readings, dtype=float).reshape(1, -1)  # shape (1,M)
    diff = readings - tracks   # shape (N,M)
    coeff = 1.0 / (np.sqrt(2 * np.pi * sigma**2))
    L = coeff * np.exp(-0.5 * (diff**2) / (sigma**2))
    return L

def likelyhood_without_gaussian(tracks,readings):
    tracks = np.array(tracks, dtype=float).reshape(-1, 1)   # shape (N,1)
    readings = np.array(readings, dtype=float).reshape(1, -1)  # shape (1,M)
    D = np.max(tracks) - np.min(tracks)
    diff = np.abs(readings - tracks)   # shape (N,M)

    L = np.where(diff <= D, 1 - diff / D, 0.0)
    return L
                
def generate_hypothesis_assignments(N, M):
    """
    Generate all feasible hypotheses (assignments only, no probabilities).

    Returns a list of tuples of length M, where each entry is:
        0..N-1 => track index
        N      => birth
        N+1    => clutter

    Enforces: each track is used at most once
              and all tracks must be assigned (no misses).
    """
    choices = range(N + 2)  # N=birth, N+1=clutter
    results = []

    for assignment in itertools.product(choices, repeat=M):
        used_tracks = set()
        valid = True
        for ch in assignment:
            if ch < N:
                if ch in used_tracks:
                    valid = False
                    break
                used_tracks.add(ch)
        if not valid:
            continue

        # require all tracks assigned (no misses allowed)
        if len(used_tracks) != N:
            continue

        results.append(assignment)

    return results

def print_hypothesis_matrix(hypothesis, N, L):
    """
    Hypothesis is a list of tuples of length M, where each entry is:
        0..N-1 => track index
        N      => birth
        N+1    => clutter
    N is the number of tracks
    """
    prob = score_hypothesis_log(hypothesis, L)
    M = len(hypothesis)
    mat = np.zeros((N + 2, M), dtype=int)
    for j, choice in enumerate(hypothesis):
        mat[choice, j] = 1

    row_labels = [f"T{i}" for i in range(N)] + [" B", " C"]
    print("Hypothesis:", hypothesis)
    print("Log Propbability", prob)
    for label, row in zip(row_labels, mat):
        print(f"{label}: {row}")

def score_hypothesis(hypothesis, L, P_D=0.9, lambda_birth=0.2, lambda_clutter=0.8):
    """
    Compute unnormalized probability (weight) of a single hypothesis.

    hypothesis: tuple of choices (len = M)
    L: likelihood matrix shape (N, M)
    """
    N, M = L.shape
    weight = 1.0
    used_tracks = set()

    # count births and clutter
    births = sum(1 for ch in hypothesis if ch == N)
    clutter = sum(1 for ch in hypothesis if ch == N + 1)

    for j, choice in enumerate(hypothesis):
        if choice < N:
            used_tracks.add(choice)
            weight *= P_D * L[choice, j]

    # missed detections for unassigned tracks
    for track in range(N):
        if track not in used_tracks:
            weight *= (1 - P_D)

    # birth/clutter priors from Poisson
    weight *= poisson_prob(births, lambda_birth)
    weight *= poisson_prob(clutter, lambda_clutter)

    return weight

def score_hypothesis_log(hypothesis, L, P_D=0.9, lambda_birth=0.2, lambda_clutter=0.8):
    """
    Compute log-probability (unnormalized) of a single hypothesis.

    Returns log(weight) instead of raw probability.
    """
    N, M = L.shape
    log_w = 0.0
    used_tracks = set()

    births = sum(1 for ch in hypothesis if ch == N)
    clutter = sum(1 for ch in hypothesis if ch == N + 1)

    for j, choice in enumerate(hypothesis):
        if choice < N:
            used_tracks.add(choice)
            if L[choice, j] <= 0:
                return float("-inf")  # impossible assignment
            log_w += math.log(P_D) + math.log(L[choice, j])

    # missed detections (tracks not assigned)
    for track in range(N):
        if track not in used_tracks:
            log_w += math.log(1 - P_D)

    # add log Poisson priors
    log_w += poisson_log_prob(births, lambda_birth)
    log_w += poisson_log_prob(clutter, lambda_clutter)

    return log_w

likelyhood = create_likelyhood_matrix(T, Z)
no_gaussian = likelyhood_without_gaussian(T,Z)

hypothesis = generate_hypothesis_assignments(len(T) , len(Z))
for hypo in hypothesis:
    print_hypothesis_matrix(hypo, len(T), likelyhood)
