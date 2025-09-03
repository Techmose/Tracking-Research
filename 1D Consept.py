import itertools
import numpy as np

#Tracks
T = [5, 9]

#Readings
Z = [4, 8, 15]

#Tn = New track / Tf = false alarm
#Matrix represntation of Tracks 
#   Tn T1 T2 Tf
#Z1 [1, 1, 0, 1]
#Z2 [1, 1, 1, 1]
#Z3 [1, 0, 1, 1]

#Likelyhood matrix is the probability of reading z_i given track t_i
def CreateLikelyhoodMatrix(tracks, readings, sigma = 1):
    tracks = np.array(tracks, dtype=float).reshape(-1, 1)   # shape (N,1)
    readings = np.array(readings, dtype=float).reshape(1, -1)  # shape (1,M)
    diff = readings - tracks   # shape (N,M)
    coeff = 1.0 / (np.sqrt(2 * np.pi * sigma**2))
    L = coeff * np.exp(-0.5 * (diff**2) / (sigma**2))
    return L

def Likelyhood_without_gaussian(tracks,readings):
    tracks = np.array(tracks, dtype=float).reshape(-1, 1)   # shape (N,1)
    readings = np.array(readings, dtype=float).reshape(1, -1)  # shape (1,M)
    D = np.max(tracks) - np.min(tracks)
    diff = np.abs(readings - tracks)   # shape (N,M)

    L = np.where(diff <= D, 1 - diff / D, 0.0)
    return L
                

def generate_hypotheses(L, P_D=0.9, lambda_clutter=0.1, rho_birth=0.1):
    N, M = L.shape
    meas_indices = range(M)

    results = []

    # Each measurement can be assigned to: one of N tracks, BIRTH, or CLUTTER
    choices = range(N + 2)  # 0..N-1=tracks, N=birth, N+1=clutter

    for assignment in itertools.product(choices, repeat=M):
        # enforce: each track at most one measurement
        used_tracks = set()
        valid = True
        for j, choice in enumerate(assignment):
            if choice < N:  # real track
                if choice in used_tracks:
                    valid = False
                    break
                used_tracks.add(choice)
        if not valid:
            continue
        
        if len(used_tracks) != N:
            continue

        # Build probability
        prob = 1.0
        track_assignments = {i: None for i in range(N)}
        births, clutter = [], []

        for j, choice in enumerate(assignment):
            if choice < N:  # track i
                track_assignments[choice] = j
                prob *= P_D * L[choice, j]
            elif choice == N:  # birth
                births.append(j)
                prob *= rho_birth
            else:  # clutter
                clutter.append(j)
                prob *= lambda_clutter

        # missed detections
        for i in range(N):
            if track_assignments[i] is None:
                prob *= (1 - P_D)

        results.append({
            "assignments": track_assignments,
            "births": births,
            "clutter": clutter,
            "prob": prob
        })

    # normalize probabilities
    total = sum(r["prob"] for r in results)
    for r in results:
        r["prob"] /= total if total > 0 else 1.0

    return results

Likelyhood = CreateLikelyhoodMatrix(T, Z)
No_gaussian = Likelyhood_without_gaussian(T,Z)
print(No_gaussian)
print(Likelyhood)
#print(generate_hypotheses(Likelyhood))
#print((generate_hypotheses(No_gaussian)))