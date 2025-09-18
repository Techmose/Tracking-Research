import itertools
import math
import numpy as np

class Hypothesis:
    def __init__(self, tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection):
        self.tracks = np.array(tracks, dtype=float)
        self.measurements = np.array(measurements, dtype=float)
        self.sigma = sigma
        self.lam_birth = lam_birth
        self.lam_clutter = lam_clutter
        self.prob_detection = prob_detection

        # Derived attributes
        self.N = len(self.tracks)
        self.M = len(self.measurements)
        self.L = self.create_likelihood_matrix()

        # Store hypotheses as list of tuples
        self.hypotheses = self.generate_hypotheses()

    def create_likelihood_matrix(self):
        tracks = self.tracks.reshape(-1, 1)
        readings = self.measurements.reshape(1, -1)
        diff = readings - tracks
        coeff = 1.0 / (np.sqrt(2 * np.pi * self.sigma**2))
        L = coeff * np.exp(-0.5 * (diff**2) / (self.sigma**2))
        return L
    
    #Poisson function for births or clutter given lambda: (lam)mean expected birth/clutter and k: number of events
    def poisson_prob(self, k, lam):
        return -lam + k * math.log(lam) - math.lgamma(k + 1)
    
    def generate_hypotheses(self):
        """
        Generate all feasible hypotheses (assignments only, no probabilities)
        using the class attributes self.N and self.M.
        Returns a list of tuples with track assignments followed by measurement labels.
        """
        choices = [-1] + list(range(self.M))
        results = []

        for track_assignment in itertools.product(choices, repeat=self.N):
            used = [j for j in track_assignment if j >= 0]
            if len(set(used)) != len(used):
                continue  # skip duplicate measurement assignment

            unused = [j for j in range(self.M) if j not in used]

            # Enumerate all birth/clutter patterns for unused measurements
            for bc_pattern in itertools.product([self.N, self.N + 1], repeat=len(unused)):
                measurement_labels = [None] * self.M

                # Fill used measurements with track index
                for t_idx, m_idx in enumerate(track_assignment):
                    if m_idx >= 0:
                        measurement_labels[m_idx] = t_idx

                # Fill unused measurements with birth/clutter
                for u_idx, label in zip(unused, bc_pattern):
                    measurement_labels[u_idx] = label

                results.append(track_assignment + tuple(measurement_labels))

        return results
    
    def hypothesis_to_matrix(self, hypothesis):
        """
        Convert a single hypothesis tuple into a binary assignment matrix with
        an extra row for missed tracks (ZM). Columns: [T0..TN-1, Tbirth, Tclutter]
        """
        track_assignments = hypothesis[:self.N]
        measurement_labels = hypothesis[self.N:]

        # Initialize matrix with an extra row for missed tracks
        matrix = np.zeros((self.M + 1, self.N + 2), dtype=int)

        # Fill measurement assignments
        for m_idx, label in enumerate(measurement_labels):
            if label < self.N:
                matrix[m_idx, label] = 1  # assigned to track
            elif label == self.N:
                matrix[m_idx, self.N] = 1  # birth
            elif label == self.N + 1:
                matrix[m_idx, self.N + 1] = 1  # clutter

        # Fill missed tracks row (ZM)
        for t_idx, m_idx in enumerate(track_assignments):
            if m_idx == -1:  # missed track
                matrix[self.M, t_idx] = 1

        return matrix

    def score_hypothesis_log(self, hypothesis):
        """
        Compute log-probability (unnormalized) of a single hypothesis using the class attributes.
        """
        N, M = self.N, self.M
        L = self.L
        P_D = self.prob_detection
        lambda_birth = self.lam_birth
        lambda_clutter = self.lam_clutter

        measurement_labels = hypothesis[N:]  # last M elements
        log_w = 0.0
        used_tracks = set()

        # Count births and clutter
        births = sum(1 for ch in hypothesis if ch == N)
        clutter = sum(1 for ch in hypothesis if ch == N + 1)

        for j, choice in enumerate(measurement_labels):
            if choice < N:
                used_tracks.add(choice)
                if L[choice, j] <= 0:
                    return float("-inf")  # impossible assignment
                log_w += math.log(P_D) + math.log(L[choice, j])

        # Missed tracks
        for track in range(N):
            if track not in used_tracks:
                log_w += math.log(1 - P_D)

        # Add Poisson log-probabilities
        log_w += self.poisson_prob(births, lambda_birth)
        log_w += self.poisson_prob(clutter, lambda_clutter)

        return log_w
    
    def score_all_hypotheses(self):
        """Compute log-scores for all stored hypotheses."""
        self.hypothesis_scores = [self.score_hypothesis_log(h) for h in self.hypotheses]
        return self.hypothesis_scores

    def best_hypothesis(self):
        """Return the hypothesis with the highest log-score."""
        if not hasattr(self, "hypothesis_scores"):
            self.score_all_hypotheses()
        idx = np.argmax(self.hypothesis_scores)
        return self.hypotheses[idx]
    
    #print top k hypotheisand prob


###The above class should cover everything below, The below was to start off and will be deleated after I'm confident in the class
#Tracks
T = [5, 9]

#Readings
Z = [4, 8, 15]


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
#ZM [0, 0, 0, 0]Missing measurement Figure this out

def create_log_likelihood_matrix(tracks, readings, sigma=1.0):
    tracks = np.array(tracks, dtype=float).reshape(-1, 1)   # shape (N,1)
    readings = np.array(readings, dtype=float).reshape(1, -1)  # shape (1,M)
    diff = readings - tracks  # shape (N,M)

    log_coeff = -0.5 * math.log(2 * math.pi * sigma**2)
    log_L = log_coeff - 0.5 * (diff**2) / (sigma**2)

    return log_L

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
            -1     => missed track

        Enforces: each track is used at most once
                and all tracks must be assigned (no misses).
        """
        choices = [-1] + list(range(M))
        results = []

        for track_assignment in itertools.product(choices, repeat=N):
            used = [j for j in track_assignment if j >= 0]
            if len(set(used)) != len(used):
                continue  # skip duplicate measurement assignment

            unused = [j for j in range(M) if j not in used]

            # Enumerate all birth/clutter patterns for unused measurements
            for bc_pattern in itertools.product([N, N+1], repeat=len(unused)):
                measurement_labels = [None] * M

                # Fill used measurements with track index
                for t_idx, m_idx in enumerate(track_assignment):
                    if m_idx >= 0:
                        measurement_labels[m_idx] = t_idx

                # Fill unused measurements with birth/clutter
                for u_idx, label in zip(unused, bc_pattern):
                    measurement_labels[u_idx] = label

                results.append(track_assignment + tuple(measurement_labels))

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

#likelyhood = create_log_likelihood_matrix(T, Z)
#no_gaussian = likelyhood_without_gaussian(T,Z)

#hypothesis = generate_hypothesis_assignments(len(T) , len(Z))
#for hypo in hypothesis:
#    print_hypothesis_matrix(hypo, len(T), likelyhood)

#print(hypothesis)
