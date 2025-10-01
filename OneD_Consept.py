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
    
    def best_k_hypotheses(self, k):
        if not hasattr(self, "hypothesis_scores"):
            self.score_all_hypotheses()

        scores = np.array(self.hypothesis_scores)
        top_idx = np.argsort(scores)[::-1][:k]  # sort descending, take top k

        return [self.hypotheses[i] for i in top_idx]
    
    def print_hypothesis_matrix(self, hypothesis):
        
        log_score = self.score_hypothesis_log(hypothesis)  # raw log-score

        N = self.N
        M = self.M

        mat = np.zeros((M + 1, N + 2), dtype=int)

        # Fill based on measurement labels
        track_assignments = hypothesis[:N]
        meas_labels = hypothesis[N:]
        for j, label in enumerate(meas_labels):
            if label < N:
                mat[j, label] = 1
            elif label == N:
                mat[j, N] = 1  # birth column
            elif label == N+1:
                mat[j, N+1] = 1  # clutter column

        # Add missed tracks (Zm column)
        for t_idx, assignment in enumerate(track_assignments):
            if assignment == -1:
                mat[M, t_idx] = 1

        # Print
        col_labels = [f"T{i}" for i in range(N)] + ["Tb", "Tc"]
        row_labels = [f"Z{j}" for j in range(M)] + ["Zm"]

        print(f"Hypothesis: {hypothesis}")
        print(f"Log-Score: {log_score:.4f}")  # raw log probability
        print("    " + "  ".join(col_labels))
        for r, row in zip(row_labels, mat):
            print(f"{r}  " + "  ".join(str(x) for x in row))
    
    def print_top_k_prob(self, k):
        top_k = self.best_k_hypotheses(k)

        for hypotheis in top_k:
            self.print_hypothesis_matrix(hypotheis)
        

tracks = [3]
measurements = [3]
sigma = 1
lam_birth = 0.1
lam_clutter = 0.1
prob_detection = 0.9093095 #.9093 is a birth

#hyp_class = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)
#print(hyp_class.hypotheses)
#hyp_class.print_hypothesis_matrix((0,0))
#hyp_class.print_hypothesis_matrix((-1,1))
#hyp_class.print_hypothesis_matrix((-1,2))
#hyp_class.print_top_k_prob(3)
