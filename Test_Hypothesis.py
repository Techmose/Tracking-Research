import math
import numpy
from OneD_Consept import Hypothesis

#This .py file tests the class hypothesis from OneD_Consept

#A hypothesis is a tuple of length N+M where N is the number of tracks and M is the number of measurements
#The first N are the assignments for the tracks Followed by the Measurements Assignments
    #        each entry is:
    #        0..N-1 => track index
    #        N      => birth
    #        N+1    => clutter
    #        -1     => missed track

def test_most_probable_hypothesis_simple():
    # Simple test case with 2 tracks and 2 measurements
    tracks = [0.0, 1.0]
    measurements = [0.1, 1.1]
    sigma = 0.1
    lam_birth = 0.1
    lam_clutter = 0.1
    prob_detection = 0.9

    hyp_class = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)
    
    # Score all hypotheses
    hyp_class.score_all_hypotheses()
    
    # Get best hypothesis
    best_hyp = hyp_class.best_hypothesis()
    
    # Expected: measurements should match tracks (minimal likelihood error)
    # For this setup, we expect track 0 → measurement 0, track 1 → measurement 1
    # Depending on how you generate hypotheses, the tuple could be something like:
    expected_best_hypothesis = (0, 1, 0, 1)  # track assignments + measurement labels
    
    assert best_hyp == expected_best_hypothesis, \
        f"Expected track assignment {expected_best_hypothesis}, got {best_hyp}"