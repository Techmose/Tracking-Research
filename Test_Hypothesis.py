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

#sigma = standard deviation of measurements
#lam_birth = mean number of births expected
#lam_clutter = mean number of clutter          

#Tests follow the following naming convetion Test / # of tracks / # of measurements / Expected Outcome
def test_zero_one_birth():
    tracks = []
    measurements = [0.1]
    sigma = 0.1
    lam_birth = 0.2
    lam_clutter = 0.1
    prob_detection = 0.9

    hyp_class = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)
    hyp_class.score_all_hypotheses()
    best_hyp = hyp_class.best_hypothesis()

    expected_best_hypothesis = tuple((0, ))

    assert best_hyp == expected_best_hypothesis, \
        f"Expected track assignment {(expected_best_hypothesis)}, got {best_hyp}"
    
def test_zero_five_():
    tracks = []
    measurements = [1,2,3,4,5]
    sigma = 0.1
    lam_birth = .5
    lam_clutter = 2
    prob_detection = 1.0

    hyp_class = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)
    hyp_class.score_all_hypotheses()
    best_hyp = hyp_class.best_hypothesis()

    expected_best_hypothesis = (0,1,1,1,1)

    assert best_hyp == expected_best_hypothesis, \
        f"Expected track assignment {(expected_best_hypothesis)}, got {best_hyp}"   

def test_zero_one_clutter():
    tracks = []
    measurements = [0.1]
    sigma = 0.1
    lam_birth = 0.1
    lam_clutter = 0.2
    prob_detection = 0.9

    hyp_class = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)
    hyp_class.score_all_hypotheses()
    best_hyp = hyp_class.best_hypothesis()

    expected_best_hypothesis = tuple((1, ))

    assert best_hyp == expected_best_hypothesis, \
        f"Expected track assignment {(expected_best_hypothesis)}, got {best_hyp}"

def test_one_one_sigma(): #Work on this case does sigma create noise and make birth/clutter morelikey and How does prob_detection change this
    tracks = [3]
    measurements = [3]
    sigma = 8
    lam_birth = 0.5
    lam_clutter = 0.5
    prob_detection = 0.9093095 #.9093 is a birth

    hyp_class = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)
    hyp_class.score_all_hypotheses()
    best_hyp = hyp_class.best_hypothesis()

    expected_best_hypothesis = (0, 0)

    assert best_hyp == expected_best_hypothesis, \
        f"Expected track assignment {(expected_best_hypothesis)}, got {best_hyp}"

def test_two_two_assign_all():
    # Simple test case with 2 tracks and 2 measurements
    tracks = [0, 1]
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
    
def test_two_two_prob_detection(): #extra cases change lam to .1 or 10
    tracks = [1, 3]
    measurements = [1, 3]
    sigma = 0.1
    lam_birth = 1
    lam_clutter = 1
    prob_detection = .2 
    prob_detection_ = .3

    hyp_class = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)
    hyp_class_ = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection_)

    hyp_class.score_all_hypotheses()
    best_hyp = hyp_class.best_hypothesis()
    hyp_class_.score_all_hypotheses()
    best_hyp_ = hyp_class_.best_hypothesis()

    expected_best_hypothesis_ = (0, 1, 0, 1)
    expected_best_hypothesis = (-1, -1, 2, 3)

    assert best_hyp_ == expected_best_hypothesis_, \
        f"Expected track assignment {(expected_best_hypothesis_)}, got {best_hyp_}"
    assert best_hyp == expected_best_hypothesis, \
        f"Expected track assignment {(expected_best_hypothesis)}, got {best_hyp}"
    
