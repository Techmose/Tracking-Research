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

#Tests follow the following naming convetion Test / # of tracks / # of measurements / Variable
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

def test_one_two_birth_vs_clutter():
    tracks = [5]
    measurements = [5, 12]  # second measurement far away
    sigma = 0.5
    prob_detection = 0.9

    hyp_birth = Hypothesis(tracks, measurements, sigma, lam_birth=10, lam_clutter=0.1, prob_detection=prob_detection)
    hyp_clutter = Hypothesis(tracks, measurements, sigma, lam_birth=0.1, lam_clutter=10, prob_detection=prob_detection)

    best_birth = hyp_birth.best_hypothesis()
    best_clutter = hyp_clutter.best_hypothesis()

    # With high birth rate, unused meas is marked as birth
    assert hyp_birth.M in best_birth, f"Expected birth assignment, got {best_birth}"

    # With high clutter rate, unused meas is marked as clutter
    assert hyp_clutter.M+1 in best_clutter, f"Expected clutter assignment, got {best_clutter}"

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

def test_two_two_number_of_hypotheses():
    tracks = [0, 10]
    measurements = [1, 11]
    sigma = 1
    lam_birth = 1
    lam_clutter = 1
    prob_detection = 0.8

    hyp = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)

    # At least 1 hypothesis, at most (M+2)^M (rough bound)
    assert 1 <= len(hyp.hypotheses) <= (hyp.M + 2)**hyp.M, \
        f"Unexpected number of hypotheses: {len(hyp.hypotheses)}"

def test_two_three_sigma_effect():
    tracks = [2, 8]
    measurements = [2, 8, 15]
    lam_birth = 1
    lam_clutter = 1
    prob_detection = 0.9

    hyp_small_sigma = Hypothesis(tracks, measurements, sigma=0.1,
                                 lam_birth=lam_birth, lam_clutter=lam_clutter, prob_detection=prob_detection)
    hyp_large_sigma = Hypothesis(tracks, measurements, sigma=10,
                                 lam_birth=lam_birth, lam_clutter=lam_clutter, prob_detection=prob_detection)

    best_small = hyp_small_sigma.best_hypothesis()
    best_large = hyp_large_sigma.best_hypothesis()

    # With small sigma, best hypothesis should assign meas0→track0, meas1→track1
    assert best_small[:2] == (0, 1), f"Expected tracks matched tightly, got {best_small}"
    # With large sigma, looser matching: clutter/birth is more likely
    assert best_large[:2] != (-1, -1), f"Expected non-trivial matching for large sigma"

def test_three_three_exact_sigma():
    tracks = [5, 12, 18]
    measurements = [5, 12, 18]
    sigma = .01
    sigma1 = .1
    sigma2 = 1
    sigma3 = 10
    sigma4 = 20
    sigma5 = 40
    lam_birth = .2
    lam_clutter = .2
    prob_detection = .9
# Scores w/ exact t=m (10.342613411376778, 3.4348581323946394, -3.4728971465874974, -10.380652425569632, -17.28840770455177)
    hyp_class = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)
    hyp_class.score_all_hypotheses()
    best_hyp = hyp_class.best_hypothesis()
    score = hyp_class.score_hypothesis_log(best_hyp)
    hyp_class1 = Hypothesis(tracks, measurements, sigma1, lam_birth, lam_clutter, prob_detection)
    hyp_class1.score_all_hypotheses()
    best_hyp1 = hyp_class1.best_hypothesis()
    score1 = hyp_class1.score_hypothesis_log(best_hyp1)
    hyp_class2 = Hypothesis(tracks, measurements, sigma2, lam_birth, lam_clutter, prob_detection)
    hyp_class2.score_all_hypotheses()
    best_hyp2 = hyp_class2.best_hypothesis()
    score2 = hyp_class2.score_hypothesis_log(best_hyp2)
    hyp_class3 = Hypothesis(tracks, measurements, sigma3, lam_birth, lam_clutter, prob_detection)
    hyp_class3.score_all_hypotheses()
    best_hyp3 = hyp_class3.best_hypothesis()
    score3 = hyp_class3.score_hypothesis_log(best_hyp3)
    hyp_class4 = Hypothesis(tracks, measurements, sigma4, lam_birth, lam_clutter, prob_detection)
    hyp_class4.score_all_hypotheses()
    best_hyp4 = hyp_class4.best_hypothesis()
    score4 = hyp_class4.score_hypothesis_log(best_hyp4)
    hyp_class5 = Hypothesis(tracks, measurements, sigma5, lam_birth, lam_clutter, prob_detection)
    hyp_class5.score_all_hypotheses()
    best_hyp5 = hyp_class5.best_hypothesis()
    score5 = hyp_class5.score_hypothesis_log(best_hyp5)

    expected_best_hypothesis = (0, 1, 2, 0, 1, 2)
    expected_best_hypothesis1 = (0, 1, 2, 0, 1, 2)
    expected_best_hypothesis2 = (0, 1, 2, 0, 1, 2)
    expected_best_hypothesis3 = (0, 1, 2, 0, 1, 2)
    expected_best_hypothesis4 = (-1, -1, 2, 3, 4, 2)
    expected_best_hypothesis5 = (-1, -1, -1, 3, 3, 4)

    assert best_hyp == expected_best_hypothesis, \
        f"Expected track assignment {(expected_best_hypothesis)}, got {best_hyp}"
    assert best_hyp1 == expected_best_hypothesis1, \
        f"Expected track assignment {(expected_best_hypothesis1)}, got {best_hyp1}"
    assert best_hyp2 == expected_best_hypothesis2, \
        f"Expected track assignment {(expected_best_hypothesis2)}, got {best_hyp2}"
    assert best_hyp3 == expected_best_hypothesis3, \
        f"Expected track assignment {(expected_best_hypothesis3)}, got {best_hyp3}"
    assert best_hyp4 == expected_best_hypothesis4, \
        f"Expected track assignment {(expected_best_hypothesis4)}, got {best_hyp4}"
    assert best_hyp5 == expected_best_hypothesis5, \
        f"Expected track assignment {(expected_best_hypothesis5)}, got {best_hyp5}"
    #assert score == score1 == score2 == score3 == score4, \
    #    f"Expeced scores to be the same: {(score)}, {(score1)}, {(score2)}, {(score3)}, {(score4)}"

def test_three_three_sigma():
    tracks = [5, 12, 18]
    measurements = [5.1, 12.1, 18.1]
    sigma = .01
    sigma1 = .1
    sigma2 = 1
    sigma3 = 10
    sigma4 = 100
    lam_birth = .2
    lam_clutter = .2
    prob_detection = .9

    hyp_class = Hypothesis(tracks, measurements, sigma, lam_birth, lam_clutter, prob_detection)
    hyp_class.score_all_hypotheses()
    best_hyp = hyp_class.best_hypothesis()
    score = hyp_class.score_hypothesis_log(best_hyp)
    hyp_class1 = Hypothesis(tracks, measurements, sigma1, lam_birth, lam_clutter, prob_detection)
    hyp_class1.score_all_hypotheses()
    best_hyp1 = hyp_class1.best_hypothesis()
    score1 = hyp_class1.score_hypothesis_log(best_hyp1)
    hyp_class2 = Hypothesis(tracks, measurements, sigma2, lam_birth, lam_clutter, prob_detection)
    hyp_class2.score_all_hypotheses()
    best_hyp2 = hyp_class2.best_hypothesis()
    score2 = hyp_class2.score_hypothesis_log(best_hyp2)
    hyp_class3 = Hypothesis(tracks, measurements, sigma3, lam_birth, lam_clutter, prob_detection)
    hyp_class3.score_all_hypotheses()
    best_hyp3 = hyp_class3.best_hypothesis()
    score3 = hyp_class3.score_hypothesis_log(best_hyp3)
    hyp_class4 = Hypothesis(tracks, measurements, sigma4, lam_birth, lam_clutter, prob_detection)
    hyp_class4.score_all_hypotheses()
    best_hyp4 = hyp_class4.best_hypothesis()
    score4 = hyp_class4.score_hypothesis_log(best_hyp4)

    expected_best_hypothesis = (-1, -1, -1, 3, 3, 4)
    expected_best_hypothesis1 = (0, 1, 2, 0, 1, 2)
    expected_best_hypothesis2 = (0, 1, 2, 0, 1, 2)
    expected_best_hypothesis3 = (0, 1, 2, 0, 1, 2)
    expected_best_hypothesis4 = (-1, -1, -1, 3, 3, 4)

    assert best_hyp == expected_best_hypothesis, \
        f"Expected track assignment {(expected_best_hypothesis)}, got {best_hyp}"
    assert best_hyp1 == expected_best_hypothesis1, \
        f"Expected track assignment {(expected_best_hypothesis1)}, got {best_hyp1}"
    assert best_hyp2 == expected_best_hypothesis2, \
        f"Expected track assignment {(expected_best_hypothesis2)}, got {best_hyp2}"
    assert best_hyp3 == expected_best_hypothesis3, \
        f"Expected track assignment {(expected_best_hypothesis3)}, got {best_hyp3}"
    assert best_hyp4 == expected_best_hypothesis4, \
        f"Expected track assignment {(expected_best_hypothesis4)}, got {best_hyp4}"
    #assert score == score1 == score2 == score3 == score4, \
    #    f"Expeced scores to be the same: {(score)}, {(score1)}, {(score2)}, {(score3)}, {(score4)}"
