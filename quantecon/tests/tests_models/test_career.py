"""
Tests for quantecon.carrer module

@author : Spencer Lyon
@date : 2014-07-31

"""
from __future__ import division
import unittest
import numpy as np
from quantecon.models import CareerWorkerProblem


class TestCareerWorkerProblem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cp = CareerWorkerProblem()
        cls.v_init = np.random.rand(cls.cp.N, cls.cp.N)
        cls.v_prime = cls.cp.bellman_operator(cls.v_init)
        cls.greedy = cls.cp.get_greedy(cls.v_init)

    def test_bellman_shape(self):
        "career: bellman shape"
        assert self.v_init.shape == self.v_prime.shape

    def test_greedy_shape(self):
        "career: greedy shape"
        assert self.v_init.shape == self.greedy.shape

    def test_greedy_new_life(self):
        "career: want new life with worst job/career?"
        if (self.greedy == 3).any():
            # if we ever want a new life, it will be with worst possible
            # theta and worst epsilon
            assert self.greedy[0, 0] == 3

    def test_greedy_new_job(self):
        "career: want new job with best carrer/worst job?"
        # we should want a new job with best career and worst job
        assert self.greedy[-1, 0] == 2

    def test_greedy_stay_put(self):
        "career: want to stayw with best career/job?"
        if (self.greedy == 1).any():
            # if we ever want to stay put, it will be with best possible
            # theta and best epsilon
            assert self.greedy[-1, -1] == 1
