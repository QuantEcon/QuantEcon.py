"""
Tests for mc_tools.py

Functions
---------
	mc_compute_stationary 	[Status: 1 x Simple Test Written]
	mc_sample_path 			[Status: TBD]

Notes
-----
[1] There is currently a section in this test file which contains the examples used for the Wiki Page: "Testing: Writing Tests"
	(Marked Below). This is technically running 3 x of the Same Tests
"""

from __future__ import division

import numpy as np
import unittest
from numpy.testing import assert_allclose

from ..mc_tools import mc_compute_stationary

### Tests: mc_compute_stationary ###

def test_mc_compute_stationary_pmatrices():
	"""
		Test mc_compute_stationary with P Matrix and Known Solutions
	"""

					#-P Matrix-#						, #-Known Solution-#
	testset = 	[ 																		
					( np.array([[0.4,0.6], [0.2,0.8]]) 	, np.array([0.25, 0.75]) ), 	
				]

	#-Loop Through TestSet-#
	for (P, known) in testset:
		computed = mc_compute_stationary(P)
		assert_allclose(computed, known)



### Tests: mc_sample_path ###


	# - Work Needed Here - #



####################################
# Wiki Example  				   #
# ------------					   #
# Tests For: mc_compute_stationary #
# Note: This can be removed but is #
# a record of examples written for #
# "Wiki: Writing Tests" 		   #
####################################

#--------------------------------#
#-Examples: Simple Test Examples-#
#--------------------------------#

# Check required infrastructure is imported
import numpy as np

# Check that the test_mc_tools.py file has imported the relevant function we wish to test: mc_compute_stationary
# Note: This will import the function from the `installed` location (may want to use relative references)

from ..mc_tools import mc_compute_stationary
#from quantecon import mc_compute_stationary

#-Example of Assertion Failures-#

	# from numpy.testing import assert_array_equal

	# def test_mc_compute_stationary_pmatrix():
	# 	"""
	# 	Test for a Known Solution 
	# 	Module:     mc_tools.py 
	# 	Function:   mc_compute_stationary
	# 	"""
	# 	P = np.array([[0.4,0.6], [0.2,0.8]])
	# 	P_known = np.array([0.25, 0.75])
	# 	computed = mc_compute_stationary(P)
	# 	assert_array_equal(computed, P_known)

# Example Output from Above Test
# ---------------------------------------------------------------------------
# AssertionError                            Traceback (most recent call last)
# #Traceback details are presented here

# AssertionError: 
# Arrays are not equal

# (mismatch 50.0%)
#  x: array([ 0.25,  0.75])
#  y: array([ 0.25,  0.75])

from numpy.testing import assert_allclose

def test_mc_compute_stationary_pmatrix():
	"""
	Test mc_compute_stationary for a Known Solution of Matrix P
	Module:     mc_tools.py 
	Function:   mc_compute_stationary
	"""
	P = np.array([[0.4,0.6], [0.2,0.8]])
	P_known = np.array([0.25, 0.75])
	computed = mc_compute_stationary(P)
	assert_allclose(computed, P_known)

#-Slightly More General Version with testset-#

def test_mc_compute_stationary_pmatrix():
	testset1 = (np.array([[0.4,0.6], [0.2,0.8]]), np.array([0.25, 0.75]))      	
	check_mc_compute_stationary_pmatrix(testset1)

def check_mc_compute_stationary_pmatrix(testset):
	"""
	Test mc_compute_stationary for a Known Solution of Matrix P
	Module:     mc_tools.py 
	Function:   mc_compute_stationary
	
	Arguments
	---------
	[1] test_set 	: 	tuple(np.array(P), np.array(known_solution))
	"""
	(P, known) = testset
	computed = mc_compute_stationary(P)
	assert_allclose(computed, known)

#-----------------------------------------------------------------#
#-Examples: Unittest vs Nose Functions (with setup) vs Nose Class-#
#-----------------------------------------------------------------#

from ..mc_tools import mc_compute_stationary                # Good to use relative imports so as not to use the system installation quantecon when running tests

# Supporting Test Function #
# Some Tests will require Setup Functions to Generate a Type of Matrix etc
# These can be Imported or Defined In the Test File as a function
# Example From: https://github.com/oyamad/test_mc_compute_stationary

def KMR_Markov_matrix_sequential(N, p, epsilon):
	"""
	Generate the Markov matrix for the KMR model with *sequential* move

	N: number of players
	p: level of p-dominance for action 1
	   = the value of p such that action 1 is the BR for (1-q, q) for any q > p,
		 where q (1-q, resp.) is the prob that the opponent plays action 1 (0, resp.)
	epsilon: mutation probability

	References: 
		KMRMarkovMatrixSequential is contributed from https://github.com/oyamad
	"""
	P = np.zeros((N+1, N+1), dtype=float)
	P[0, 0], P[0, 1] = 1 - epsilon * (1/2), epsilon * (1/2)
	for n in range(1, N):
		P[n, n-1] = \
			(n/N) * (epsilon * (1/2) +
					 (1 - epsilon) * (((n-1)/(N-1) < p) + ((n-1)/(N-1) == p) * (1/2))
					 )
		P[n, n+1] = \
			((N-n)/N) * (epsilon * (1/2) +
						 (1 - epsilon) * ((n/(N-1) > p) + (n/(N-1) == p) * (1/2))
						 )
		P[n, n] = 1 - P[n, n-1] - P[n, n+1]
	P[N, N-1], P[N, N] = epsilon * (1/2), 1 - epsilon * (1/2)
	return P

##############
## unittest ##
##############

# Pro: Consistent Naming Convention, Benefits of inheritance from unittest.TestCase in terms of assertion logic, Nose can parse unittest
# Con: Can be a bit less flexible, Larger Initial Barier to Entry for Beginners

class Test_mc_compute_stationary_KMRMarkovMatrix1(unittest.TestCase):
	"""
	Test Suite for mc_compute_stationary using KMR Markov Matrix [using unittest.TestCase]
	"""

	# Starting Values #

	N = 27
	epsilon = 1e-2
	p = 1/3
	TOL = 1e-2

	def setUp(self):
		""" Calculate KMR Matrix and Compute the Stationary Distribution """
		self.P = KMR_Markov_matrix_sequential(self.N, self.p, self.epsilon)
		self.v = mc_compute_stationary(self.P)

	def test_markov_matrix(self):
		for i in range(len(self.P)):
			self.assertEqual(sum(self.P[i, :]), 1)

	def test_sum_one(self):
		self.assertTrue(np.allclose(sum(self.v), 1, atol=self.TOL))

	def test_nonnegative(self):
		self.assertEqual(np.prod(self.v >= 0-self.TOL), 1)

	def test_left_eigen_vec(self):
		self.assertTrue(np.allclose(np.dot(self.v, self.P), self.v, atol=self.TOL))

	def tearDown(self):
		pass


####################
## Nose Functions ##
####################

# Nose offers a variety of ways to construct tests: Functions, Classes, etc.

# Individual Test Functions with setup decorator #
# ---------------------------------------------- #

from nose.tools import with_setup

#-Starting Values-#

N = 27
epsilon = 1e-2
p = 1/3
TOL = 1e-2

def setup_func():
	""" Setup a KMRMarkovMatrix and Compute Stationary Values """
	global P                                            # Not Usually Recommended!
	P = KMR_Markov_matrix_sequential(N, p, epsilon)
	global v                                            # Not Usually Recommended!
	v = mc_compute_stationary(P)

@with_setup(setup_func)
def test_markov_matrix():
	for i in range(len(P)):
		assert sum(P[i, :]) == 1, "sum(P[i,:]) %s != 1" % sum(P[i, :])

@with_setup(setup_func)
def test_sum_one():
	assert np.allclose(sum(v), 1, atol=TOL) == True, "np.allclose(sum(v), 1, atol=%s) != True" % TOL

@with_setup(setup_func)
def test_nonnegative():
	assert np.prod(v >= 0-TOL) == 1, "np.prod(v >= 0-TOL) %s != 1" % np.prod(v >= 0-TOL)

@with_setup(setup_func)
def test_left_eigen_vec():
	assert np.allclose(np.dot(v, P), v, atol=TOL) == True, "np.allclose(np.dot(v, P), v, atol=%s) != True" % TOL

# Basic Class Structure with Setup #
####################################

class Test_mc_compute_stationary_KMRMarkovMatrix2():
	"""
	Test Suite for mc_compute_stationary using KMR Markov Matrix [suitable for nose]
	"""

	#-Starting Values-#

	N = 27
	epsilon = 1e-2
	p = 1/3
	TOL = 1e-2

	def setUp(self):
		""" Setup a KMRMarkovMatrix and Compute Stationary Values """
		self.P = KMR_Markov_matrix_sequential(self.N, self.p, self.epsilon)
		self.v = mc_compute_stationary(self.P)

	def test_markov_matrix(self):
		for i in range(len(self.P)):
			assert sum(self.P[i, :]) == 1, "sum(P[i,:]) %s != 1" % sum(self.P[i, :])

	def test_sum_one(self):
		assert np.allclose(sum(self.v), 1, atol=self.TOL) == True, "np.allclose(sum(v), 1, atol=%s) != True" % self.TOL

	def test_nonnegative(self):
		assert np.prod(self.v >= 0-self.TOL) == 1, "np.prod(v >= 0-TOL) %s != 1" % np.prod(self.v >= 0-self.TOL)

	def test_left_eigen_vec(self):
		assert np.allclose(np.dot(self.v, self.P), self.v, atol=self.TOL) == True, "np.allclose(np.dot(v, P), v, atol=%s) != True" % self.TOL


