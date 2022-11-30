r"""
Module for solving dynamic programs (also known as Markov decision
processes) with finite states and actions.

Discrete Dynamic Programming
----------------------------

A discrete dynamic program consists of the following components:

* finite set of states :math:`S = \{0, \ldots, n-1\}`;
* finite set of available actions :math:`A(s)` for each state :math:`s
  \in S` with :math:`A = \bigcup_{s \in S} A(s) = \{0, \ldots, m-1\}`,
  where :math:`\mathit{SA} = \{(s, a) \in S \times A \mid a \in A(s)\}`
  is the set of feasible state-action pairs;
* reward function :math:`r\colon \mathit{SA} \to \mathbb{R}`, where
  :math:`r(s, a)` is the reward when the current state is :math:`s` and
  the action chosen is :math:`a`;
* transition probability function :math:`q\colon \mathit{SA} \to
  \Delta(S)`, where :math:`q(s'|s, a)` is the probability that the state
  in the next period is :math:`s'` when the current state is :math:`s`
  and the action chosen is :math:`a`; and
* discount factor :math:`0 \leq \beta < 1`.

For a policy function :math:`\sigma`, let :math:`r_{\sigma}` and
:math:`Q_{\sigma}` be the reward vector and the transition probability
matrix for :math:`\sigma`, which are defined by :math:`r_{\sigma}(s) =
r(s, \sigma(s))` and :math:`Q_{\sigma}(s, s') = q(s'|s, \sigma(s))`,
respectively. The policy value function :math:`v_{\sigma}` for
:math:`\sigma` is defined by

.. math::

    v_{\sigma}(s) = \sum_{t=0}^{\infty}
                    \beta^t (Q_{\sigma}^t r_{\sigma})(s)
                    \quad (s \in S).

The *optimal value function* :math:`v^*` is the function such that
:math:`v^*(s) = \max_{\sigma} v_{\sigma}(s)` for all :math:`s \in S`. A
policy function :math:`\sigma^*` is *optimal* if :math:`v_{\sigma^*}(s)
= v^*(s)` for all :math:`s \in S`.

The *Bellman equation* is written as

.. math::

    v(s) = \max_{a \in A(s)} r(s, a)
           + \beta \sum_{s' \in S} q(s'|s, a) v(s') \quad (s \in S).

The *Bellman operator* :math:`T` is defined by the right hand side of
the Bellman equation:

.. math::

    (T v)(s) = \max_{a \in A(s)} r(s, a)
               + \beta \sum_{s' \in S} q(s'|s, a) v(s') \quad (s \in S).

For a policy function :math:`\sigma`, the operator :math:`T_{\sigma}` is
defined by

.. math::

    (T_{\sigma} v)(s) = r(s, \sigma(s))
                        + \beta \sum_{s' \in S} q(s'|s, \sigma(s)) v(s')
                        \quad (s \in S),

or :math:`T_{\sigma} v = r_{\sigma} + \beta Q_{\sigma} v`.

The main result of the theory of dynamic programming states that the
optimal value function :math:`v^*` is the unique solution to the Bellman
equation, or the unique fixed point of the Bellman operator, and that
:math:`\sigma^*` is an optimal policy function if and only if it is
:math:`v^*`-greedy, i.e., it satisfies :math:`T v^* = T_{\sigma^*} v^*`.

Solution Algorithms
-------------------

The `DiscreteDP` class currently implements the following solution
algorithms:

* value iteration;
* policy iteration;
* modified policy iteration;
* linear programming.

Policy iteration computes an exact optimal policy in finitely many
iterations, while value iteration and modified policy iteration return
an :math:`\varepsilon`-optimal policy and an
:math:`\varepsilon/2`-approximation of the optimal value function for a
prespecified value of :math:`\varepsilon`.

Our implementations of value iteration and modified policy iteration
employ the norm-based and span-based termination rules, respectively.

* Value iteration is terminated when the condition :math:`\lVert T v - v
  \rVert < [(1 - \beta) / (2\beta)] \varepsilon` is satisfied.

* Modified policy iteration is terminated when the condition
  :math:`\mathrm{span}(T v - v) < [(1 - \beta) / \beta] \varepsilon` is
  satisfied, where :math:`\mathrm{span}(z) = \max(z) - \min(z)`.

The linear programming method solves the problem as a linear program by
the simplex method with `optimize.linprog_simplex` routines (implemented
only for dense matrix formulation).

References
----------

M. L. Puterman, Markov Decision Processes: Discrete Stochastic Dynamic
Programming, Wiley-Interscience, 2005.

"""
import warnings
import numpy as np
import scipy.sparse as sp

from .core import MarkovChain
from ._ddp_linprog_simplex import ddp_linprog_simplex
from .utilities import (
    _fill_dense_Q, _s_wise_max_argmax, _s_wise_max, _find_indices,
    _has_sorted_sa_indices, _generate_a_indptr
)


class DiscreteDP:
    r"""
    Class for dealing with a discrete dynamic program.

    There are two ways to represent the data for instantiating a
    `DiscreteDP` object. Let n, m, and L denote the numbers of states,
    actions, and feasbile state-action pairs, respectively.

    1. `DiscreteDP(R, Q, beta)`

       with parameters:

       * n x m reward array `R`,
       * n x m x n transition probability array `Q`, and
       * discount factor beta,

       where `R[s, a]` is the reward for action `a` when the state is
       `s` and `Q[s, a, s_next]` is the probability that the state in the
       next period is `s_next` when the current state is `s` and the action
       chosen is `a`.

    2. `DiscreteDP(R, Q, beta, s_indices, a_indices)`

       with parameters:

       * length L reward vector `R`,
       * L x n transition probability array `Q`,
       * discount factor `beta`,
       * length L array `s_indices`, and
       * length L array `a_indices`,

       where the pairs (`s_indices[0]`, `a_indices[0]`), ...,
       (`s_indices[L-1]`, `a_indices[L-1]`) enumerate feasible
       state-action pairs, and `R[i]` is the reward for action
       `a_indices[i]` when the state is `s_indices[i]` and `Q[i, s_next]` is
       the probability that the state in the next period is `s_next` when
       the current state is `s_indices[i]` and the action chosen is
       `a_indices[i]`. With this formulation, `Q` may be represented by
       a scipy.sparse matrix.

    Parameters
    ----------
    R : array_like(float, ndim=2 or 1)
        Reward array.

    Q : array_like(float, ndim=3 or 2) or scipy.sparse matrix
        Transition probability array.

    beta : scalar(float)
        Discount factor. Must be in [0, 1].

    s_indices : array_like(int, ndim=1), optional(default=None)
        Array containing the indices of the states.

    a_indices : array_like(int, ndim=1), optional(default=None)
        Array containing the indices of the actions.

    Attributes
    ----------
    R, Q, beta : see Parameters.

    num_states : scalar(int)
        Number of states.

    num_sa_pairs : scalar(int)
        Number of feasible state-action pairs (or those that yield
        finite rewards).

    epsilon : scalar(float), default=1e-3
        Default value for epsilon-optimality.

    max_iter : scalar(int), default=250
        Default value for the maximum number of iterations.

    Notes
    -----
    DiscreteDP accepts beta=1 for convenience. In this case, infinite
    horizon solution methods are disabled, and the instance is then seen
    as merely an object carrying the Bellman operator, which may be used
    for backward induction for finite horizon problems.

    Examples
    --------
    Consider the following example, taken from Puterman (2005), Section
    3.1, pp.33-35.

    * Set of states S = {0, 1}

    * Set of actions A = {0, 1}

    * Set of feasible state-action pairs SA = {(0, 0), (0, 1), (1, 0)}

    * Rewards r(s, a):

          r(0, 0) = 5, r(0, 1) =10, r(1, 0) = -1

    * Transition probabilities q(s_next|s, a):

          q(0|0, 0) = 0.5, q(1|0, 0) = 0.5,
          q(0|0, 1) = 0,   q(1|0, 1) = 1,
          q(0|1, 0) = 0,   q(1|1, 0) = 1

    * Discount factor 0.95

    **Creating a `DiscreteDP` instance**

    *Product formulation*

    This approach uses the product set S x A as the domain by treating
    action 1 as yielding a reward negative infinity at state 1.

    >>> R = [[5, 10], [-1, -float('inf')]]
    >>> Q = [[(0.5, 0.5), (0, 1)], [(0, 1), (0.5, 0.5)]]
    >>> beta = 0.95
    >>> ddp = DiscreteDP(R, Q, beta)

    (`Q[1, 1]` is an arbitrary probability vector.)

    *State-action pairs formulation*

    This approach takes the set of feasible state-action pairs SA as
    given.

    >>> s_indices = [0, 0, 1]  # State indices
    >>> a_indices = [0, 1, 0]  # Action indices
    >>> R = [5, 10, -1]
    >>> Q = [(0.5, 0.5), (0, 1), (0, 1)]
    >>> beta = 0.95
    >>> ddp = DiscreteDP(R, Q, beta, s_indices, a_indices)

    **Solving the model**

    *Policy iteration*

    >>> res = ddp.solve(method='policy_iteration', v_init=[0, 0])
    >>> res.sigma  # Optimal policy function
    array([0, 0])
    >>> res.v  # Optimal value function
    array([ -8.57142857, -20.        ])
    >>> res.num_iter  # Number of iterations
    2

    *Value iteration*

    >>> res = ddp.solve(method='value_iteration', v_init=[0, 0],
    ...                 epsilon=0.01)
    >>> res.sigma  # (Approximate) optimal policy function
    array([0, 0])
    >>> res.v  # (Approximate) optimal value function
    array([ -8.5665053 , -19.99507673])
    >>> res.num_iter  # Number of iterations
    162

    *Modified policy iteration*

    >>> res = ddp.solve(method='modified_policy_iteration',
    ...                 v_init=[0, 0], epsilon=0.01)
    >>> res.sigma  # (Approximate) optimal policy function
    array([0, 0])
    >>> res.v  # (Approximate) optimal value function
    array([ -8.57142826, -19.99999965])
    >>> res.num_iter  # Number of iterations
    3

    *Linear programming*

    >>> res = ddp.solve(method='linear_programming', v_init=[0, 0])
    >>> res.sigma  # Optimal policy function
    array([0, 0])
    >>> res.v  # Optimal value function
    array([ -8.57142857, -20.        ])
    >>> res.num_iter  # Number of iterations (within the LP solver)
    4

    """
    def __init__(self, R, Q, beta, s_indices=None, a_indices=None):
        self._sa_pair = False
        self._sparse = False

        if sp.issparse(Q):
            self.Q = Q.tocsr()
            self._sa_pair = True
            self._sparse = True
        else:
            self.Q = np.asarray(Q)
            if self.Q.ndim == 2:
                self._sa_pair = True
            elif self.Q.ndim != 3:
                raise ValueError('Q must be 2- or 3-dimensional')

        self.R = np.asarray(R)
        if not (self.R.ndim in [1, 2]):
            raise ValueError('R must be 1- or 2-dimensional')

        msg_dimension = 'dimensions of R and Q must be either 1 and 2, ' \
                        'or 2 and 3'
        msg_shape = 'shapes of R and Q must be either (n, m) and (n, m, n), ' \
                    'or (L,) and (L, n)'

        if self._sa_pair:
            self.num_sa_pairs, self.num_states = self.Q.shape

            if self.R.ndim != 1:
                raise ValueError(msg_dimension)
            if self.R.shape != (self.num_sa_pairs,):
                raise ValueError(msg_shape)

            if s_indices is None:
                raise ValueError('s_indices must be supplied')
            if a_indices is None:
                raise ValueError('a_indices must be supplied')
            if not (len(s_indices) == self.num_sa_pairs and
                    len(a_indices) == self.num_sa_pairs):
                raise ValueError(
                    'length of s_indices and a_indices must be equal to '
                    'the number of state-action pairs'
                )

            self.s_indices = np.asarray(s_indices)
            self.a_indices = np.asarray(a_indices)

            if _has_sorted_sa_indices(self.s_indices, self.a_indices):
                a_indptr = np.empty(self.num_states+1, dtype=int)
                _generate_a_indptr(self.num_states, self.s_indices,
                                   out=a_indptr)
                self.a_indptr = a_indptr
            else:
                # Sort indices and elements of R and Q
                sa_ptrs = sp.coo_matrix(
                    (np.arange(self.num_sa_pairs), (s_indices, a_indices))
                ).tocsr()
                sa_ptrs.sort_indices()
                self.a_indices = sa_ptrs.indices
                self.a_indptr = sa_ptrs.indptr

                self.R = self.R[sa_ptrs.data]
                self.Q = self.Q[sa_ptrs.data]

                _s_indices = np.empty(self.num_sa_pairs,
                                      dtype=self.a_indices.dtype)
                for i in range(self.num_states):
                    for j in range(self.a_indptr[i], self.a_indptr[i+1]):
                        _s_indices[j] = i
                self.s_indices = _s_indices

            # Define state-wise maximization
            def s_wise_max(vals, out=None, out_argmax=None):
                """
                Return the vector max_a vals(s, a), where vals is represented
                by a 1-dimensional ndarray of shape (self.num_sa_pairs,).
                out and out_argmax must be of length self.num_states; dtype of
                out_argmax must be int.

                """
                if out is None:
                    out = np.empty(self.num_states)
                if out_argmax is None:
                    _s_wise_max(self.a_indices, self.a_indptr, vals,
                                out_max=out)
                else:
                    _s_wise_max_argmax(self.a_indices, self.a_indptr, vals,
                                       out_max=out, out_argmax=out_argmax)
                return out

            self.s_wise_max = s_wise_max

        else:  # Not self._sa_pair
            if self.R.ndim != 2:
                raise ValueError(msg_dimension)
            n, m = self.R.shape

            if self.Q.shape != (n, m, n):
                raise ValueError(msg_shape)

            self.num_states = n
            self.s_indices, self.a_indices = None, None
            self.num_sa_pairs = (self.R > -np.inf).sum()

            # Define state-wise maximization
            def s_wise_max(vals, out=None, out_argmax=None):
                """
                Return the vector max_a vals(s, a), where vals is represented
                by a 2-dimensional ndarray of shape (n, m). Stored in out,
                which must be of length self.num_states.
                out and out_argmax must be of length self.num_states; dtype of
                out_argmax must be int.

                """
                if out is None:
                    out = np.empty(self.num_states)
                if out_argmax is None:
                    vals.max(axis=1, out=out)
                else:
                    vals.argmax(axis=1, out=out_argmax)
                    out[:] = vals[np.arange(self.num_states), out_argmax]
                return out

            self.s_wise_max = s_wise_max

        # Check that for every state, at least one action is feasible
        self._check_action_feasibility()

        if not (0 <= beta <= 1):
            raise ValueError('beta must be in [0, 1]')
        if beta == 1:
            msg = 'infinite horizon solution methods are disabled with beta=1'
            warnings.warn(msg)
            self._error_msg_no_discounting = 'method invalid for beta=1'
        self.beta = beta

        self.epsilon = 1e-3
        self.max_iter = 250

        # Linear equation solver to be used in evaluate_policy
        if self._sparse:
            import scipy.sparse.linalg
            self._lineq_solve = scipy.sparse.linalg.spsolve
            self._I = sp.identity(self.num_states, format='csr')
        else:
            self._lineq_solve = np.linalg.solve
            self._I = np.identity(self.num_states)

    def _check_action_feasibility(self):
        """
        Check that for every state, reward is finite for some action,
        and for the case sa_pair is True, that for every state, there is
        some action available.

        """
        # Check that for every state, reward is finite for some action
        R_max = self.s_wise_max(self.R)
        if (R_max == -np.inf).any():
            # First state index such that all actions yield -inf
            s = np.where(R_max == -np.inf)[0][0]
            raise ValueError(
                'for every state the reward must be finite for some action: '
                'violated for state {s}'.format(s=s)
            )

        if self._sa_pair:
            # Check that for every state there is at least one action available
            diff = np.diff(self.a_indptr)
            if (diff == 0).any():
                # First state index such that no action is available
                s = np.where(diff == 0)[0][0]
                raise ValueError(
                    'for every state at least one action must be available: '
                    'violated for state {s}'.format(s=s)
                )

    def to_sa_pair_form(self, sparse=True):
        """
        Convert this instance of `DiscreteDP` to SA-pair form

        Parameters
        ----------
        sparse : bool, optional(default=True)
            Should the `Q` matrix be stored as a sparse matrix?
            If true the CSR format is used

        Returns
        -------
        ddp_sa : DiscreteDP
            The correspnoding DiscreteDP instance in SA-pair form

        Notes
        -----
        If this instance is already in SA-pair form then it is returned
        un-modified
        """

        if self._sa_pair:
            return self
        else:
            s_ind, a_ind = np.where(self.R > - np.inf)
            RL = self.R[s_ind, a_ind]
            if sparse:
                QL = sp.csr_matrix(self.Q[s_ind, a_ind])
            else:
                QL = self.Q[s_ind, a_ind]
            return DiscreteDP(RL, QL, self.beta, s_ind, a_ind)

    def to_product_form(self):
        """
        Convert this instance of `DiscreteDP` to the "product" form.

        The product form uses the version of the init method taking
        `R`, `Q` and `beta`.

        Returns
        -------
        ddp_sa : DiscreteDP
            The correspnoding DiscreteDP instance in product form

        Notes
        -----
        If this instance is already in product form then it is returned
        un-modified

        """
        if self._sa_pair:
            ns = self.num_states
            na = self.a_indices.max() + 1
            R = np.full((ns, na), -np.inf)
            R[self.s_indices, self.a_indices] = self.R
            Q = np.zeros((ns, na, ns))
            if self._sparse:
                _fill_dense_Q(self.s_indices, self.a_indices,
                              self.Q.toarray(), Q)
            else:
                _fill_dense_Q(self.s_indices, self.a_indices, self.Q, Q)
            return DiscreteDP(R, Q, self.beta)
        else:
            return self

    def RQ_sigma(self, sigma):
        """
        Given a policy `sigma`, return the reward vector `R_sigma` and
        the transition probability matrix `Q_sigma`.

        Parameters
        ----------
        sigma : array_like(int, ndim=1)
            Policy vector, of length n.

        Returns
        -------
        R_sigma : ndarray(float, ndim=1)
            Reward vector for `sigma`, of length n.

        Q_sigma : ndarray(float, ndim=2)
            Transition probability matrix for `sigma`, of shape (n, n).

        """
        if self._sa_pair:
            sigma = np.asarray(sigma)
            sigma_indices = np.empty(self.num_states, dtype=int)
            _find_indices(self.a_indices, self.a_indptr, sigma,
                          out=sigma_indices)
            R_sigma, Q_sigma = self.R[sigma_indices], self.Q[sigma_indices]
        else:
            R_sigma = self.R[np.arange(self.num_states), sigma]
            Q_sigma = self.Q[np.arange(self.num_states), sigma]

        return R_sigma, Q_sigma

    def bellman_operator(self, v, Tv=None, sigma=None):
        """
        The Bellman operator, which computes and returns the updated
        value function `Tv` for a value function `v`.

        Parameters
        ----------
        v : array_like(float, ndim=1)
            Value function vector, of length n.

        Tv : ndarray(float, ndim=1), optional(default=None)
            Optional output array for Tv.

        sigma : ndarray(int, ndim=1), optional(default=None)
            If not None, the v-greedy policy vector is stored in this
            array. Must be of length n.

        Returns
        -------
        Tv : ndarray(float, ndim=1)
            Updated value function vector, of length n.

        """
        vals = self.R + self.beta * self.Q.dot(v)  # Shape: (L,) or (n, m)

        if Tv is None:
            Tv = np.empty(self.num_states)
        self.s_wise_max(vals, out=Tv, out_argmax=sigma)
        return Tv

    def T_sigma(self, sigma):
        """
        Given a policy `sigma`, return the T_sigma operator.

        Parameters
        ----------
        sigma : array_like(int, ndim=1)
            Policy vector, of length n.

        Returns
        -------
        callable
            The T_sigma operator.

        """
        R_sigma, Q_sigma = self.RQ_sigma(sigma)
        return lambda v: R_sigma + self.beta * Q_sigma.dot(v)

    def compute_greedy(self, v, sigma=None):
        """
        Compute the v-greedy policy.

        Parameters
        ----------
        v : array_like(float, ndim=1)
            Value function vector, of length n.

        sigma : ndarray(int, ndim=1), optional(default=None)
            Optional output array for `sigma`.

        Returns
        -------
        sigma : ndarray(int, ndim=1)
            v-greedy policy vector, of length n.

        """
        if sigma is None:
            sigma = np.empty(self.num_states, dtype=int)
        self.bellman_operator(v, sigma=sigma)
        return sigma

    def evaluate_policy(self, sigma):
        """
        Compute the value of a policy.

        Parameters
        ----------
        sigma : array_like(int, ndim=1)
            Policy vector, of length n.

        Returns
        -------
        v_sigma : ndarray(float, ndim=1)
            Value vector of `sigma`, of length n.

        """
        if self.beta == 1:
            raise NotImplementedError(self._error_msg_no_discounting)

        # Solve (I - beta * Q_sigma) v = R_sigma for v
        R_sigma, Q_sigma = self.RQ_sigma(sigma)
        b = R_sigma

        A = self._I - self.beta * Q_sigma

        v_sigma = self._lineq_solve(A, b)

        return v_sigma

    def operator_iteration(self, T, v, max_iter, tol=None, *args, **kwargs):
        """
        Iteratively apply the operator `T` to `v`. Modify `v` in-place.
        Iteration is performed for at most a number `max_iter` of times.
        If `tol` is specified, it is terminated once the distance of
        `T(v)` from `v` (in the max norm) is less than `tol`.

        Parameters
        ----------
        T : callable
            Operator that acts on `v`.

        v : ndarray
            Object on which `T` acts. Modified in-place.

        max_iter : scalar(int)
            Maximum number of iterations.

        tol : scalar(float), optional(default=None)
            Error tolerance.

        args, kwargs :
            Other arguments and keyword arguments that are passed
            directly to the function T each time it is called.

        Returns
        -------
        num_iter : scalar(int)
            Number of iterations performed.

        """
        # May be replaced with quantecon.compute_fixed_point
        if max_iter <= 0:
            return v, 0

        for i in range(max_iter):
            new_v = T(v, *args, **kwargs)
            if tol is not None and np.abs(new_v - v).max() < tol:
                v[:] = new_v
                break
            v[:] = new_v

        num_iter = i + 1

        return num_iter

    def solve(self, method='policy_iteration',
              v_init=None, epsilon=None, max_iter=None, k=20):
        """
        Solve the dynamic programming problem.

        Parameters
        ----------
        method : str, optinal(default='policy_iteration')
            Solution method, str in {'value_iteration', 'vi',
            'policy_iteration', 'pi', 'modified_policy_iteration',
            'mpi', 'linear_programming', 'lp'}.

        v_init : array_like(float, ndim=1), optional(default=None)
            Initial value function, of length n. If None, `v_init` is
            set such that v_init(s) = max_a r(s, a) for value iteration,
            policy iteration, and linear programming; for modified
            policy iteration, v_init(s) = min_(s_next, a)
            r(s_next, a)/(1 - beta) to guarantee convergence.

        epsilon : scalar(float), optional(default=None)
            Value for epsilon-optimality. If None, the value stored in
            the attribute `epsilon` is used.

        max_iter : scalar(int), optional(default=None)
            Maximum number of iterations. If None, the value stored in
            the attribute `max_iter` is used.

        k : scalar(int), optional(default=20)
            Number of iterations for partial policy evaluation in
            modified policy iteration (irrelevant for other methods).

        Returns
        -------
        res : DPSolveResult
            Optimization result represetned as a DPSolveResult. See
            `DPSolveResult` for details.

        """
        if method in ['value_iteration', 'vi']:
            res = self.value_iteration(v_init=v_init,
                                       epsilon=epsilon,
                                       max_iter=max_iter)
        elif method in ['policy_iteration', 'pi']:
            res = self.policy_iteration(v_init=v_init,
                                        max_iter=max_iter)
        elif method in ['modified_policy_iteration', 'mpi']:
            res = self.modified_policy_iteration(v_init=v_init,
                                                 epsilon=epsilon,
                                                 max_iter=max_iter,
                                                 k=k)
        elif method in ['linear_programming', 'lp']:
            res = self.linprog_simplex(v_init=v_init,
                                       max_iter=max_iter)
        else:
            raise ValueError('invalid method')

        return res

    def value_iteration(self, v_init=None, epsilon=None, max_iter=None):
        """
        Solve the optimization problem by value iteration. See the
        `solve` method.

        """
        if self.beta == 1:
            raise NotImplementedError(self._error_msg_no_discounting)

        if max_iter is None:
            max_iter = self.max_iter
        if epsilon is None:
            epsilon = self.epsilon

        try:
            tol = epsilon * (1-self.beta) / (2*self.beta)
        except ZeroDivisionError:  # Raised if beta = 0
            tol = np.inf

        v = np.empty(self.num_states)
        if v_init is None:
            self.s_wise_max(self.R, out=v)
        else:
            v[:] = v_init

        # Storage array for self.bellman_operator
        Tv = np.empty(self.num_states)

        num_iter = self.operator_iteration(T=self.bellman_operator,
                                           v=v, max_iter=max_iter, tol=tol,
                                           Tv=Tv)
        sigma = self.compute_greedy(v)

        res = DPSolveResult(v=v,
                            sigma=sigma,
                            num_iter=num_iter,
                            mc=self.controlled_mc(sigma),
                            method='value iteration',
                            epsilon=epsilon,
                            max_iter=max_iter)

        return res

    def policy_iteration(self, v_init=None, max_iter=None):
        """
        Solve the optimization problem by policy iteration. See the
        `solve` method.

        """
        if self.beta == 1:
            raise NotImplementedError(self._error_msg_no_discounting)

        if max_iter is None:
            max_iter = self.max_iter

        # What for initial condition?
        if v_init is None:
            v_init = self.s_wise_max(self.R)

        sigma = self.compute_greedy(v_init)
        new_sigma = np.empty(self.num_states, dtype=int)

        for i in range(max_iter):
            # Policy evaluation
            v_sigma = self.evaluate_policy(sigma)
            # Policy improvement
            self.compute_greedy(v_sigma, sigma=new_sigma)
            if np.array_equal(new_sigma, sigma):
                break
            sigma[:] = new_sigma

        num_iter = i + 1

        res = DPSolveResult(v=v_sigma,
                            sigma=sigma,
                            num_iter=num_iter,
                            mc=self.controlled_mc(sigma),
                            method='policy iteration',
                            max_iter=max_iter)

        return res

    def modified_policy_iteration(self, v_init=None, epsilon=None,
                                  max_iter=None, k=20):
        """
        Solve the optimization problem by modified policy iteration. See
        the `solve` method.

        """
        if self.beta == 1:
            raise NotImplementedError(self._error_msg_no_discounting)

        if max_iter is None:
            max_iter = self.max_iter
        if epsilon is None:
            epsilon = self.epsilon

        def span(z):
            return z.max() - z.min()

        def midrange(z):
            return (z.min() + z.max()) / 2

        v = np.empty(self.num_states)
        if v_init is None:
            v[:] = self.R[self.R > -np.inf].min() / (1 - self.beta)
        else:
            v[:] = v_init

        u = np.empty(self.num_states)
        sigma = np.empty(self.num_states, dtype=int)

        try:
            tol = epsilon * (1-self.beta) / self.beta
        except ZeroDivisionError:  # Raised if beta = 0
            tol = np.inf

        for i in range(max_iter):
            # Policy improvement
            self.bellman_operator(v, Tv=u, sigma=sigma)
            diff = u - v
            if span(diff) < tol:
                v[:] = u + midrange(diff) * self.beta / (1 - self.beta)
                break
            # Partial policy evaluation with k iterations
            self.operator_iteration(T=self.T_sigma(sigma), v=u, max_iter=k)
            v[:] = u

        num_iter = i + 1

        res = DPSolveResult(v=v,
                            sigma=sigma,
                            num_iter=num_iter,
                            mc=self.controlled_mc(sigma),
                            method='modified policy iteration',
                            epsilon=epsilon,
                            max_iter=max_iter,
                            k=k)

        return res

    def linprog_simplex(self, v_init=None, max_iter=None):
        if self.beta == 1:
            raise NotImplementedError(self._error_msg_no_discounting)

        if self._sparse:
            raise NotImplementedError('method invalid for sparse formulation')

        if max_iter is None:
            max_iter = self.max_iter * self.num_states

        # What for initial condition?
        if v_init is None:
            v_init = self.s_wise_max(self.R)
        v_init = np.asarray(v_init)

        sigma = self.compute_greedy(v_init)

        ddp_sa = self.to_sa_pair_form(sparse=False)
        R, Q = ddp_sa.R, ddp_sa.Q
        a_indices, a_indptr = ddp_sa.a_indices, ddp_sa.a_indptr

        _, num_iter, v, sigma = ddp_linprog_simplex(
            R, Q, self.beta, a_indices, a_indptr, sigma, max_iter=max_iter
        )

        res = DPSolveResult(v=v,
                            sigma=sigma,
                            num_iter=num_iter,
                            mc=self.controlled_mc(sigma),
                            method='linear programming',
                            max_iter=max_iter)

        return res

    def controlled_mc(self, sigma):
        """
        Returns the controlled Markov chain for a given policy `sigma`.

        Parameters
        ----------
        sigma : array_like(int, ndim=1)
            Policy vector, of length n.

        Returns
        -------
        mc : MarkovChain
            Controlled Markov chain.

        """
        _, Q_sigma = self.RQ_sigma(sigma)
        return MarkovChain(Q_sigma)


class DPSolveResult(dict):
    """
    Contain the information about the dynamic programming result.

    Attributes
    ----------
    v : ndarray(float, ndim=1)
        Computed optimal value function

    sigma : ndarray(int, ndim=1)
        Computed optimal policy function

    num_iter : int
        Number of iterations

    mc : MarkovChain
        Controlled Markov chain

    method : str
        Method employed

    epsilon : float
        Value of epsilon

    max_iter : int
        Maximum number of iterations

    """
    # This is sourced from sicpy.optimize.OptimizeResult.
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in self.items()])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return self.keys()


def backward_induction(ddp, T, v_term=None):
    r"""
    Solve by backward induction a :math:`T`-period finite horizon
    discrete dynamic program with stationary reward and transition
    probability functions :math:`r` and :math:`q` and discount factor
    :math:`\beta \in [0, 1]`.

    The optimal value functions :math:`v^*_0, \ldots, v^*_T` and policy
    functions :math:`\sigma^*_0, \ldots, \sigma^*_{T-1}` are obtained by
    :math:`v^*_T = v_T`, and

    .. math::

        v^*_{t-1}(s) = \max_{a \in A(s)} r(s, a) +
            \beta \sum_{s' \in S} q(s'|s, a) v^*_t(s')
            \quad (s \in S)

    and

    .. math::

        \sigma^*_{t-1}(s) \in \operatorname*{arg\,max}_{a \in A(s)}
            r(s, a) + \beta \sum_{s' \in S} q(s'|s, a) v^*_t(s')
            \quad (s \in S)

    for :math:`t = T, \ldots, 1`, where the terminal value function
    :math:`v_T` is exogenously given.

    Parameters
    ----------
    ddp : DiscreteDP
        DiscreteDP instance storing reward array `R`, transition
        probability array `Q`, and discount factor `beta`.

    T : scalar(int)
        Number of decision periods.

    v_term : array_like(float, ndim=1), optional(default=None)
        Terminal value function, of length equal to n (the number of
        states). If None, it defaults to the vector of zeros.

    Returns
    -------
    vs : ndarray(float, ndim=2)
        Array of shape (T+1, n) where `vs[t]` contains the optimal
        value function at period `t = 0, ..., T`.

    sigmas : ndarray(int, ndim=2)
        Array of shape (T, n) where `sigmas[t]` contains the optimal
        policy function at period `t = 0, ..., T-1`.

    """
    n = ddp.num_states
    vs = np.empty((T+1, n))
    sigmas = np.empty((T, n), dtype=int)

    if v_term is None:
        v_term = np.zeros(n)
    vs[T, :] = v_term

    for t in range(T, 0, -1):
        ddp.bellman_operator(vs[t, :], Tv=vs[t-1, :], sigma=sigmas[t-1, :])

    return vs, sigmas
