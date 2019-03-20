import numpy as np
import numbers
from scipy import sparse

class LocalInteraction(object):
	def __init__(self, payoff_matrix, adj_matrix):
		self.adj_matrix = sparse.csr_matrix(adj_matrix)
		M, N = self.adj_matrix.shape
		if N != M:
			raise ValueError('adjacency matrix must be square')
		self.N = N

		A = np.asarray(payoff_matrix)
		if A.ndim != 2 or A.shape[0] != A.shape[1]:
			raise ValueError('payoff matrix must be square')
		self.num_actions = A.shape[0]

		self.players = [Player(A) for i in range(self.N)]
		self.tie_breaking = 'smallest'

	def _play(self, actions, player_ind):
		actions_matrix = sparse.csr_matrix(
			(np.ones(self.N, dtype=int), actions, np.arange(self.N+1)),
			shape=(self.N, self.num_actions))

		opponent_act_dict = self.adj_matrix[player_ind].dot(
			actions_matrix).toarray()

		for k, i in enumerate(player_ind):
			actions[k] = self.players[i].best_response(opponent_act_dict[k, :],
				tie_breaking=self.tie_breaking)

		return actions

	def play(self, init_actions=None, player_ind=None, num_reps=1):
		if init_actions is None:
			nums_actions = tuple([self.num_actions] * self.N)
			init_actions = random_pure_actions(nums_actions)

		if player_ind is None:
			player_ind = list(range(self.N))
		elif isinstance(player_ind, numbers.Integral):
			player_ind = [player_ind]

		actions = [action for action in init_actions]
		for t in range(num_reps):
			actions = self._play(actions, player_ind)

		return actions

	def time_series(self, init_actions=None, player_ind_seq=None, ts_length):
		if init_actions is None:
			nums_actions = tuple([self.num_actions] * self.N)
			init_actions = random_pure_actions(nums_actions)

		if player_ind_seq is None:
			player_ind_seq = [list(range(self.N))] * (ts_length - 1)

		actions = [action for action in init_actions]
		out = np.empty((ts_length, self.N), dtype=int)
		for t in range(ts_length):
			for i in range(self.N):
				out[t,i] = actions[i]
			actions = self._play(actions, player_ind_seq[t])

		return out