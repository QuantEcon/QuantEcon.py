import numpy as np

class FictitiousPlay(object):

	def __init__(self, data, gain=None):
		if isinstance(data, NormalFormGame):
			self.g = data
		else:  # data must be array_like
			payoffs = np.asarray(data)
			self.g = NormalFormGame(payoffs)

		self.N = self.g.N 
		self.players = self.g.players
		self.nums_actions = self.g.nums_actions
		self.tie_breaking = 'smallest'

		if gain is None:
			self.step_size = lambda t: 1 / (t+1) # decreasing gain
		else:
			self.step_size = lambda t: gain # constant gain

	def _play(self, actions, t):
		brs = np.zeros(self.N, dtype=int)
		for i, player in enumerate(self.players):
			index = [j for j in range(i+1, self.N)]
			index.extend([j for j in range(i)])
			opponent_actions = np.asarray([actions[i] for i in index])
			brs[i] = player.best_response(opponent_actions
					if self.N > 2 else opponent_actions[0],
					tie_breaking=self.tie_breaking)

		for i in range(self.N):
			actions[i][:] *= 1 - self.step_size(t+1)
			actions[i][brs[i]] += self.step_size(t+1)

		return actions

	def play(self, init_actions=None, num_reps=1, t_init=0):
		if init_actions is None:
			init_actions = random_pure_actions(self.nums_actions)
		actions = [i for i in init_actions]
		for i in range(self.N):
			actions[i] = pure2mixed(self.nums_actions[i], init_actions[i])
		for t in range(num_reps):
			actions = self._play(actions, t+t_init)
		return actions

	def time_series(self, ts_length, init_actions=None, t_init=0):
		if init_actions is None:
			init_actions = random_pure_actions(self.nums_actions)
		out = [np.empty((ts_length, self.nums_actions[i]))
			for i in range(self.N)]
		actions = [np.empty(self.nums_actions[i]) for i in range(self.N)]
		for i in range(self.N):
			actions[i] = pure2mixed(self.nums_actions[i], init_actions[i])[:]
		for t in range(ts_length):
			for i in range(self.N):
				out[i][t,:] = actions[i][:] 
			actions = self._play(actions, t+t_init)
		return out

class StochasticFictitiousPlay(FictitiousPlay):

	def __init__(self, data, gain=None, distribution='extreme'):
		FictitiousPlay.__init__(self, data, gain)

		if distribution == 'extreme':
			loc = -np.euler_gamma * np.sqrt(6) / np.pi
			scale = np.sqrt(6) / np.pi
			self.payoff_perturbation_dist = \
				lambda size: np.random.gumbel(loc=loc, scale=scale, size=size)
		elif distribution == 'normal':  # standard normal distribution
			self.payoff_perturbation_dist = \
				lambda size: np.random.standard_normal(size=size)
		else:
			raise ValueError("`distribution` must be 'extreme' or 'normal'")

		self.tie_breaking = 'smallest'

	def _play(self, actions, t):
		brs = np.zeros(self.N, dtype=int)
		for i, player in enumerate(self.players):
			index = [j for j in range(i+1, self.N)]
			index.extend([j for j in range(i)])
			opponent_actions = np.asarray([actions[i] for i in index])
			payoff_perturbation = \
				self.payoff_perturbation_dist(size=self.nums_actions[i])
			brs[i] = player.best_response(opponent_actions
				if self.N > 2 else opponent_actions[0],
				tie_breaking=self.tie_breaking,
				payoff_perturbation=payoff_perturbation)

		for i in range(self.N):
			actions[i][:] *= 1 - self.step_size(t+1)
			actions[i][brs[i]] += self.step_size(t+1)

		return actions