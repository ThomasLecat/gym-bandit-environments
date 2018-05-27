import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class BanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit
    in different variations

    p_dist:
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist:
        A list of either rewards (if number) or means and standard deviations (if list)
        of the payout that bandit has
    info:
        Info about the environment that the agents is not supposed to know. For instance,
        info can releal the index of the optimal arm, or the value of prior parameter.
        Can be useful to evaluate the agent's perfomance
    """
    def __init__(self, p_dist, r_dist, info={}):
        if len(p_dist) != len(r_dist):
            raise ValueError("Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.p_dist = p_dist
        self.r_dist = r_dist

        self.info = info

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.observation_space = spaces.box.Box(-1.0, 1.0, (1)) #
        #self.observation_space = spaces.Discrete(1)

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = True

        if np.random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]
            else:
                reward = np.random.normal(self.r_dist[action][0], self.r_dist[action][1])

        return [0], reward, done, self.info #

    def _reset(self):
        return [0] #

    def _render(self, mode='human', close=False):
        pass


############### Bandits written by Thomas Lecat #################

class BanditTwoArmedIndependentUniform(BanditEnv):
	"""
	2 armed bandit giving a reward of 1 with inDependent probabilities p_1 and p_2
	"""
	def __init__(self, bandits=2):
		p_dist = np.random.uniform(size=bandits)
		r_dist = np.full(bandits,1)

		BanditEnv.__init__(self, p_dist = p_dist, r_dist=r_dist)

class BanditTwoArmedDependentUniform(BanditEnv):
    """
    2 armed bandit giving a reward of 1 with probabilities p_1 ~ U[0,1] and p_2 = 1 - p_1
    """
    def __init__(self):
        p = np.random.uniform()
        p_dist = [p, 1-p]
        r_dist = [1, 1]
        info={'parameter':p}
        BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist, info = info)

class BanditTwoArmedDependentEasy(BanditEnv):
    """
    2 armed bandit giving a reward of 1 with probabilities p_1 ~ U[0.1,0.9] and p_2 = 1 - p_1
    """
    def __init__(self):
        p = [0.1,0.9][np.random.randint(0,2)]
        p_dist = [p, 1-p]
        r_dist = [1, 1]
        optimal_arm = np.abs(1-int(round(p)))
        info = {'optimal_arm':optimal_arm}
        BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist, info=info)

class BanditTwoArmedDependentMedium(BanditEnv):
    """
    2 armed bandit giving a reward of 1 with probabilities p_1 ~ U[0.25,0.75] and p_2 = 1 - p_1
    """
    def __init__(self):
        p = [0.25,0.75][np.random.randint(0,2)]
        p_dist = [p, 1-p]
        r_dist = [1, 1]
        optimal_arm = np.abs(1-int(round(p)))
        info = {'optimal_arm':optimal_arm}
        BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist, info=info)

class BanditTwoArmedDependentHard(BanditEnv):
    """
    2 armed bandit giving a reward of 1 with probabilities p_1 ~ U[0.4,0.6] and p_2 = 1 - p_1
    """
    def __init__(self):
        p = [0.4,0.6][np.random.randint(0,2)]
        p_dist = [p, 1-p]
        r_dist = [1, 1]
        optimal_arm = np.abs(1-int(round(p)))
        info = {'optimal_arm':optimal_arm}
        BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist, info=info)

class BanditElevenArmedWithIndex(BanditEnv):
    """
    11 armed bandit:
    1 out of the 10 first arms gives a reward of 5 (optimal arm), the 9 other arms give reward of 1.1. The 11th arm gives a reward of 0.1 * index of the optimal arm.
    """
    def __init__(self):
        index = np.random.randint(0,10)
        p_dist = np.full(11,1)
        r_dist = np.full(11,1.1)
        r_dist[index] = 5
        r_dist[-1] = 0.1*(index+1) # Note: we add 1 because the arms are indexed from 1 to 10, not 0 to 9
        info = {'optimal_arm':10*index}
        BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist, info=info)


###################### Bandits written by Jesse Coopper #########################

class BanditTwoArmedDeterministicFixed(BanditEnv):
    """Simplest case where one bandit always pays, and the other always doesn't"""
    def __init__(self):
        BanditEnv.__init__(self, p_dist=[1, 0], r_dist=[1, 1], info={'optimal_arm':1})


class BanditTwoArmedHighLowFixed(BanditEnv):
    """Stochastic version with a large difference between which bandit pays out of two choices"""
    def __init__(self):
        BanditEnv.__init__(self, p_dist=[0.8, 0.2], r_dist=[1, 1], info={'optimal_arm':1})


class BanditTwoArmedHighHighFixed(BanditEnv):
    """Stochastic version with a small difference between which bandit pays where both are good"""
    def __init__(self):
        BanditEnv.__init__(self, p_dist=[0.8, 0.9], r_dist=[1, 1], info={'optimal_arm':2})


class BanditTwoArmedLowLowFixed(BanditEnv):
    """Stochastic version with a small difference between which bandit pays where both are bad"""
    def __init__(self):
        BanditEnv.__init__(self, p_dist=[0.1, 0.2], r_dist=[1, 1], info={'optimal_arm':2})


class BanditTenArmedRandomFixed(BanditEnv):
    """10 armed bandit with random probabilities assigned to payouts"""
    def __init__(self, bandits=10):
        p_dist = np.random.uniform(size=bandits)
        r_dist = np.full(bandits, 1)
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditTenArmedUniformDistributedReward(BanditEnv):
    """10 armed bandit with that always pays out with a reward selected from a uniform distribution"""
    def __init__(self, bandits=10):
        p_dist = np.full(bandits, 1)
        r_dist = np.random.uniform(size=bandits)
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditTenArmedRandomRandom(BanditEnv):
    """10 armed bandit with random probabilities assigned to both payouts and rewards"""
    def __init__(self, bandits=10):
        p_dist = np.random.uniform(size=bandits)
        r_dist = np.random.uniform(size=bandits)
        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)


class BanditTenArmedGaussian(BanditEnv):
    """
    10 armed bandit mentioned on page 30 of Sutton and Barto's
    [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf?dl=0)

    Actions always pay out
    Mean of payout is pulled from a normal distribution (0, 1) (called q*(a))
    Actual reward is drawn from a normal distribution (q*(a), 1)
    """
    def __init__(self, bandits=10):
        p_dist = np.full(bandits, 1)
        r_dist = []

        for i in range(bandits):
            r_dist.append([np.random.normal(0, 1), 1])

        BanditEnv.__init__(self, p_dist=p_dist, r_dist=r_dist)
