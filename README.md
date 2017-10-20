# Bandit Environments

Series of n-armed bandit environments for the OpenAI Gym

This code is inspired by Jesse Cooper's work:
https://github.com/JKCooper2/gym-bandits

The environments added in this repository are based on Wang et. al experiments described in the paper Learning to Reinforcement Learn.
https://arxiv.org/abs/1611.05763#

### Notes

Each environment uses a different set of:
* Probability Distributions - A list of probabilities of the likelihood that a particular bandit will pay out
* Reward Distributions - A list of either rewards (if number) or means and standard deviations (if list) of the payout that bandit has

E.g. BanditTwoArmedHighLowFixed-v0 has `p_dist=[0.8, 0.2]`, `r_dist=[1, 1]`, meaning 80% of the time that action 0 is
selected it will payout 1, and 20% of the time action 2 is selected it will payout 1

You can access the distributions through the p_dist and r_dist variables using `env.p_dist` or `env.r_dist` if you want to match
your weights against the true values for plotting results of various algorithms

To fit the universe-starter-agent, the observation of the bandits has been modified from 0 (type: gym.spaces.Discrete) to [0] (type: gym.spaces.box.Box).

Some of the environments return pieces of information regarding the arms. For example: the index of the optimal arm or the value of a parameter.

### List of Environments

New in this repository:

* BanditTwoArmedIndependentUniform-v0: The two arms return a reward of 1 with probabilities p1 and p2 ~ U[0,1]
* BanditTwoArmedDependentUniform-v0: The first arm returns a reward of 1 with probability p ~ U[0,1], the second arm with probability 1-p
* BanditTwoArmedDependentEasy-v0: The first arm returns a reward of 1 with probability p ~ U{0.1,0.9}, the second arm with probability 1-p
* BanditTwoArmedDependentMedium-v0: The first arm returns a reward of 1 with probability p ~ U{0.25,0.75}, the second arm with probability 1-p
* BanditTwoArmedDependentHard-v0: The first arm returns a reward of 1 with probability p ~ U{0.4,0.6}, the second arm with probability 1-p
* BanditElevenArmedWithIndex: One optimal arm always returns a reward of 5, the other arms a reward of 1.1 ; The 11th arm return a reward of 0.1*<Index of the optimal arm>


Other environments:

* BanditTwoArmedDeterministicFixed-v0: Simplest case where one bandit always pays, and the other always doesn't
* BanditTwoArmedHighLowFixed-v0: Stochastic version with a large difference between which bandit pays out of two choices
* BanditTwoArmedHighHighFixed-v0: Stochastic version with a small difference between which bandit pays where both are good
* BanditTwoArmedLowLowFixed-v0: Stochastic version with a small difference between which bandit pays where both are bad
* BanditTenArmedRandomFixed-v0: 10 armed bandit with random probabilities assigned to payouts
* BanditTenArmedRandomRandom-v0: 10 armed bandit with random probabilities assigned to both payouts and rewards
* BanditTenArmedUniformDistributedReward-v0: 10 armed bandit with that always pays out with a reward selected from a uniform distribution
* BanditTenArmedGaussian-v0: 10 armed bandit mentioned on page 30 of [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf?dl=0) (Sutton and Barto)

### Installation
```
git clone git@github.com:JKCooper2/gym-bandits.git
cd gym-bandits
pip install -e .
```

In your gym environment
```
import gym_bandits
env = gym.make("BanditTenArmedGaussian-v0") # Replace with relevant env
```
