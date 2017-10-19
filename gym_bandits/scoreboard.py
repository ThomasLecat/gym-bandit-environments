from gym.scoreboard.registration import add_task, add_group


add_group(
    id='bandits',
    name='Bandits',
    description='Various N-Armed Bandit environments'
)

add_task(
    id='BanditTwoArmedDeterministicFixed-v0',
    group='bandits',
    experimental=True,
    contributor='jkcooper2',
    summary="Simplest bandit where one action always pays, and the other never does.",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p_dist = [1, 0]
    r_dist = [1, 1]
    """,
    background=""
)

add_task(
    id='BanditTwoArmedHighHighFixed-v0',
    group='bandits',
    experimental=True,
    contributor='jkcooper2',
    summary="Stochastic version with a small difference between which bandit pays where both are likely",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p_dist = [0.8, 0.9]
    r_dist = [1, 1]
    """,
    background="Bandit B Figure 2.3 from Reinforcement Learning: An Introduction (Sutton & Barto) [link](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node18.html)"
)

add_task(
    id='BanditTwoArmedLowLowFixed-v0',
    group='bandits',
    experimental=True,
    contributor='jkcooper2',
    summary="Stochastic version with a small difference between which bandit pays where both are unlikley",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p_dist = [0.1, 0.2]
    r_dist = [1, 1]
    """,
    background="Bandit A Figure 2.3 from Reinforcement Learning: An Introduction (Sutton & Barto) [link](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node18.html)"
)

add_task(
    id='BanditTwoArmedHighLowFixed-v0',
    group='bandits',
    experimental=True,
    contributor='jkcooper2',
    summary="Stochastic version with a large difference between which bandit pays out of two choices",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p_dist = [0.8, 0.2]
    r_dist = [1, 1]
    """,
    background=""
)

add_task(
    id='BanditTenArmedGaussian-v0',
    group='bandits',
    experimental=True,
    contributor='jkcooper2',
    summary="10 armed bandit mentioned with reward based on a Gaussian distribution",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p_dist = [1] (* 10)
    r_dist = [numpy.random.normal(0, 1), 1] (* 10)

    Every bandit always pays out
    Each action has a reward mean (selected from a normal distribution with mean 0 and std 1), and the actual
    reward returns is selected with a std of 1 around the selected mean
    """,
    background="Described on page 30 of Sutton and Barto's [Reinforcement Learning: An Introduction](https://www.dropbox.com/s/b3psxv2r0ccmf80/book2015oct.pdf?dl=0)"
)

add_task(
    id='BanditTenArmedRandomRandom-v0',
    group='bandits',
    experimental=True,
    contributor='jkcooper2',
    summary="10 armed bandit with random probabilities assigned to both payouts and rewards",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p_dist = numpy.random.uniform(size=10)
    r_dist = numpy.random.uniform(size=10)

    Bandits have uniform probability of paying out and payout a reward of uniform probability
    """,
    background=""
)

add_task(
    id='BanditTenArmedRandomFixed-v0',
    group='bandits',
    experimental=True,
    contributor='jkcooper2',
    summary="10 armed bandit with random probabilities assigned to how often the action will provide a reward",
    description="""
        Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
        and a reward distribution, which is the value or distribution of what the agent will be rewarded
        the bandit does payout.

        p_dist = numpy.random.uniform(size=10)
        r_dist = numpy.full(bandits, 1)

        Bandits have a uniform probability of rewarding and always reward 1
        """,
    background=""
)

add_task(
    id='BanditTenArmedUniformDistributedReward-v0',
    group='bandits',
    experimental=True,
    contributor='jkcooper2',
    summary="10 armed bandit with that always pays out with a reward selected from a uniform distribution",
    description="""
        Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
        and a reward distribution, which is the value or distribution of what the agent will be rewarded
        the bandit does payout.

        p_dist = numpy.full(bandits, 1)
        r_dist = numpy.random.uniform(size=10)

        Bandits always pay out. Reward is selected from uniform distribution
        """,
    background="Based on comparisons from http://sudeepraja.github.io/Bandits/"
)

add_task(
    id='BanditTwoArmedIndependentUniform-v0',
    group='bandits',
    experimental=True,
    contributor='Thomas_Lecat',
    summary="Simple two independent armed bandit giving a reward of one with probabilities p_1 and p_2",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p_dist = np.random.uniform(2)
    r_dist = [1, 1]
    """,
    background="For the first experience, called 'Bandit with independent arms' of https://arxiv.org/abs/1611.05763"

add_task(
    id='BanditTwoArmedDependentUniform-v0',
    group='bandits',
    experimental=True,
    contributor='Thomas_Lecat',
    summary="Two armed bandit giving a reward of one with probabilities p_1 ~ U[0,1] and p_2 = 1 - p_1",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p = np.random.uniform()
    p_dist = [p, 1-p]
    r_dist = [1, 1]
    """,
    background="For the experience called 'Bandits with dependent arms (I)' of https://arxiv.org/abs/1611.05763"

add_task(
    id='BanditTwoArmedDependentEasy-v0',
    group='bandits',
    experimental=True,
    contributor='Thomas_Lecat',
    summary="Two armed bandit giving a reward of one with probabilities p_1 ~ U[0.1,0.9] and p_2 = 1 - p_1",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p = [0.1,0,9][np.random.randint(0,2)]
    p_dist = [p, 1-p]
    r_dist = [1, 1]
    """,
    background="For the experience called 'Bandits with dependent arms (I)' of https://arxiv.org/abs/1611.05763"

add_task(
    id='BanditTwoArmedDependentMedium-v0',
    group='bandits',
    experimental=True,
    contributor='Thomas_Lecat',
    summary="Two armed bandit giving a reward of one with probabilities p_1 ~ U[0.25,0.75] and p_2 = 1 - p_1",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p = [0.25,0,75][np.random.randint(0,2)]
    p_dist = [p, 1-p]
    r_dist = [1, 1]
    """,
    background="For the experience called 'Bandits with dependent arms (I)' of https://arxiv.org/abs/1611.05763"

add_task(
    id='BanditTwoArmedDependentHard-v0',
    group='bandits',
    experimental=True,
    contributor='Thomas_Lecat',
    summary="Two armed bandit giving a reward of one with probabilities p_1 ~ U[0.4,0.6] and p_2 = 1 - p_1",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    p = [0.4,0,6][np.random.randint(0,2)]
    p_dist = [p, 1-p]
    r_dist = [1, 1]
    """,
    background="For the experience called 'Bandits with dependent arms (I)' of https://arxiv.org/abs/1611.05763"

    add_task(
    id='BanditEleveArmedWithIndex-v0',
    group='bandits',
    experimental=True,
    contributor='Thomas_Lecat',
    summary="11 armed bandit with deterministic payouts. \
        Nine 'non-target' return a reward of 1.1, \
        one 'target' returns a reward of 5, \
        the 11th arm has reward = 0.1 * index of the target arm (ranging from 0.1 to 1.0)",
    description="""
    Each bandit takes in a probability distribution, which is the likelihood of the action paying out,
    and a reward distribution, which is the value or distribution of what the agent will be rewarded
    the bandit does payout.

    index = np.random.randint(0,10)
    p_dist = np.full(11,1)
    r_dist = np.full(11,1.1)
    r_dist[index] = 5
    r_dist[-1] = 0.1*index
    BanditEnv.__init__(self, p_dist = p_dist, r_dist = r_dist)
    """,
    background="For the experience called 'Bandits with dependent arms (II)' of https://arxiv.org/abs/1611.05763"
