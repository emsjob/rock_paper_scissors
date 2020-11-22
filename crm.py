import os
import numpy as np
import matplotlib.pyplot as plt
import random 

# --- CONTERFACTUAL REGRET MINIMIZATION ---
# --- AI to solve imperfect information games ---


# -- FUNCTIONS --
def value(p1, p2, NUM_ACTIONS):
	if p1 == p2:
		return 0
	elif (p1-1) % NUM_ACTIONS == p2:
		return 1
	else:
		return -1

def normalize(strategy):
	strategy = np.copy(strategy)
	normalizingSum = np.sum(strategy)
	if normalizingSum > 0:
		strategy /= normalizingSum
	else:
		strategy = np.ones(strategy.shape[0])/strategy.shape[0]
	return strategy

def getStrategy(regretSum):
	return normalize(np.maximum(regretSum, 0))

def getAction(strategy):
	strategy /= np.sum(strategy)
	return np.searchsorted(np.cumsum(strategy), random.random())

def innertrain(regretSum, strategySum, oppStrategy, NUM_ACTIONS):
	# accumulates the current strategy based on regret
	strategy = getStrategy(regretSum)
	strategySum += strategy

	# select my action and opponent actiob
	myAction = getAction(strategy)
	otherAction = getAction(oppStrategy)

	# for rock, paper, scissors
	actionUtility = np.zeros(NUM_ACTIONS)
	actionUtility[otherAction] = 0		
	actionUtility[(otherAction + 1) % NUM_ACTIONS] = 1
	actionUtility[(otherAction - 1) % NUM_ACTIONS]  = -1

	regretSum += actionUtility - actionUtility[myAction]

	return regretSum, strategySum,

def train(iterations, NUM_ACTIONS):
	regretSum = np.zeros(NUM_ACTIONS)
	strategySum = np.zeros(NUM_ACTIONS)
	oppStrategy = np.array([0.4, 0.3, 0.3])
	for i in range(iterations):
		regretSum, strategySum, = innertrain(regretSum, strategySum, oppStrategy, NUM_ACTIONS)
	return strategySum

def train2p(iterations, NUM_ACTIONS):
	regretSumP1 = np.zeros(NUM_ACTIONS)
	strategySumP1 = np.zeros(NUM_ACTIONS)
	regretSumP2 = np.zeros(NUM_ACTIONS)
	strategySumP2 = np.zeros(NUM_ACTIONS)
	for i in range(iterations):
		# train P1
		oppStrategy = normalize(strategySumP2)
		regretSuP1, strategySumP1, = innertrain(regretSumP1, strategySumP1, oppStrategy, NUM_ACTIONS)

		# train P2
		oppStrategy = normalize(strategySumP1)
		regretSuP2, strategySumP2, = innertrain(regretSumP2, strategySumP2, oppStrategy, NUM_ACTIONS)

	return strategySumP1, strategySumP2


 # -- MAIN FUNCTION --
def main():
	# ROCK PAPER SCISSORS PROBABILITIES
	ROCK, PAPER, SCISSORS = 0,1,2
	NUM_ACTIONS = 3
	regretSum = np.zeros(NUM_ACTIONS)
	strategySum = np.zeros(NUM_ACTIONS)
	strategySum = train(100000, NUM_ACTIONS)
	oppStrategy = np.array([0.4, 0.3, 0.3])
	strategy = normalize(strategySum)
	vvv = []
	for j in range(100):
		vv = 0
		for i in range(100):
			myAction = getAction(strategy)
			otherAction = getAction(oppStrategy)
			vv += value(myAction, otherAction, NUM_ACTIONS)
		vvv.append(vv)

	print(np.mean(vvv), np.median(vvv))
	plt.plot(sorted(vvv))
	plt.show()

if __name__ == "__main__":
	main()
