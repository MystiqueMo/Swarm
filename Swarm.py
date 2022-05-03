import math, random, copy
import numpy as np
from itertools import repeat, count
from multiprocessing import Pool
from operator import attrgetter

class Particle:
    def __init__(self, variableLimits, variableTypes, *inputs, params={'w': .8, 'c1': .1, 'c2': .1}):
        self.variableLimits = variableLimits
        self.variableTypes = variableTypes
        self.myParams = params
        self.myPosition = np.array([inputs]).transpose()
        self.myBestPosition = np.copy(self.myPosition)
        self.myBestScore = math.inf
        self.myVelocity = np.random.normal(loc=0., scale=5., size=(len(inputs), 1))
        self.isEvaluated = False
        
    def __str__(self):
        return f"\nMy Position: {list(self.myPosition.flatten())}\nMy Velocity: {list(self.myVelocity.flatten())}\n"
    
    def move(self):
        leftBound, rightBound, types = (
                                            np.array(self.variableLimits)[: , : 1],
                                            np.array(self.variableLimits)[: , 1: ],
                                            np.array([self.variableTypes]).transpose()
                                        )
        self.myPosition = np.add(self.myPosition, self.myVelocity)
        self.myPosition = np.where(self.myPosition > rightBound, rightBound, self.myPosition)
        self.myPosition = np.where(self.myPosition < leftBound, leftBound, self.myPosition)
        self.myPosition = np.where(types == 'int', np.rint(self.myPosition), self.myPosition)

    def accelerate(self, globalBestPosition):
        self.myVelocity = (self.myParams['w'] * self.myVelocity) + (self.myParams['c1'] * random.random() * (self.myBestPosition - self.myPosition)) + (self.myParams['c2'] * random.random() * (globalBestPosition - self.myPosition))


class Particles:
    def __init__(self, lossFunction, numberOfVariables, variableLimits, variableTypes, speedLimit=.25, verbose=False, detoxify=False, params={'w': .8, 'c1': .1, 'c2': .1}):
        if not(len(variableLimits) == numberOfVariables) or not(len(variableTypes) == numberOfVariables):
            if not(len(variableLimits) == numberOfVariables) and len(variableTypes) == numberOfVariables:
                culprit = "Variable Limits"
            elif not(len(variableTypes) == numberOfVariables) and len(variableLimits) == numberOfVariables:
                culprit = "Variable Types"
            else:   culprit = "Variable Limits and Variable Types"
            print(f"\n{culprit} Defined do not Match with the Number of Variables...")
            return
        
        self.lossFunction = lossFunction
        self.dimension = numberOfVariables
        self.variableLimits = variableLimits
        self.variableTypes = variableTypes
        self.leftBounds = np.array(self.variableLimits)[: , : 1]
        self.rightBounds = np.array(self.variableLimits)[: , 1: ]
        self.typeBounds = np.array([self.variableTypes]).transpose()
        self.speedLimit = speedLimit
        self.detoxify = detoxify
        self.verbose = verbose
        self.myParams = copy.deepcopy(params)
        self.particles = list()

    def isExploiting(self):
        # this membership represents exploitation...
        return .2 < self.f <= .6

    def normalize(self, vector):
        return np.divide(np.subtract(vector, self.leftBounds), np.subtract(self.rightBounds, self.leftBounds))
        
    def abnormalize(self, vector):
        result = np.add(self.leftBounds, np.multiply(vector, np.subtract(self.rightBounds, self.leftBounds)))
        return np.where(self.typeBounds == 'int', np.rint(result), result)
        
    def initializeSwarm(self, particleCount=10):
        self.particleCount = particleCount
        for _ in range(self.particleCount):
            position = [(random.uniform(limit[0], limit[1]) if self.variableTypes[self.variableLimits.index(limit)] == 'real' else random.randint(limit[0], limit[1])) for limit in self.variableLimits]
            self.particles.append(Particle(self.variableLimits, self.variableTypes, *position, params=copy.deepcopy(self.myParams)))
        
    def evaluateSwarm(self, secondary_args=None):
        candidateList = [list(particle.myPosition.flatten()) for particle in self.particles]
        if not isinstance(secondary_args, type(None)):
            argRepitition = [repeat(arg, self.particleCount) for arg in secondary_args]
            with Pool(processes=self.particleCount) as p:
                scores = p.starmap(self.lossFunction, zip(candidateList, *argRepitition))
        else:
            with Pool(processes=self.particleCount) as p:
                scores = p.map(self.lossFunction, candidateList)
        
        i = np.argmin(scores)
        self.globalBestPosition = np.array([candidateList[i]]).transpose()
        j = 0
        di = 0.
        dmin = math.inf
        dmax = -math.inf
        
        for particle, score in zip(self.particles, scores):
            if score < particle.myBestScore:
                particle.myBestPosition = np.array([candidateList[scores.index(score)]]).transpose()
                particle.myBestScore = score
            particle.isEvaluated = True
            dj = np.linalg.norm(self.globalBestPosition - particle.myPosition)
            dmin = dj if dj < dmin else dmin
            dmax = dj if dj > dmax else dmax
            if not(j == i):
                di += (dj - di) / float(j + 1)
                j += 1
        
        self.f = (di - dmin) / (dmax - dmin)
    
        return self.globalBestPosition
    
    def detoxifySwarm(self):
        print("\nDetoxifying Swarm...")
        self.particles = sorted(self.particles, key=attrgetter('myBestScore'))
        [self.particles.pop(-1) for _ in range(self.particleCount // 2)]
        for _ in range(self.particleCount - len(self.particles)):
            position = [(random.uniform(limit[0], limit[1]) if self.variableTypes[self.variableLimits.index(limit)] == 'real' else random.randint(limit[0], limit[1])) for limit in self.variableLimits]
            self.particles.append(Particle(self.variableLimits, self.variableTypes, *position, params=copy.deepcopy(self.myParams)))
              
    def accelerateSwarm(self):
        [particle.accelerate(self.globalBestPosition) for particle in self.particles if particle.isEvaluated]
    
    def moveSwarm(self):
        [particle.move() for particle in self.particles if particle.isEvaluated]
        
    def swarm(self, convergenceAccuracy, learningRate=.1, secondary_args=None):
        self.typeBounds = np.array([self.variableTypes]).transpose()
        solution = self.evaluateSwarm(secondary_args=secondary_args)
        for i in count(0):
            self.accelerateSwarm()
            self.moveSwarm()
            pastSolution = np.copy(solution)
            solution = self.evaluateSwarm(secondary_args=secondary_args)
            change = np.subtract(solution, pastSolution) if i == 0 else change + learningRate * (np.subtract(solution, pastSolution) - change)
            
            if self.verbose:    print(f"\nSolution, Change: {np.where(self.typeBounds == 'int', np.rint(solution), solution).flatten()}, {np.linalg.norm(change)}")
            
            if np.linalg.norm(change) < convergenceAccuracy:
                return list(np.where(self.typeBounds == 'int', np.rint(solution), solution).flatten())
            elif self.detoxify and self.f == 0.:   self.detoxifySwarm()
