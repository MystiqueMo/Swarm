from math import cos, exp, pi, e, sqrt, sin
from Swarm import Particles

variableLimits = (
                    (-10, 10),
                    (-10, 10)
                    )

variableTypes = (
                    'real',
                    'real'
                    )

def lossFunction(X):
    # return (1 - X[0])**2 + 100 * (X[1] - X[0]**2)**2
    # return (X[0] - 5.7)**2 + (X[1] - 5)**2
    # return 0.26 * (X[0]**2 + X[1]**2) - 0.48 * X[0] * X[1]
    # return -cos(X[0]) * cos(X[1]) * exp(-((X[0] - pi)**2 + (X[1] - pi)**2))
    # return -20.0 * exp(-0.2 * sqrt(0.5 * (X[0]**2 + X[1]**2))) - exp(0.5 * (cos(2 * pi * X[0]) + cos(2 * pi * X[1]))) + e + 20
    return (X[0]**2 + X[1] - 11)**2 + (X[0] + X[1]**2 - 7)**2
    # return (X[0]-3.14)**2 + (X[1]-2.72)**2 + np.sin(3*X[0]+1.41) + np.sin(4*X[1]-1.73)

if __name__ == '__main__':
    learningRate = 0.1
    swarm = Particles(lossFunction=lossFunction, numberOfVariables=2,
                    variableLimits=variableLimits,
                    variableTypes=variableTypes,
                    detoxify=True, verbose=True)
    
    swarm.initializeSwarm(particleCount=10)
    solution = swarm.swarm(1e-6, learningRate=learningRate)
    del swarm
