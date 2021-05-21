from random import shuffle
import random
import chess
import numpy as np
from functools import reduce
from tensorflow.keras.models import clone_model
import copy
import controller
import timeit

# Shift random weights by val in range [-1,1]

def mutate_weights(weights, mr):
    rng = np.random.default_rng()
    for i in range(0, len(weights), 2):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                if rng.random() < mr:
                    amnt = (rng.random() * 2) - 1
                    weights[i][j][k] += amnt
    return weights

def mutate(model, mr):
    weights = mutate_weights(model.get_weights(), mr)
    out = clone_model(model)
    out.set_weights(weights)
    return out

def uniform_crossover(m1, m2, mutate=False, mr=None):
    weights = m1.get_weights()
    w2 = m2.get_weights()
    rng = np.random.default_rng()
    for i in range(0, len(weights), 2):
        for j in range(len(weights[i])):
            if rng.random() < 0.5:
                weights[i][j] = np.copy(w2[i][j])
    out = clone_model(m1)
    if mutate:
        weights = mutate_weights(weights, mr)
    out.set_weights(weights)
    return out

def gen_individual():
    r = np.random.default_rng()
    weights = np.array([
        r.random(size=(772,64)) * 10, 
        np.zeros(shape=(64,)),
        r.random(size=(64,1880)) * 10, 
        np.zeros(shape=(1880,))
    ], dtype=object)
    return controller.build_model(weights)

def gen_pop(size):
    return [gen_individual() for _ in range(size)]

# Stochastic universal selection
def next_generation(pop, fitnesses, crossover, mr):
    F = sum(fitnesses)
    N = len(pop) * (2 if crossover else 1) 
    P = F/N
    rng = np.random.default_rng()
    start = rng.random() * P
    pointers = [start + i*P for i in range(N)]

    # RWS
    new_pop = []
    for pointer in pointers:
        i = 0
        while sum(fitnesses[:i+1]) < pointer:
            i += 1
        if crossover:
            new_pop.append(pop[i])
        else:
            child = mutate(pop[i], mr)
            new_pop.append(child)
    print("Crossing")
    if crossover:
        shuffle(new_pop, random=rng.random)
        temp = []
        for i in range(0, len(new_pop), 2):
            p1 = new_pop[i]
            p2 = new_pop[i+1]
            child = uniform_crossover(p1, p2, mutate=True, mr=mr)
            temp.append(child)
        new_pop = temp
    return new_pop

def play_game(ind, opp):
    rng = np.random.default_rng()
    players = [ind, opp]
    next_player = rng.random() < 0.5
    col = chess.WHITE if players[next_player] == ind else chess.BLACK
    board = chess.Board()
    while not board.is_game_over():
        model = players[next_player]
        move = controller.choose_move(board, model)
        board.push(move)
        next_player = not next_player
    print(board.fen())
    if board.outcome().winner is not None:
        if board.outcome().winner == col:
            return 1
        else:
            return -1
    return 0

def evaluate_individual(ind, pop2, sample_size):
    rng = np.random.default_rng()
    sample = rng.choice(pop2, size=sample_size, replace=False)
    score = 0
    for opp in sample:
        score += play_game(ind, opp)
    return score

def evaluate_pops(pop1, pop2, sample_size):
    fit1 = []
    for i in range(len(pop1)):
        fit1.append(evaluate_individual(pop1[i], pop2, sample_size))
    fit2 = []
    for i in range(len(pop2)):
        fit2.append(evaluate_individual(pop2[i], pop1, sample_size))
    return fit1, fit2

def evolve(pop1, pop2, sample_size, mr, crossover, gens):
    for gen in range(gens):
        fit1, fit2 = evaluate_pops(pop1, pop2, sample_size)
        pop1 = next_generation(pop1, fit1, crossover, mr)
        pop2 = next_generation(pop2, fit2, crossover, mr)
    return pop1, pop2
mr = 1/171672
pop1 = gen_pop(100)
fit1 = list(range(100, 1, -1))
print("Spawning next gen")
print(timeit.timeit(lambda: next_generation(pop1, fit1, True, mr), number=1))