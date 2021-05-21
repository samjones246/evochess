import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from random import shuffle
import random
import chess
import numpy as np
from functools import reduce
from tensorflow import keras
from tensorflow.keras.models import clone_model
import copy
import controller
import timeit
from tqdm import tqdm
from itertools import count

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
    for move_num in tqdm(count(0),"Move:",leave=False,unit=""):
        model = players[next_player]
        move = controller.choose_move(board, model)
        board.push(move)
        next_player = not next_player
        if board.is_game_over():
            break
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
    for i in tqdm(range(sample_size), "Playing Game", leave=False):
        opp = sample[i]
        outcome = play_game(ind, opp)
        score += outcome
    return score

def evaluate_pops(pop1, pop2, sample_size):
    fits = []
    pops = [pop1, pop2]
    for i in tqdm(range(len(pops)), "Evaluating Population", leave=False):
        pop = pops[i]
        fit = []
        for j in tqdm(range(len(pop)), "Evaluating Individual", leave=False):
            fit.append(evaluate_individual(pop[j], pops[not i], sample_size))
        fits.append(fit)
    return tuple(fits)

def evolve(pop1, pop2, sample_size, mr, crossover, gens):
    for gen in tqdm(range(gens), "Generation"):
        fit1, fit2 = evaluate_pops(pop1, pop2, sample_size)
        pop1 = next_generation(pop1, fit1, crossover, mr)
        pop2 = next_generation(pop2, fit2, crossover, mr)
    fit1, fit2 = evaluate_pops(pop1, pop2, sample_size)
    return (pop1,fit1), (pop2,fit2)

def best_individual(pop1, fit1, pop2, fit2) -> keras.Model:
    best1 = max(zip(pop1, fit1), key=lambda x: x[1])
    best2 = max(zip(pop2, fit2), key=lambda x: x[1])
    return max(best1, best2, key=lambda x: x[1])[0]

mr = 1/171672
pop1 = gen_pop(100)
pop2 = gen_pop(100)

popfit1, popfit2 = evolve(pop1, pop2, 10, mr, True, 10)
best = best_individual(popfit1[0], popfit1[1], popfit2[0], popfit2[1])
best.save("model.h5")