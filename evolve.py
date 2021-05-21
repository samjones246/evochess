from os import replace
import chess
import numpy as np
from functools import reduce
import copy
import controller

def mutate(weights, mr):
    out = copy.deepcopy(weights)
    rng = np.random.default_rng()
    for i in range(len(out)):
        for j in range(len(out[i][0])):
            for k in range(len(out[i][0][j])):
                if rng.random() < mr:
                    amnt = (rng.random() * 2) - 1
                    out[i][0][j][k] += amnt
    return out

def crossover(w1, w2):
    out = copy.deepcopy(w1)
    rng = np.random.default_rng()
    for i in range(len(out)):
        for j in range(len(out[i][0])):
            if rng.random() < 0.5:
                out[i][0][j] = np.copy(w2[i][0][j])
    return out

def gen_individual():
    r = np.random.default_rng()
    weights = np.array([
        np.array([r.random(size=(772,64)) * 10, np.zeros(shape=(64,))], dtype=object),
        np.array([r.random(size=(64,1880)) * 10, np.zeros(shape=(1880,))], dtype=object)
    ], dtype=object)
    return weights

def gen_pop(size):
    pop = []
    for _ in range(size):
        pop.append(gen_individual())
    return pop

# Stochastic universal selection
def next_generation(pop, fitnesses, crossover, mr):
    pass

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

def pop_models(pop):
    return [controller.build_model(w) for w in pop]

def evaluate_pops(pop1, pop2, sample_size):
    mods1 = pop_models(pop1)
    mods2 = pop_models(pop2)
    fit1 = []
    for i in range(len(pop1)):
        fit1.append(evaluate_individual(mods1[i], mods2, sample_size))
    fit2 = []
    for i in range(len(pop2)):
        fit1.append(evaluate_individual(mods2[i], mods1, sample_size))
    return fit1, fit2

def evolve(pop1, pop2, sample_size, mr, crossover, gens):
    pass

p1 = controller.build_model(gen_individual())
p2 = controller.build_model(gen_individual())
print("Starting game...")
outcome = play_game(p1, p2)
print("Done")
print("WIN" if outcome is 1 else ("LOSE" if outcome is -1 else "DRAW"))