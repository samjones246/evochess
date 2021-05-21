import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import chess
from functools import reduce

def pawn_promos(color):
    out = []
    if color == chess.WHITE:
        a = 48
        b = 56
    else:
        a = 8
        b = 16
    for i in range(a, b):
        if color == chess.WHITE:
            left = max(56, i+7)
            right = min(63, i+9) + 1
        else:
            left = max(0, i-9)
            right = min(7, i-7) + 1
        for j in range(left, right):
            for p in ["q","r","b","n"]:
                out.append(chess.square_name(i) + chess.square_name(j) + p)
    return out

def queen_moves():
    out = []
    for i in range(64):
        start = chess.square_name(i)
        # Left -> Right
        rank_start = (i // 8) * 8
        for j in range(rank_start, rank_start+8):
            if i == j:
                continue
            end = chess.square_name(j)
            out.append(start + end)

        # Bottom -> Top
        file_start = i % 8
        for j in range(file_start, 64, 8):
            if i == j:
                continue
            end = chess.square_name(j)
            out.append(start + end)

        # Diagonal 1
        if i != 7 and i != 56:
            if chess.square_rank(i) == 7 or chess.square_file(i) == 7:
                align = i - 9
            else:
                align = i + 9
            diagonal = list(chess.SquareSet.ray(i, align))
            for j in diagonal:
                if i == j:
                    continue
                end = chess.square_name(j)
                out.append(start + end)
        # Diagonal 2
        if i != 0 and i != 63:
            if chess.square_rank(i) == 7 or chess.square_file(i) == 0:
                align = i - 7
            else:
                align = i + 7
            diagonal = list(chess.SquareSet.ray(i, align))
            for j in diagonal:
                if i == j:
                    continue
                end = chess.square_name(j)
                out.append(start + end)
    return out

def knight_moves():
    out = []
    for i in range(64):
        start = chess.square_name(i)
        # Up
        if chess.square_rank(i) < 6:
            if chess.square_file(i) != 7:
                out.append(start + chess.square_name(i+17))
            if chess.square_file(i) != 0:
                out.append(start + chess.square_name(i+15))
        # Right
        if chess.square_file(i) < 6:
            if chess.square_rank(i) != 7:
                out.append(start + chess.square_name(i+10))
            if chess.square_rank(i) != 0:
                out.append(start + chess.square_name(i-6))
        # Down
        if chess.square_rank(i) > 1:
            if chess.square_file(i) != 7:
                out.append(start + chess.square_name(i-15))
            if chess.square_file(i) != 0:
                out.append(start + chess.square_name(i-17))
        # Left
        if chess.square_file(i) > 1:
            if chess.square_rank(i) != 7:
                out.append(start + chess.square_name(i+6))
            if chess.square_rank(i) != 0:
                out.append(start + chess.square_name(i-10))
    return out

def gen_moves_list(color):
    return pawn_promos(color) + queen_moves() + knight_moves()

MOVE_LISTS = [gen_moves_list(chess.BLACK),gen_moves_list(chess.WHITE)]

def get_board_bits(board : chess.Board):
    out = [board.pieces(p, chess.WHITE).tolist() for p in chess.PIECE_TYPES]
    out += [board.pieces(p, chess.BLACK).tolist() for p in chess.PIECE_TYPES]
    out = list(reduce(lambda a,b:a+b, out, []))
    out += [board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK)]
    return list(map(int,out))

# Weights: [[(772,64),(64,)],[(64,1880),(1880,)]]
def build_model(weights=None):
    inputs = keras.Input(shape=(772,))
    lh = layers.Dense(64, activation="relu")
    lo = layers.Dense(1880)
    outputs = lo(lh(inputs))
    model = keras.Model(inputs=inputs, outputs=outputs)
    if weights is not None:
        model.set_weights(weights)
    return model

def choose_move(board, model):
    inp = [get_board_bits(board)]
    out = model.predict(inp)
    sorted_moves = sorted(zip(out[0], range(1880)), key=lambda x: x[0], reverse=True)
    move = None
    for i in range(1880):
        uci = MOVE_LISTS[board.turn][sorted_moves[i][1]]
        move = chess.Move.from_uci(uci)
        if board.is_legal(move):
            break
    return move