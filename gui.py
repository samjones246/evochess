import tkinter as tk
import tensorflow as tf
from tensorflow import keras
from litemodel import LiteModel
import chess
import chess.svg
import sys
from PIL import Image
import PIL.ImageTk as ImageTk
import cairosvg
import io
import controller

class MainWindow(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.board = chess.Board()
        self.columnconfigure(index=0, weight=1)
        self.rowconfigure(index=0, weight=1)
        self.canvas = tk.Canvas(self, width=400, height=400)
        self.input = tk.Entry(self)
        self.input.bind("<Return>", self.player_move)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.input.grid(row=1, column=0, sticky="ew")
        self.imgid = None
        self.redraw()
        self.load_model()
        
    def load_model(self):
        model = keras.models.load_model("model-10gen.h5")
        self.model = LiteModel.from_keras_model(model)

    def player_move(self, event):
        try:
            move = chess.Move.from_uci(self.input.get())
        except:
            return
        if not self.board.is_legal(move):
            return
        self.input.delete(0, tk.END)
        self.board.push(move)
        self.redraw()
        if self.board.is_game_over():
            self.game_over(True)
            return
        self.model_move()

    def model_move(self):
        move = controller.choose_move(self.board, self.model)
        self.board.push(move)
        self.redraw()
        if self.board.is_game_over():
            self.game_over(False)

    def game_over(self, win):
        self.input.delete(0, tk.END)
        self.input.insert(0, "YOU WIN" if win else "YOU LOSE")
        self.input.configure(state="disabled")
    
    def redraw(self):
        if self.imgid is not None:
            self.canvas.delete(self.imgid)
        svg = chess.svg.board(self.board).encode("utf-8")
        img_data = cairosvg.svg2png(bytestring=svg)#, parent_width=self.canvas.winfo_width(), parent_height=self.canvas.winfo_height())
        image = Image.open(io.BytesIO(img_data))
        tk_image = ImageTk.PhotoImage(image)
        self.imgid = self.canvas.create_image(0, 0, anchor="nw", image=tk_image)
        self.canvas.image = tk_image



if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root).pack(side="top", fill="both", expand=True)
    root.mainloop()