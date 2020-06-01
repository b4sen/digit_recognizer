from model_loader import ModelLoader
from image_processor import ImageProcessor
import pygame as pg 
import sys
import torch
import numpy as np
WIDTH = HEIGHT = 680
class Game:

    def __init__(self, width, height):
        pg.init()
        self.W = width
        self.H = height
        self.screen = pg.display.set_mode((self.W, self.H))
        self.running = True
        self.scl = 40
        self.cols = self.W//self.scl
        self.rows = self.H//self.scl
        self.grid = [[0 for i in range(self.cols)] for j in range(self.rows)]
        self.ip = ImageProcessor()
        self.model = ModelLoader('model/trained_model.pt').load_model()

    def draw_grid(self):
        black = 0, 0, 0
        for col in range(self.cols):
            var = (col * self.scl) - 1
            pg.draw.line(self.screen, black, (var, 0), (var, self.H), 1)

        for row in range(self.rows):
            var = (row * self.scl) - 1
            pg.draw.line(self.screen, black, (0, var), (self.W, var), 1)
    
    def draw_node(self):
        if pg.mouse.get_pressed()[0]:
            x,y = pg.mouse.get_pos()
            x = int(x // self.scl)
            y = int(y // self.scl)
            pg.draw.rect(self.screen, 0, (x * self.scl, y * self.scl, self.scl, self.scl ))
            self.grid[y][x] = 255

        for c in range(self.cols):
            for r in range(self.rows):
                if self.grid[r][c] == 255:
                    pg.draw.rect(self.screen, 0, (c * self.scl, r * self.scl, self.scl, self.scl ))
    def reset(self):
        self.__init__(self.W, self.H)
    
    def run(self):
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_RETURN:
                        self.img = self.ip.process_image(np.array(self.grid))
                        out = self.model(torch.tensor(self.img)).detach().numpy()
                        print(f'Prediction: {np.argmax(out)}')
                        self.reset()
            self.screen.fill((255,255,255))
            self.draw_grid()
            self.draw_node()
            pg.display.flip()

if __name__ == "__main__":
    game = Game(WIDTH,HEIGHT)
    game.run()