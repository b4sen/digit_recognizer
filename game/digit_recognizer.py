from model_loader import ModelLoader
from image_processor import ImageProcessor
import pygame as pg 

class Game:

    def __init__(self, width, height):
        pg.init()
        self.W = width
        self.H = height
        self.screen = pg.display.set_mode((self.W, self.H))
        self.running = True

    def run(self):
        while self.running:
            for event in pg.event.get():
            # only do something if the event is of type QUIT
                if event.type == pg.QUIT:
                    # change the value to False, to exit the main loop
                    running = False


if __name__ == "__main__":
    game = Game(400,400)
    game.run()