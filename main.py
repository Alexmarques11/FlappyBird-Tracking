import asyncio
import sys
import cv2
import pygame
from src.flappy import Flappy

if __name__ == "__main__":
    game = Flappy()
    asyncio.run(game.start())
