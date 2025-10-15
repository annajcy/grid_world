from typing import Any
import pygame
from abc import ABC, abstractmethod

class Renderer(ABC):
    def __init__(self, 
                 mdp: Any, 
                 caption: str = 'RL Renderer',
                 screen_width: int = 800,
                 screen_height: int = 800) -> None:

        self.mdp = mdp
        self.screen_width = screen_width
        self.screen_height = screen_height
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 20)
        self.running = True

    @abstractmethod
    def render(self) -> None:
        pass

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
    def close(self) -> None:
        pygame.quit()