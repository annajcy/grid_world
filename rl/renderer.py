import pygame
from typing import Generic, TypeVar
from abc import ABC, abstractmethod
from rl.mdp import MDP

MDPType = TypeVar('MDPType', bound=MDP)

class Renderer(ABC, Generic[MDPType]):
    def __init__(self, 
                 mdp: MDPType, 
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