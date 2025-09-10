from rl.renderer import Renderer
from rl.mdp import MDP
from .grid_world_mdp import GridWorldState, GridWorldAction, GridWorldStateSpace, GridWorldActionSpace, GridWorldMDP
import pygame

class GridWorldRenderer(Renderer[GridWorldMDP]):
    def __init__(self, 
                 gw_mdp: GridWorldMDP, 
                 caption: str = 'Grid World Renderer',
                 screen_width: int = 800, 
                 screen_height: int = 800, 
                 screen_width_margin: int = 5,
                 screen_height_margin: int = 5) -> None:
        super().__init__(gw_mdp, caption, screen_width, screen_height)
        self.screen_width_margin = screen_height_margin
        self.screen_height_margin = screen_height_margin
        self.cell_width = (screen_width - 2 * screen_width_margin) // gw_mdp.state_space.width
        self.cell_height = (screen_height - 2 * screen_height_margin) // gw_mdp.state_space.height
        
    def draw_grid(self) -> None:
        # Draw grid
        for x in range(self.mdp.state_space.width):
            for y in range(self.mdp.state_space.height):
                rect = pygame.Rect(
                    self.screen_width_margin + x * self.cell_width, 
                    self.screen_height_margin + y * self.cell_height, 
                    self.cell_width, 
                    self.cell_height)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)  # Light gray grid lines
                
    def draw_forbiddens(self) -> None:
        for forbidden in self.mdp.forbiddens:
            rect = pygame.Rect(
                self.screen_width_margin + forbidden.x * self.cell_width, 
                self.screen_height_margin + forbidden.y * self.cell_height, 
                self.cell_width, 
                self.cell_height)
            pygame.draw.rect(self.screen, (255, 0, 0), rect)  # Red forbidden cells
    
    def draw_agent(self) -> None:
        agent_pos = self.mdp.current_state.position()
        agent_rect = pygame.Rect(
            self.screen_width_margin + agent_pos[0] * self.cell_width, 
            self.screen_height_margin + agent_pos[1] * self.cell_height, 
            self.cell_width, 
            self.cell_height)
        pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)

    def render(self, fps: int = 1) -> None:
        self.screen.fill((255, 255, 255))  # White background

        self.draw_grid()
        self.draw_forbiddens()
        self.draw_agent()

        pygame.display.flip()
        
        pygame.display.flip()
        self.clock.tick(fps)
