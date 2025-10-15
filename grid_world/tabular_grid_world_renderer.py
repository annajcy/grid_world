from typing import Dict, Optional
import pygame

from .tabular_grid_world_mdp import TabularGridWorldMDP
from .grid_world_renderer import GridWorldRenderer

class TabularGridWorldRenderer(GridWorldRenderer[TabularGridWorldMDP]):
    def __init__(self, 
                 tabular_gw_mdp: TabularGridWorldMDP, 
                 caption: str = 'Tabular Grid World Renderer',
                 screen_width: int = 800, 
                 screen_height: int = 800, 
                 screen_width_margin: int = 50,
                 screen_height_margin: int = 50,
                 show_policy: bool = True,
                 show_values: bool = True) -> None:
        super().__init__(tabular_gw_mdp, caption, screen_width, screen_height, 
                         screen_width_margin, screen_height_margin)
        self.show_policy = show_policy
        self.show_values = show_values
        self.state_values: Optional[Dict] = None
        
    def update_state_values(self, state_values: Optional[Dict] = None) -> None:
        self.state_values = state_values
        
    def draw_policy_arrows(self) -> None:
        if not self.show_policy:
            return
            
        max_arrow_length = min(self.cell_width, self.cell_height) // 3
        min_arrow_length = max_arrow_length // 4  # Minimum arrow length for visibility
        
        for state in self.mdp.state_space.to_list():
            state_policy = self.mdp.get_state_policy(state)
            
            center_x = self.screen_width_margin + state.x * self.cell_width + self.cell_width // 2
            center_y = self.screen_height_margin + state.y * self.cell_height + self.cell_height // 2
            
            # Draw arrows for all actions with non-zero probability
            for action, prob in state_policy.items():
                if prob > 0.01:  # Only draw if probability is significant
                    # Arrow length proportional to probability
                    arrow_length = min_arrow_length + (max_arrow_length - min_arrow_length) * prob
                    
                    # Calculate arrow end position
                    end_x = int(center_x + action.dx * arrow_length)
                    end_y = int(center_y + action.dy * arrow_length)
                    
                    # Arrow thickness also proportional to probability
                    thickness = max(1, int(3 * prob))
                    
                    # Draw arrow
                    self._draw_arrow(center_x, center_y, end_x, end_y, (255, 0, 0), thickness)
    
    def draw_state_values(self) -> None:
        if not self.show_values or self.state_values is None:
            return
            
        small_font = pygame.font.SysFont('Arial', 14)
        
        for state in self.mdp.state_space.to_list():
            if state in self.state_values:
                value = self.state_values[state]
                
                # Format value to 2 decimal places
                value_text = f"{value:.2f}"
                text = small_font.render(value_text, True, (50, 50, 50))
                
                # Position in top-left corner of cell
                text_x = self.screen_width_margin + state.x * self.cell_width + 5
                text_y = self.screen_height_margin + state.y * self.cell_height + 5
                
                self.screen.blit(text, (text_x, text_y))
    
    def render(self, fps: int = 30) -> None:
        self.screen.fill((255, 255, 255))  # White background

        self.draw_goal()
        self.draw_initial_state()
        self.draw_grid()
        self.draw_state_values()
        self.draw_policy_arrows()
        self.draw_agent()
        # self.draw_legend()

        pygame.display.flip()
        self.clock.tick(fps)
