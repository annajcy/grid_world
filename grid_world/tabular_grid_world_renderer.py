from rl.renderer import Renderer
from .tabular_grid_world_mdp import TabularGridWorldMDP
from typing import Dict, Optional
import pygame
import math

class TabularGridWorldRenderer(Renderer[TabularGridWorldMDP]):
    def __init__(self, 
                 tabular_gw_mdp: TabularGridWorldMDP, 
                 caption: str = 'Tabular Grid World Renderer',
                 screen_width: int = 800, 
                 screen_height: int = 800, 
                 screen_width_margin: int = 50,
                 screen_height_margin: int = 50,
                 show_policy: bool = True,
                 show_values: bool = True) -> None:
        super().__init__(tabular_gw_mdp, caption, screen_width, screen_height)
        self.screen_width_margin = screen_width_margin
        self.screen_height_margin = screen_height_margin
        self.cell_width = (screen_width - 2 * screen_width_margin) // tabular_gw_mdp.state_space.width
        self.cell_height = (screen_height - 2 * screen_height_margin) // tabular_gw_mdp.state_space.height
        self.show_policy = show_policy
        self.show_values = show_values
        self.state_values: Optional[Dict] = None
        
    def update_state_values(self, state_values: Optional[Dict] = None) -> None:
        self.state_values = state_values
        
    def draw_grid(self) -> None:
        for x in range(self.mdp.state_space.width):
            for y in range(self.mdp.state_space.height):
                rect = pygame.Rect(
                    self.screen_width_margin + x * self.cell_width, 
                    self.screen_height_margin + y * self.cell_height, 
                    self.cell_width, 
                    self.cell_height)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)
                
    def draw_goal(self) -> None:
        goal = self.mdp.goal_state
        rect = pygame.Rect(
            self.screen_width_margin + goal.x * self.cell_width, 
            self.screen_height_margin + goal.y * self.cell_height, 
            self.cell_width, 
            self.cell_height)
        pygame.draw.rect(self.screen, (0, 255, 0), rect)
        
        # Draw "G" for goal
        text = self.font.render('G', True, (0, 100, 0))
        text_rect = text.get_rect(center=(
            self.screen_width_margin + goal.x * self.cell_width + self.cell_width // 2,
            self.screen_height_margin + goal.y * self.cell_height + self.cell_height // 2
        ))
        self.screen.blit(text, text_rect)
    
    def draw_initial_state(self) -> None:
        initial = self.mdp.initial_state
        rect = pygame.Rect(
            self.screen_width_margin + initial.x * self.cell_width, 
            self.screen_height_margin + initial.y * self.cell_height, 
            self.cell_width, 
            self.cell_height)
        pygame.draw.rect(self.screen, (173, 216, 230), rect)
        
        # Draw "S" for start
        text = self.font.render('S', True, (0, 0, 100))
        text_rect = text.get_rect(center=(
            self.screen_width_margin + initial.x * self.cell_width + self.cell_width // 2,
            self.screen_height_margin + initial.y * self.cell_height + self.cell_height // 2
        ))
        self.screen.blit(text, text_rect)
    
    def draw_agent(self) -> None:
        agent_pos = self.mdp.current_state.position()
        center_x = self.screen_width_margin + agent_pos[0] * self.cell_width + self.cell_width // 2
        center_y = self.screen_height_margin + agent_pos[1] * self.cell_height + self.cell_height // 2
        radius = min(self.cell_width, self.cell_height) // 4
        pygame.draw.circle(self.screen, (0, 0, 255), (center_x, center_y), radius)

    def draw_policy_arrows(self) -> None:
       
        if not self.show_policy:
            return
            
        arrow_length = min(self.cell_width, self.cell_height) // 3
        
        for state in self.mdp.state_space.to_list():
            state_policy = self.mdp.get_state_policy(state)
            
            # Find the action with highest probability
            best_action = max(state_policy.items(), key=lambda x: x[1])[0]
            best_prob = state_policy[best_action]
            
            # Only draw if there's a clear best action (probability > 0.5)
            if best_prob > 0.5:
                center_x = self.screen_width_margin + state.x * self.cell_width + self.cell_width // 2
                center_y = self.screen_height_margin + state.y * self.cell_height + self.cell_height // 2
                
                # Calculate arrow end position
                end_x = center_x + best_action.dx * arrow_length
                end_y = center_y + best_action.dy * arrow_length
                
                # Draw arrow
                self._draw_arrow(center_x, center_y, end_x, end_y, (255, 0, 0), 2)
    
    def _draw_arrow(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                    color: tuple, thickness: int) -> None:
        # Draw line
        pygame.draw.line(self.screen, color, (start_x, start_y), (end_x, end_y), thickness)
        
        # Draw arrowhead
        angle = math.atan2(end_y - start_y, end_x - start_x)
        arrow_angle = math.pi / 6  # 30 degrees
        arrow_length = 8
        
        # Left side of arrowhead
        left_x = end_x - arrow_length * math.cos(angle - arrow_angle)
        left_y = end_y - arrow_length * math.sin(angle - arrow_angle)
        
        # Right side of arrowhead
        right_x = end_x - arrow_length * math.cos(angle + arrow_angle)
        right_y = end_y - arrow_length * math.sin(angle + arrow_angle)
        
        pygame.draw.polygon(self.screen, color, [(end_x, end_y), (left_x, left_y), (right_x, right_y)])
    
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
    
    def draw_legend(self) -> None:
        legend_y = 10
        legend_x = 10
        small_font = pygame.font.SysFont('Arial', 14)
        
        legends = [
            ("Green: Goal", (0, 150, 0)),
            ("Light Blue: Start", (0, 0, 150)),
            ("Blue Circle: Agent", (0, 0, 255)),
            ("Red Arrow: Policy", (255, 0, 0)),
        ]
        
        for i, (text_str, color) in enumerate(legends):
            text = small_font.render(text_str, True, color)
            self.screen.blit(text, (legend_x, legend_y + i * 20))

    def render(self, fps: int = 30) -> None:
        self.screen.fill((255, 255, 255))  # White background

        self.draw_goal()
        self.draw_initial_state()
        self.draw_grid()
        self.draw_state_values()
        self.draw_policy_arrows()
        self.draw_agent()
        self.draw_legend()

        pygame.display.flip()
        self.clock.tick(fps)
