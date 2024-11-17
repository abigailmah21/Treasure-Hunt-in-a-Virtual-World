import pygame
import random
import numpy as np
import heapq
from itertools import permutations
import copy

# Initialize pygame
pygame.init()

# Define the board size and hexagon parameters
rows = 6
cols = 10
hex_radius = 30  # Smaller radius to fit the window better
hex_height = np.sqrt(3) * hex_radius
hex_width = 2 * hex_radius * 3 / 4

# Set up the display
border_size = 50
screen_width = int(cols * hex_width + hex_radius + border_size * 2)
screen_height = int(rows * hex_height + hex_radius + border_size * 2 + 150)  # Extra space for text
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Treasure Hunt Game")

# Define colors
colors = {
    'E': (255, 255, 255),  # Entry
    'T1': (218, 112, 214),  # Trap 1
    'T2': (186, 85, 211),  # Trap 2
    'T3': (153, 50, 204),  # Trap 3
    'T4': (128, 0, 128),  # Trap 4
    'R1': (32, 178, 170),  # Reward 1
    'R2': (60, 179, 113),  # Reward 2
    'TR': (255, 215, 0),  # Treasure
    'O': (169, 169, 169),  # Obstacle
    ' ': (255, 255, 255),  # Empty
    'path': (255, 255, 0),  # Path taken
    'player': (0, 0, 255),  # Player
    'background': (173, 216, 230),  # Light blue background
    'border': (0, 0, 0)  # Border color
}

# Define the board and other initial parameters
board = [[' ' for _ in range(cols)] for _ in range(rows)]
deactivated = [[' ' for _ in range(cols)] for _ in range(rows)]
start = (0, 0)
board[start[1]][start[0]] = 'E'
player_position = start

# Number of each elements in the board
elements = {
    'T1': 1, 'T2': 2, 'T3': 2, 'T4': 1,
    'R1': 2, 'R2': 2, 'TR': 4, 'O': 9
}

# Function to scatter elements randomly except obstacles and start point
def scatter_elements(board, elements):
    all_positions = [(col, row) for row in range(rows) for col in range(cols) if (col, row) != start]
    random.shuffle(all_positions)
    
    for element, count in elements.items():
        for _ in range(count):
            while all_positions:
                col, row = all_positions.pop()
                if board[row][col] == ' ':
                    board[row][col] = element
                    break

scatter_elements(board, elements)


# Heuristic function
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Define movements
movements_even = {'UP': (0, 1), 'UL': (-1, 0), 'UR': (1, 0), 'DN': (0, -1), 'DR': (1, -1), 'DL': (-1, -1)}
movements_odd = {'UP': (0, 1), 'UL': (-1, 1), 'UR': (1, 1), 'DN': (0, -1), 'DR': (1, 0), 'DL': (-1, 0)}
T3_movement = {'UP': (0, 2), 'UL': (-2, 1), 'UR': (2, 1), 'DN': (0, -2), 'DR': (2, -1), 'DL': (-2, -1)}


# A* search function
def a_star_search(start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    cost_so_far = {}
    state_at = {}
    trap_reward_list = []  # Purpose: traps and rewards can only be picked up once 

    initial_state = {
        'position': start,
        'energy': 100,
        'steps': 0,
        'treasures': [],
        'energy_per_step': 1,
        'steps_per_move': 1,
        'last_move': None,
    }
    
    came_from[start] = None
    cost_so_far[start] = 0
    state_at[start] = initial_state

    while open_list:
        _, current = heapq.heappop(open_list)
        
        current_state = state_at[current]
        
        if current == goal:
            break
        
        def get_movements(position):
            col, row = position
            last_move = current_state['last_move']
            # Checks if the current position is in Trap 3
            if board[row][col] == 'T3' and last_move:
                move = T3_movement[last_move]
                next_pos = (col + move[0], row + move[1])
                # Checks if next_pos is within the boundaries of a 2D board
                if 0 <= next_pos[1] < len(board) and 0 <= next_pos[0] < len(board[0]):
                    # Checks if the next_pos is not an obstacle cell 
                    if board[next_pos[1]][next_pos[0]] != 'O':
                        return {last_move: move}
            if col % 2 == 0: # Current position in even column
                return movements_even
            else: # Current position in odd column 
                return movements_odd
        
        movements = get_movements(current)
        
        for direction, move in movements.items():
            next_pos = (current[0] + move[0], current[1] + move[1])
            
            # Prevent algorithm from moving into obstacle and trap 4 cells
            if 0 <= next_pos[1] < len(board) and 0 <= next_pos[0] < len(board[0]) and board[next_pos[1]][next_pos[0]] not in ['O', 'T4']:
                next_state = copy.deepcopy(current_state)
                next_state['last_move'] = direction
                next_state['position'] = next_pos
                
                # The effects of trap 1, trap 2, reward 1 and reward 2 are implemented here
                cell_type = board[next_pos[1]][next_pos[0]]
                if cell_type == 'T1' and next_pos not in trap_reward_list:
                    trap_reward_list.append(next_pos)
                    next_state['steps_per_move'] *= 2
                elif cell_type == 'T2' and next_pos not in trap_reward_list:
                    trap_reward_list.append(next_pos)
                    next_state['energy_per_step'] *= 2
                elif cell_type == 'R1' and next_pos not in trap_reward_list:
                    trap_reward_list.append(next_pos)
                    next_state['steps_per_move'] /= 2
                elif cell_type == 'R2' and next_pos not in trap_reward_list:
                    trap_reward_list.append(next_pos)
                    next_state['energy_per_step'] /= 2
                elif cell_type == 'TR' and next_pos not in next_state['treasures']:
                    next_state['treasures'].append(next_pos)
                
                next_state['steps'] += next_state['steps_per_move']
                next_state['energy'] -= next_state['energy_per_step']
                
                new_cost = cost_so_far[current] + next_state['energy_per_step'] * next_state['steps_per_move']
                
                # Find next_pos with the least cost 
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal, next_pos)
                    heapq.heappush(open_list, (priority, next_pos))
                    came_from[next_pos] = current
                    state_at[next_pos] = next_state

    return came_from, cost_so_far, state_at



def find_optimal_path(start, treasures):
    min_path = None  
    min_cost = float('inf')  
    state_at_all = {}  
    error = False  

    # Iterate through all permutations of the treasures
    for perm in permutations(treasures):
        total_cost = 0  
        current_path = []  
        current_position = start  
        current_state_at = {}  

        try:
            # Process each treasure in the current permutation
            for treasure in perm:
                # Perform A* search to find the path and cost to the next treasure
                came_from, cost_so_far, state_at = a_star_search(current_position, treasure)
                total_cost += cost_so_far[treasure]  # Accumulate the cost of reaching the treasure
                
                path_segment = []  # List to store the path segment to the current treasure
                current = treasure
                
                # Trace back path from the treasure to starting point
                while current is not None:
                    path_segment.append(current)
                    current_state_at[current] = state_at[current]  # Save the state of each cell in the path
                    current = came_from.get(current)  # Move to the previous cell in the path
                path_segment.reverse()  # Reverse the path segment to get the correct order
                
                current_path.extend(path_segment[:-1])  # Add the path segment to the current path
                current_position = treasure  # Update the current position to the treasure

            # Ensure the final position is included in the path
            if current_position not in current_path:
                current_path.append(current_position)

            # Update the minimum cost and path if the current one is better
            if total_cost < min_cost:
                min_cost = total_cost
                min_path = current_path
                state_at_all = current_state_at

        except KeyError:
            # Handle cases where a treasure is unreachable
            if not error:
                print("The optimal path is obstructed by obstacles that completely block access to treasure.")
                error = True
            continue  # Skip this permutation and continue with the next

    return min_path, state_at_all  # Return the optimal path and state information



# Draw the hexagon grid
def draw_hex_grid(path):
    for row in range(rows):
        for col in range(cols):
            x = col * 3 / 2 * hex_radius + (screen_width - cols * hex_width +40) // 2
            y = np.sqrt(3) * (row + 0.5 * (col % 2)) * hex_radius + border_size + 25
            cell_type = board[row][col]
            if cell_type == ' ' and (col, row) in treasures:
                cell_type = 'TR'
            color = colors.get(cell_type, (255, 255, 255))
            pygame.draw.polygon(screen, color, hex_points(x, y, hex_radius))
            pygame.draw.polygon(screen, (0, 0, 0), hex_points(x, y, hex_radius), 1)
            draw_text(screen, cell_type, x, y)  # Always show element name

    # Draw the path in yellow
    for position in path:
        x = position[0] * 3 / 2 * hex_radius + (screen_width - cols * hex_width +40) // 2
        y = np.sqrt(3) * (position[1] + 0.5 * (position[0] % 2)) * hex_radius + border_size + 25
        pygame.draw.polygon(screen, colors['path'], hex_points(x, y, hex_radius))
        pygame.draw.polygon(screen, (0, 0, 0), hex_points(x, y, hex_radius), 1)
        draw_text(screen, 'TR' if board[position[1]][position[0]] == 'TR' else board[position[1]][position[0]], x, y)  # Show element name on the path

# Draw text on the hexagon grid
def draw_text(surface, text, x, y):
    font = pygame.font.SysFont(None, 20)  # Make the text smaller
    text_surface = font.render(text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=(x, y))
    surface.blit(text_surface, text_rect)

# Calculate the points of a hexagon
def hex_points(x, y, radius):
    points = []
    for i in range(6):
        angle = np.pi / 3 * i
        points.append((x + radius * np.cos(angle), y + radius * np.sin(angle)))
    return points

# Find all treasure locations
treasures = [(col, row) for row in range(len(board)) for col in range(len(board[row])) if board[row][col] == 'TR']

# Find the optimal path
path, state_at_all = find_optimal_path(start, treasures)

# Main game loop
running = True
clock = pygame.time.Clock()
path_index = 0

# Variables to store total steps and energy
total_steps = 0
steps_per_movement = 1
total_energy = 0
energy_per_movement = 1
trap_reward = []

# Tracing the optimal path for output display
try:
    for cell in path:
        col, row = cell
        cell_type = board[row][col]
    
        if cell_type == 'R1' and cell not in trap_reward:
            trap_reward.append(cell)
            energy_per_movement *= 0.5
        elif cell_type == 'R2' and cell not in trap_reward:
            trap_reward.append(cell)
            steps_per_movement *= 0.5
        elif cell_type == 'T1' and cell not in trap_reward:
            trap_reward.append(cell)
            steps_per_movement *= 2
        elif cell_type == 'T2' and cell not in trap_reward:
            trap_reward.append(cell)
            energy_per_movement *= 2
    
        total_steps += steps_per_movement
        total_energy += energy_per_movement
except TypeError: 
    running = False

# Total number of treasures to collect
total_treasures = len(treasures)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(colors['background'])  # Fill the screen with the background color

    # Draw border
    pygame.draw.rect(screen, colors['border'], (border_size//2, border_size//2, screen_width-border_size, screen_height-border_size-150), 2)

    # Draw hex grid and path
    draw_hex_grid(path[:path_index])  # Draw the path up to the current position

    if path_index < len(path):
        player_position = path[path_index]
        path_index += 1

    # Draw player
    x = player_position[0] * 3 / 2 * hex_radius + (screen_width - cols * hex_width +40) // 2
    y = np.sqrt(3) * (player_position[1] + 0.5 * (player_position[0] % 2)) * hex_radius + border_size +25
    pygame.draw.circle(screen, colors['player'], (int(x), int(y)), int(hex_radius / 2))

    # Check if all treasures have been collected
    if len(state_at_all[player_position]['treasures']) == total_treasures:
        running = False

    # Display optimal path, total steps, and energy consumption at the end
    if path_index == len(path):
        font = pygame.font.SysFont(None, 24)  # Make the text smaller

        # Wrap the optimal path text if it's too long
        path_text_lines = []
        path_text = f'Optimal Path: {path}'
        while len(path_text) > 60:  # Arbitrary limit for line length
            split_idx = path_text.rfind(' ', 0, 60)
            path_text_lines.append(path_text[:split_idx])
            path_text = path_text[split_idx + 1:]
        path_text_lines.append(path_text)

        y_offset = screen_height - 150
        for line in path_text_lines:
            line_surface = font.render(line, True, (0, 0, 0))
            screen.blit(line_surface, (border_size, y_offset))
            y_offset += 20
        
        steps_text = font.render(f'Total Steps Taken: {total_steps}', True, (0, 0, 0))
        energy_text = font.render(f'Total Energy Consumption: {total_energy}', True, (0, 0, 0))
        screen.blit(steps_text, (border_size, y_offset))
        screen.blit(energy_text,(border_size, y_offset + 30))

    pygame.display.flip()
    clock.tick(2)  # Control the speed of animation (adjust as needed)

pygame.quit()