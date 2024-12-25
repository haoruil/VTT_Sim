import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import math

# Physical parameters
memberpadoffset = 0.54
mass = 1.92
weight = mass * 9.81
friction_static = 0.55
friction_dynamic = 0.43
max_speed = 0.02
edge_mass = 1.92

# Import the solver classes
from Test import SquareTrussSolver
from TEST2 import AdvancedTriangleSolver

def rotate_points(points, angle):
    """Rotate points around origin by angle (in radians)"""
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return [np.dot(rot_matrix, np.array(p)) for p in points]

class CombinedVisualization:
    """
    A class for visualizing and simulating the movement of connected triangles and squares in 3D space.
    This visualization includes shape deformation, rotation, and transition to 3D positions.
    """
    def __init__(self, triangle_positions=[(3,0), (1,-5), (-5,1)], 
                 square_pos=(0,0), 
                 triangle_angles=[0, np.pi/6, -np.pi/4],
                 square_angle=0,
                 initial_lengths=[1.5, 1.0, 1],  # Initial lengths for triangles: first=1.5, second=2.0, third=2.5
                 target_lengths=[2.5, 2.0, 2],   # Corresponding target lengths
                 num_cycles=1, 
                 triangle_cycles=[1,1,1],  # Allow different cycle counts for each triangle
                 time_acceleration=20.0,
                 dt=0.1):
        """
        Initialize the visualization with specified parameters.
        
        Args:
            triangle_positions (list): Initial positions for three triangles
            square_pos (tuple): Initial position for the square
            triangle_angles (list): Initial rotation angles for triangles
            square_angle (float): Initial rotation angle for square
            initial_lengths (list): Starting lengths for each triangle
            target_lengths (list): Target lengths for triangle deformation
            num_cycles (int): Number of simulation cycles
            triangle_cycles (list): Number of cycles for each triangle
            time_acceleration (float): Time acceleration factor
            dt (float): Time step size
        """
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # State tracking flags
        self.extension_modes = [False] * 3  # Track extension mode for each triangle
        self.extension_completes = [False] * 3  # Track completion of extensions
        self.current_target_sequence = [0, 0, 0]  # Track target sequence for each triangle
        
        # 3D movement phase flags
        self.final_3d_phase = False  # Flag for entering final 3D movement phase
        self.final_3d_complete = False  # Flag for 3D movement completion
        
        # Define 3D target positions for square nodes (x, y, z)
        self.square_3d_targets = [
            (0, 0, 0),      # Node 1
            (1, -3, 0),     # Node 2
            (0, -2, -1.5),  # Node 3
            (-1, -1, -3)    # Node 4
        ]
        
        # Define 3D target positions for triangle nodes
        self.triangle_3d_targets = [
            # First triangle target points
            [(-3, 1, 0), (-1, -1, 3), (0, 0, 0)],
            # Second triangle target points
            [(-1, -1, 3), (-2, -2, 0), (1, -3, 0)],
            # Third triangle target points
            [(-2, -2, 0), (-3, 1, 0), (-1, -1, -3)]
        ]

        # Define extension targets for each triangle
        # Each target includes sequence of node movements
        target1 = {
            "sequence": [
                {"nodes": [2, 1, 0], "targets": [(0.091, 0.120), (-1, -2.5), (-2.5, -1.1)]},
                {"nodes": [1, 0], "targets": [(-0.75, -1.5), (-1.5, -0.75)]}
            ]
        }
        target2 = {
            "sequence": [
                {"nodes": [2, 0, 1], "targets": [(0.654, -1.863), (-0.75, -1.5), (0.1, -4.5)]},
                {"nodes": [1], "targets": [(-2.7, -2.7)]}  # Move node 1
            ]
        }
        target3 = {
            "sequence": [
                {"nodes": [2, 1, 0], "targets": [(-1.854, 0.642), (-1.5, -0.75), (-4.5, -0.5)]},
                {"nodes": [0], "targets": [(-2.7, -2.7)]}  # Move node 0
            ]
        }
        self.extension_targets = [target1, target2, target3]
        self.final_coordinates_printed = False
        
        # Store initial parameters
        self.triangle_positions = triangle_positions
        self.square_pos = square_pos
        self.triangle_angles = triangle_angles
        self.square_angle = square_angle
        self.time_acceleration = time_acceleration
        self.dt = dt
        self.triangle_initial_lengths = initial_lengths
        self.triangle_target_lengths = target_lengths
        
        # Initialize square geometry
        square_initial_length = 2.0
        half_side = square_initial_length / 2
        square_base_positions = [
            (-half_side, -half_side),
            (half_side, -half_side),
            (half_side, half_side),
            (-half_side, half_side)
        ]
        
        # Apply rotation to square
        rotated_square = rotate_points(square_base_positions, square_angle)
        self.square_positions = [
            (p[0] + square_pos[0], p[1] + square_pos[1]) 
            for p in rotated_square
        ]
        
        # Calculate square member lengths
        diagonal_length = math.sqrt(2) * square_initial_length
        square_lengths = [square_initial_length] * 4
        square_lengths.append(diagonal_length)
        self.square_solver = SquareTrussSolver(square_lengths)
        
        # Initialize triangles
        self.triangle_positions_list = []
        self.triangle_solvers = []
        
        # Create and position each triangle
        for idx, (pos, angle) in enumerate(zip(triangle_positions, triangle_angles)):
            triangle_initial_length = initial_lengths[idx]
            triangle_height = triangle_initial_length * math.sqrt(3) / 2
            triangle_base_positions = [
                (-triangle_initial_length/2, -triangle_height/3),
                (triangle_initial_length/2, -triangle_height/3),
                (0, 2*triangle_height/3)
            ]
            
            # Apply rotation and translation
            rotated_triangle = rotate_points(triangle_base_positions, angle)
            triangle_positions = [
                (p[0] + pos[0], p[1] + pos[1]) 
                for p in rotated_triangle
            ]
            self.triangle_positions_list.append(triangle_positions)
            
            # Create solver for each triangle
            triangle_solver = AdvancedTriangleSolver(
                [triangle_initial_length] * 3,
                target_length=self.triangle_target_lengths[idx]
            )
            self.triangle_solvers.append(triangle_solver)
        
        # Store reference lengths
        self.square_initial_length = square_initial_length
        self.triangle_initial_length = triangle_initial_length
        
        # Initialize cycle counters
        self.num_cycles = num_cycles
        self.triangle_cycles = triangle_cycles
        self.current_cycle = 0
        self.current_triangle_cycles = [0] * 3
        self.active_triangle = 0  # Index of currently active triangle
        self.trajectory = []

    def update_frame(self, frame):
        """
        Update and render a single frame of the visualization.
        
        Args:
            frame: Current frame number
            
        Returns:
            matplotlib.axes._subplots.Axes3DSubplot or None: Updated 3D axes object,
            or None if visualization is complete
        """
        # Clear and setup 3D axes
        self.ax.clear()
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Z Position')
        self.ax.set_title('Multi-Triangle and Square Truss Simulation')
        
        # Set fixed axis ranges
        self.ax.set_xlim([-5, 2])
        self.ax.set_ylim([-5, 2])
        self.ax.set_zlim([-4, 4])
        
        # Update square positions with time acceleration
        dt = self.dt
        for _ in range(int(self.time_acceleration)):
            square_lengths, new_square_positions, square_velocities = \
                self.square_solver.update_motion(self.square_positions, dt)
            self.square_positions = new_square_positions
            
            if self.square_solver.phase_complete:
                break
        
        # Update active triangle
        if self.active_triangle < 3:
            i = self.active_triangle
            if self.extension_modes[i]:
                # Handle extension mode
                new_positions, complete = self.extend_triangle_to_target(
                    self.triangle_positions_list[i], 
                    dt, 
                    self.extension_targets[i]
                )
                self.triangle_positions_list[i] = new_positions
                self.extension_completes[i] = complete
                
                # Update triangle member lengths
                for j in range(3):
                    n1 = self.triangle_solvers[i].memberends[j][0] - 1
                    n2 = self.triangle_solvers[i].memberends[j][1] - 1
                    dx = new_positions[n2][0] - new_positions[n1][0]
                    dy = new_positions[n2][1] - new_positions[n1][1]
                    triangle_lengths = list(self.triangle_solvers[i].memberlengths)
                    triangle_lengths[j] = math.sqrt(dx*dx + dy*dy)
                    self.triangle_solvers[i].memberlengths = triangle_lengths
            else:
                # Normal motion update
                for _ in range(int(self.time_acceleration)):
                    triangle_lengths, new_triangle_positions, triangle_velocities = \
                        self.triangle_solvers[i].update_motion(self.triangle_positions_list[i], dt)
                    self.triangle_positions_list[i] = new_triangle_positions
                    
                    if self.triangle_solvers[i].phase_complete:
                        break
        
        # Draw current state
        self._draw_square(square_lengths)
        for i, triangle_positions in enumerate(self.triangle_positions_list):
            self._draw_triangle(triangle_positions, self.triangle_solvers[i])
        
        # Update status information
        status_text = f'Square Phase: {"Diagonal Contraction" if not self.square_solver.phase else "Edge Extension"}\n'
        for i, solver in enumerate(self.triangle_solvers):
            phase_names = ['Extension P1', 'Extension P2', 'Stabilization', 'Contraction']
            status_text += f'Triangle {i+1}: {phase_names[solver.extension_phase]} (Cycle {self.current_triangle_cycles[i]+1}/{self.triangle_cycles[i]})\n'
            if i == self.active_triangle:
                status_text += f'Active Triangle: {i+1}\n'
        
        self.ax.text2D(0.02, 0.98, status_text, transform=self.ax.transAxes)
        
        # Check completion and cycle reset
        if self.active_triangle < 3:
            i = self.active_triangle
            if self.triangle_solvers[i].phase_complete:
                self.current_triangle_cycles[i] += 1
                if self.current_triangle_cycles[i] < self.triangle_cycles[i]:
                    # Reset solver with initial length
                    self.triangle_solvers[i] = AdvancedTriangleSolver(
                        [self.triangle_initial_lengths[i]] * 3,
                        target_length=self.triangle_target_lengths[i]
                    )
                elif not self.extension_modes[i]:
                    self.extension_modes[i] = True
                elif self.extension_completes[i]:
                    # Move to next triangle
                    self.active_triangle += 1
        
        # Handle square cycle reset
        if self.square_solver.phase_complete:
            self.current_cycle += 1
            if self.current_cycle < self.num_cycles:
                self.reset_square()
        
        # Check for transition to 3D phase
        if self.active_triangle >= 3 and not self.final_3d_phase:
            self.final_3d_phase = True
            print("\nStarting final 3D movement phase")
        
        # Handle 3D movement phase
        if self.final_3d_phase and not self.final_3d_complete:
            self.final_3d_complete = self.move_to_3d_targets(self.dt)
        
        # Update 3D positions
        positions_3d = []
        if hasattr(self, 'square_z'):
            positions_3d = [(x, y, z) for (x, y), z in zip(self.square_positions, self.square_z)]
        else:
            positions_3d = [(x, y, 0) for x, y in self.square_positions]
        
        # Print final coordinates if complete
        if self.final_3d_complete and not self.final_coordinates_printed:
            print("\nFinal 3D Coordinates:")
            print("\nSquare Nodes:")
            for i, (x, y) in enumerate(self.square_positions):
                z = self.square_z[i] if hasattr(self, 'square_z') else 0
                print(f"Node {i+1}: (x={x:.3f}, y={y:.3f}, z={z:.3f})")
            
            for i, triangle_positions in enumerate(self.triangle_positions_list):
                print(f"\nTriangle {i+1} Nodes:")
                for j, (x, y) in enumerate(triangle_positions):
                    z = self.triangle_z[i][j] if hasattr(self, 'triangle_z') else 0
                    print(f"Node {j+1}: (x={x:.3f}, y={y:.3f}, z={z:.3f})")
            
            self.final_coordinates_printed = True
            return None
            
        return self.ax

    def _draw_square(self, lengths):
        # Draw current state with z coordinates
        if hasattr(self, 'square_z'):
            positions_3d = np.array([[x, y, z] for (x, y), z in zip(self.square_positions, self.square_z)])
        else:
            positions_3d = np.array([[x, y, 0] for x, y in self.square_positions])
        
        # Draw edges
        for i in range(4):
            j = (i + 1) % 4
            self.ax.plot([positions_3d[i,0], positions_3d[j,0]],
                        [positions_3d[i,1], positions_3d[j,1]],
                        [positions_3d[i,2], positions_3d[j,2]], 'b-', linewidth=2)
        
        # Draw diagonal
        self.ax.plot([positions_3d[0,0], positions_3d[2,0]],
                     [positions_3d[0,1], positions_3d[2,1]],
                     [positions_3d[0,2], positions_3d[2,2]], 
                     'b-', linewidth=2)
        
        # Draw pads
        for i in range(5):
            if i < 4:
                n1, n2 = i, (i + 1) % 4
            else:
                n1, n2 = 0, 2
            
            p1 = positions_3d[n1]
            p2 = positions_3d[n2]
            ratio = memberpadoffset / lengths[i]
            pad_pos = p1 + (p2 - p1) * ratio
            self.ax.scatter([pad_pos[0]], [pad_pos[1]], [pad_pos[2]], 
                           c='g' if i < 4 else 'g', s=50, marker='s')
        
        self.ax.scatter(positions_3d[:,0], positions_3d[:,1], 
                       positions_3d[:,2], c='r', s=50)
    
    def _draw_triangle(self, positions, solver):
        # Draw current state with z coordinates
        if hasattr(self, 'triangle_z'):
            tri_idx = self.triangle_positions_list.index(positions)
            positions_3d = np.array([[x, y, z] for (x, y), z in zip(positions, self.triangle_z[tri_idx])])
        else:
            positions_3d = np.array([[x, y, 0] for x, y in positions])
        
        # Draw edges
        for i in range(3):
            j = (i + 1) % 3
            self.ax.plot([positions_3d[i,0], positions_3d[j,0]],
                        [positions_3d[i,1], positions_3d[j,1]],
                        [positions_3d[i,2], positions_3d[j,2]], 
                        'b-', linewidth=2)
        
        # Draw pads
        for i in range(3):
            n1 = solver.memberends[i][0] - 1
            n2 = solver.memberends[i][1] - 1
            p1 = positions_3d[n1]
            p2 = positions_3d[n2]
            ratio = memberpadoffset / solver.memberlengths[i]
            pad_pos = p1 + (p2 - p1) * ratio
            self.ax.scatter([pad_pos[0]], [pad_pos[1]], [pad_pos[2]], 
                           c='g', s=50, marker='s', alpha=1.0)
        
        self.ax.scatter(positions_3d[:,0], positions_3d[:,1], 
                       positions_3d[:,2], c='r', s=50)

    def extend_triangle_to_target(self, positions, dt, target_config):
        """
        Moves triangle nodes towards their target positions in a sequence.
        
        Args:
            positions (list): Current positions of triangle nodes
            dt (float): Time step
            target_config (dict): Configuration defining target positions and sequence
        
        Returns:
            tuple: (new_positions, all_complete) where new_positions are updated coordinates
                  and all_complete indicates if sequence is finished
        """
        new_positions = list(positions)
        all_complete = True
        
        # Get current sequence index for active triangle
        current_sequence_idx = self.current_target_sequence[self.active_triangle]
    
        # Get current target configuration from sequence
        current_target = target_config["sequence"][current_sequence_idx]

        # Apply time acceleration to speed up movement
        base_speed = 0.01  # Base movement speed
        accelerated_speed = base_speed * self.time_acceleration  # Speed after acceleration
        
        # Move each specified node towards its target
        for node_idx, target_pos in zip(current_target["nodes"], current_target["targets"]):
            current_pos = positions[node_idx]
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # If not at target (using small threshold)
            if distance > 0.001:
                all_complete = False
                move_speed = min(accelerated_speed, distance)  # Prevent overshooting
                dir_x = dx / distance * move_speed
                dir_y = dy / distance * move_speed
                
                # Update node position
                new_positions[node_idx] = (
                    current_pos[0] + dir_x * dt,
                    current_pos[1] + dir_y * dt
                )
        
        # Check for next sequence if current target complete
        if all_complete and current_sequence_idx < len(target_config["sequence"]) - 1:
            self.current_target_sequence[self.active_triangle] += 1
            all_complete = False  # Continue to next target
        
        return new_positions, all_complete
    
    def move_to_3d_targets(self, dt):
        """
        Handles the final movement phase where nodes transition to their 3D target positions.
        
        Args:
            dt (float): Time step for movement calculation
            
        Returns:
            bool: True if all nodes have reached their targets, False otherwise
        """
        all_complete = True
        move_speed = 0.05 * self.time_acceleration
        
        # Update square node positions in 3D
        for i, current_pos in enumerate(self.square_positions):
            target = self.square_3d_targets[i]
            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]
            dz = target[2] - self.square_z[i] if hasattr(self, 'square_z') else target[2]
            
            # Calculate 3D distance and movement scale
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            if distance > 0.001:
                all_complete = False
                scale = min(move_speed * dt, distance) / distance
                
                # Update XY coordinates
                self.square_positions[i] = (
                    current_pos[0] + dx * scale,
                    current_pos[1] + dy * scale
                )
                # Initialize and update Z coordinates
                if not hasattr(self, 'square_z'):
                    self.square_z = [0] * 4
                self.square_z[i] += dz * scale
        
        # Update triangle node positions in 3D
        for tri_idx, triangle_positions in enumerate(self.triangle_positions_list):
            for node_idx, current_pos in enumerate(triangle_positions):
                target = self.triangle_3d_targets[tri_idx][node_idx]
                dx = target[0] - current_pos[0]
                dy = target[1] - current_pos[1]
                
                # Initialize triangle Z coordinates if needed
                if not hasattr(self, 'triangle_z'):
                    self.triangle_z = [[0, 0, 0] for _ in range(3)]
                dz = target[2] - self.triangle_z[tri_idx][node_idx]
                
                # Calculate 3D distance and movement scale
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                if distance > 0.001:
                    all_complete = False
                    scale = min(move_speed * dt, distance) / distance
                    
                    # Update node position
                    self.triangle_positions_list[tri_idx][node_idx] = (
                        current_pos[0] + dx * scale,
                        current_pos[1] + dy * scale
                    )
                    self.triangle_z[tri_idx][node_idx] += dz * scale
        
        return all_complete

def run_combined_simulation(
    triangle_positions=[(3, 0), (1, -5), (-5, 1)],
    square_pos=(0, 0),
    triangle_angles=[0, np.pi/6, -np.pi/4],
    square_angle=0,
    num_cycles=1, 
    initial_lengths=[1.5, 1.0, 1],  # Initial lengths for triangles: 1.5, 2.0, 2.5
    target_lengths=[2.5, 2.0, 2],   # Corresponding target lengths
    triangle_cycles=[1,1,1],  # Specify cycle count for each triangle
    animation_speed=1.0,
    time_acceleration=5.0,
    dt=0.1
):
    """
    Runs the combined simulation of triangles and square movement with specified parameters.
    
    Args:
        triangle_positions (list): Initial positions for three triangles
        square_pos (tuple): Initial position for the square
        triangle_angles (list): Initial angles for triangles in radians
        square_angle (float): Initial angle for square in radians
        num_cycles (int): Number of animation cycles
        initial_lengths (list): Starting lengths for each triangle
        target_lengths (list): Final lengths for each triangle
        triangle_cycles (list): Number of cycles for each triangle
        animation_speed (float): Speed multiplier for animation
        time_acceleration (float): Time acceleration factor for physics
        dt (float): Time step size
        
    Returns:
        None: Displays the animation using matplotlib
    """
    # Initialize visualization with parameters
    visualizer = CombinedVisualization(
        triangle_positions=triangle_positions,
        square_pos=square_pos,
        triangle_angles=triangle_angles,
        square_angle=square_angle,    
        initial_lengths=initial_lengths,    
        target_lengths=target_lengths,      
        num_cycles=num_cycles,
        triangle_cycles=triangle_cycles,  
        time_acceleration=time_acceleration,
        dt=dt
    )
    
    # Calculate animation timing
    interval = 20 / animation_speed
    delay_frames = int(1000 / interval)  # Calculate frames needed for 10-second delay
    
    def delay_start(frame):
        """
        Handles animation startup delay and frame updates.
        
        Args:
            frame (int): Current frame number
            
        Returns:
            matplotlib.axes: Updated plot axes
        """
        if frame < delay_frames:
            # Display countdown during delay
            visualizer.ax.set_title('Simulation starting in {:.1f} seconds...'.format(
                (delay_frames - frame) * interval / 1000))
            return visualizer.ax
        return visualizer.update_frame(frame - delay_frames)  # Update after delay
    
    # Create and start animation
    ani = animation.FuncAnimation(
        visualizer.fig,
        delay_start,
        frames=range(15000 + delay_frames),  # Add delay frames to total
        interval=interval,
        repeat=False,
        blit=False,
        cache_frame_data=False
    )
    
    plt.show()

if __name__ == "__main__":
    # Example: Run the simulation with faster animation and physical update speeds
    run_combined_simulation(
        triangle_positions=[(-2.3, -2.5), (-0.2, -3.6), (-3.6, -0.5)],           
        square_pos=(0, 0),               
        triangle_angles=[np.pi/8, np.pi/6, np.pi/3],        
        square_angle=0,                  
        num_cycles=1,                    
        triangle_cycles=[7, 4, 3],        
        initial_lengths=[1.5, 1, 1],  
        target_lengths=[2.5, 2, 2],   
        animation_speed=4.0,             
        time_acceleration=20.0,          
        dt=0.1                           
    )