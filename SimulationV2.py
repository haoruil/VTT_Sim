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

class NodeConnections:
    """Class for managing node connections."""
    
    def __init__(self, square_positions, triangle_positions_list, connection_threshold=0.01):
        # Set the threshold distance for considering nodes as connected
        self.connection_threshold = connection_threshold
        # Initialize node groups to store connected nodes
        self.node_groups = []
        # Initialize connections between nodes
        self.initialize_connections(square_positions, triangle_positions_list)
    
    def initialize_connections(self, square_positions, triangle_positions_list):
        """Initialize connections between nodes."""
        # Create a list to store all nodes with position and index information
        all_nodes = []
        
        # Add square nodes
        for i, pos in enumerate(square_positions):
            all_nodes.append({
                'position': pos,
                'source': 'square',  # Indicates the source of the node
                'group_index': i,
                'local_index': i
            })
        
        # Add triangle nodes
        for tri_idx, triangle_positions in enumerate(triangle_positions_list):
            for node_idx, pos in enumerate(triangle_positions):
                all_nodes.append({
                    'position': pos,
                    'source': f'triangle_{tri_idx}',  # Indicates the triangle source
                    'group_index': tri_idx,
                    'local_index': node_idx
                })
        
        # Find and establish connections between nodes
        n = len(all_nodes)
        connected_nodes = set()
        
        for i in range(n):
            if i in connected_nodes:
                continue
                
            # Create a new group for connected nodes
            current_group = [all_nodes[i]]
            connected_nodes.add(i)
            
            for j in range(i + 1, n):
                if j in connected_nodes:
                    continue
                    
                # Check if nodes are within the connection threshold
                pos1 = all_nodes[i]['position']
                pos2 = all_nodes[j]['position']
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance < self.connection_threshold:
                    current_group.append(all_nodes[j])
                    connected_nodes.add(j)
            
            # Add the group if it contains any connected nodes
            if len(current_group) > 0:
                self.node_groups.append(current_group)
    
    def update_positions(self, square_positions, triangle_positions_list, square_z, triangle_z):
        """Update the positions of all connected nodes."""
        for group in self.node_groups:
            if len(group) > 1:  # Process only groups with multiple nodes
                # Calculate the average position of all nodes in the group
                avg_x, avg_y, avg_z = 0, 0, 0
                for node in group:
                    if node['source'] == 'square':
                        avg_x += square_positions[node['local_index']][0]
                        avg_y += square_positions[node['local_index']][1]
                        avg_z += square_z[node['local_index']]
                    else:
                        tri_idx = int(node['source'].split('_')[1])
                        avg_x += triangle_positions_list[tri_idx][node['local_index']][0]
                        avg_y += triangle_positions_list[tri_idx][node['local_index']][1]
                        avg_z += triangle_z[tri_idx][node['local_index']]
                
                avg_x /= len(group)
                avg_y /= len(group)
                avg_z /= len(group)
                
                # Move all connected nodes to the average position
                for node in group:
                    if node['source'] == 'square':
                        square_positions[node['local_index']] = (avg_x, avg_y)
                        square_z[node['local_index']] = avg_z
                    else:
                        tri_idx = int(node['source'].split('_')[1])
                        triangle_positions_list[tri_idx][node['local_index']] = (avg_x, avg_y)
                        triangle_z[tri_idx][node['local_index']] = avg_z
        
        return square_positions, triangle_positions_list, square_z, triangle_z


class RigidBodyConstraints:
    """Class for handling rigid body constraints."""
    
    def __init__(self, initial_square_positions, initial_triangle_positions_list):
        # Store the initial configuration lengths
        self.initial_square_lengths = self._calculate_member_lengths(initial_square_positions)
        self.initial_triangle_lengths = [
            self._calculate_member_lengths(tri_pos) 
            for tri_pos in initial_triangle_positions_list
        ]
        # Set the minimum angle constraint in radians
        self.min_angle = np.radians(15)
        
    def _calculate_member_lengths(self, positions):
        """Calculate lengths of members between all pairs of nodes."""
        lengths = []
        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                dz = positions[j][2] if len(positions[i]) > 2 else 0
                lengths.append(np.sqrt(dx*dx + dy*dy + dz*dz))
        return lengths
    
    def _calculate_angle(self, p1, p2, p3):
        """Calculate the angle formed by three points."""
        v1 = np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2] if len(p2) > 2 else 0])
        v2 = np.array([p3[0]-p2[0], p3[1]-p2[1], p3[2] if len(p3) > 2 else 0])
        
        # Normalize vectors
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0
            
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        # Handle numerical errors
        cos_angle = min(1.0, max(-1.0, cos_angle))
        return np.arccos(cos_angle)
    
    def enforce_constraints(self, square_positions, triangle_positions_list, square_z, triangle_z):
        """Enforce rigid body and angle constraints."""
        # Enforce constraints for squares
        square_positions, square_z = self._enforce_rigid_body(
            square_positions, 
            square_z, 
            self.initial_square_lengths
        )
        
        # Enforce constraints for triangles
        for i, (tri_pos, initial_lengths) in enumerate(zip(triangle_positions_list, self.initial_triangle_lengths)):
            positions_3d = [(p[0], p[1], triangle_z[i][j]) for j, p in enumerate(tri_pos)]
            new_positions, new_z = self._enforce_rigid_body(
                tri_pos,
                triangle_z[i],
                initial_lengths
            )
            triangle_positions_list[i] = new_positions
            triangle_z[i] = new_z
            
        # Enforce angle constraints
        self._enforce_angle_constraints(square_positions, square_z)
        for i, tri_pos in enumerate(triangle_positions_list):
            self._enforce_angle_constraints(tri_pos, triangle_z[i])
        
        return square_positions, triangle_positions_list, square_z, triangle_z
    
    def _enforce_rigid_body(self, positions, z_coords, target_lengths):
        """Enforce rigid body constraints to maintain member lengths."""
        positions_3d = [(p[0], p[1], z) for p, z in zip(positions, z_coords)]
        n = len(positions)
        max_iterations = 10
        tolerance = 1e-4
        
        for _ in range(max_iterations):
            max_error = 0
            length_idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    p1 = np.array(positions_3d[i])
                    p2 = np.array(positions_3d[j])
                    current_length = np.linalg.norm(p2 - p1)
                    target_length = target_lengths[length_idx]
                    length_idx += 1
                    
                    if abs(current_length - target_length) > tolerance:
                        diff = p2 - p1
                        factor = (target_length - current_length) / (2 * current_length)
                        adjustment = diff * factor
                        
                        positions_3d[i] = tuple(p1 - adjustment)
                        positions_3d[j] = tuple(p2 + adjustment)
                        
                        max_error = max(max_error, abs(current_length - target_length))
            
            if max_error < tolerance:
                break
        
        new_positions = [(p[0], p[1]) for p in positions_3d]
        new_z = [p[2] for p in positions_3d]
        return new_positions, new_z
    
    def _enforce_angle_constraints(self, positions, z_coords):
        """Enforce minimum angle constraints."""
        positions_3d = [(p[0], p[1], z) for p, z in zip(positions, z_coords)]
        n = len(positions_3d)
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i != j and j != k and i != k:
                        angle = self._calculate_angle(
                            positions_3d[i],
                            positions_3d[j],
                            positions_3d[k]
                        )
                        
                        if angle < self.min_angle:
                            center = np.array(positions_3d[j])
                            p1 = np.array(positions_3d[i])
                            p2 = np.array(positions_3d[k])
                            
                            v1 = p1 - center
                            v2 = p2 - center
                            current_angle = angle
                            needed_rotation = (self.min_angle - current_angle) / 2
                            
                            rotation_matrix = np.array([
                                [np.cos(needed_rotation), -np.sin(needed_rotation), 0],
                                [np.sin(needed_rotation), np.cos(needed_rotation), 0],
                                [0, 0, 1]
                            ])
                            
                            new_v1 = np.dot(rotation_matrix, v1)
                            new_v2 = np.dot(rotation_matrix.T, v2)
                            
                            positions_3d[i] = tuple(center + new_v1)
                            positions_3d[k] = tuple(center + new_v2)
        
        for i in range(n):
            positions[i] = (positions_3d[i][0], positions_3d[i][1])
            z_coords[i] = positions_3d[i][2]



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
    
        
        self.pit_phase_complete = False
        self.post_pit_timer = 0
        self.specific_movement_started = False
        self.pit_timer = 0
        self.fixed_positions = None
        self.movement_directions = {}
        self.start_positions = {}

        # Define target positions for specific nodes
        self.specific_node_targets = {
            6: (-0.75, -1.5, -1.8),  # Node 6 to pit bottom
            4: (-1, -1, 2),  # Node 4 to specified position
            3: (0, -1.5, 1)  # Node 3 to specified position
        }

        
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
    
    def _calculate_pit_surface_height(self, x, y, pit_center, pit_radius, pit_depth):
        """计算pit表面在给定(x,y)位置的高度
        
        Args:
            self: 类实例
            x, y: 位置坐标
            pit_center: 凹坑中心坐标元组 (x, y, z)
            pit_radius: 凹坑半径
            pit_depth: 凹坑深度
            
        Returns:
            float: 表面高度
        """
        dx = x - pit_center[0]
        dy = y - pit_center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance < pit_radius:
            normalized_dist = (distance / pit_radius)
            return -pit_depth * (1.0 - normalized_dist*normalized_dist) + pit_center[2]
        return pit_center[2]

    def _enforce_pit_collision(self, pos_x, pos_y, pos_z, pit_center, pit_radius, pit_depth):
        """
        Enforce collision constraints with the pit surface.

        Args:
            pos_x, pos_y, pos_z: Coordinates of the point.
            pit_center: Tuple (x, y, z) representing the center of the pit.
            pit_radius: Radius of the pit.
            pit_depth: Depth of the pit.

        Returns:
            float: Adjusted z-coordinate after enforcing collision constraints.
        """
        # Calculate the surface height of the pit at the given (x, y) position
        surface_height = self._calculate_pit_surface_height(pos_x, pos_y, pit_center, pit_radius, pit_depth)
        # Ensure the z-coordinate is not below the surface height
        return max(surface_height, pos_z)

    def move_to_3d_targets(self, dt):
        """
        Guide the structure from a planar configuration to a 3D structure 
        using physical environment constraints. After 10 seconds, movement
        of specific nodes is performed without considering rigid body collisions.
        """
        # Initialize attributes if not already initialized
        if not hasattr(self, 'constraints'):
            self.constraints = RigidBodyConstraints(
                [(p[0], p[1], 0) for p in self.square_positions],
                [[(p[0], p[1], 0) for p in tri_pos] for tri_pos in self.triangle_positions_list]
            )
            self.node_connections = NodeConnections(
                self.square_positions,
                self.triangle_positions_list
            )
            self.pit_timer = 0
            self.specific_movement_started = False
            self.fixed_positions = None

        # Adjust effective time step based on whether specific movement has started
        effective_dt = dt * 0.25 if self.specific_movement_started else dt
        move_speed = 0.05 * self.time_acceleration  # Set movement speed

        # Define pit parameters
        pit_center = (-0.75, -1.5, -0.1)
        pit_radius = 1.75
        pit_depth = 1.5

        # Initialize z-coordinates if not already initialized
        if not hasattr(self, 'square_z'):
            self.square_z = [0.0] * len(self.square_positions)
            self.triangle_z = [[0.0] * 3 for _ in self.triangle_positions_list]

        # Increment the pit timer
        self.pit_timer += dt

        # Apply pit forces and constraints if under 10 seconds
        if self.pit_timer < 10.0:
            def calculate_environment_forces(x, y, z):
                """
                Calculate environmental forces acting on a point.

                Args:
                    x, y, z: Coordinates of the point.

                Returns:
                    dict: Forces acting on the point in dx, dy, dz directions.
                """
                forces = {'dx': 0, 'dy': 0, 'dz': 0}
                dx = x - pit_center[0]
                dy = y - pit_center[1]
                dz = z - pit_center[2]
                distance = np.sqrt(dx * dx + dy * dy)

                if distance < pit_radius:
                    # Calculate the normalized distance from the pit center
                    normalized_dist = (distance / pit_radius)
                    pit_influence = 1.0 - (normalized_dist * normalized_dist)

                    # Calculate the pit surface height at the given position
                    surface_height = self._calculate_pit_surface_height(x, y, pit_center, pit_radius, pit_depth)

                    # Apply vertical forces if above the pit surface
                    if z > surface_height:
                        vertical_factor = 1.0 - abs(z - pit_center[2]) / pit_depth
                        vertical_factor = max(0.0, min(1.0, vertical_factor))
                        forces['dz'] = -pit_depth * pit_influence * vertical_factor * 0.1

                    # Apply radial forces if not at the pit center
                    if distance > 0:
                        radial_force = pit_influence * 0.3
                        forces['dx'] -= (dx / distance) * radial_force
                        forces['dy'] -= (dy / distance) * radial_force

                return forces

            # Apply pit forces to square nodes
            for i, (x, y) in enumerate(self.square_positions):
                forces = calculate_environment_forces(x, y, self.square_z[i])
                new_x = x + forces['dx'] * dt * move_speed
                new_y = y + forces['dy'] * dt * move_speed
                new_z = self.square_z[i] + forces['dz'] * dt * move_speed
                new_z = self._enforce_pit_collision(new_x, new_y, new_z, pit_center, pit_radius, pit_depth)

                self.square_positions[i] = (new_x, new_y)
                self.square_z[i] = new_z

            # Apply pit forces to triangle nodes
            for tri_idx, triangle_positions in enumerate(self.triangle_positions_list):
                for node_idx, (x, y) in enumerate(triangle_positions):
                    forces = calculate_environment_forces(x, y, self.triangle_z[tri_idx][node_idx])
                    new_x = x + forces['dx'] * dt * move_speed
                    new_y = y + forces['dy'] * dt * move_speed
                    new_z = self.triangle_z[tri_idx][node_idx] + forces['dz'] * dt * move_speed
                    new_z = self._enforce_pit_collision(new_x, new_y, new_z, pit_center, pit_radius, pit_depth)

                    self.triangle_positions_list[tri_idx][node_idx] = (new_x, new_y)
                    self.triangle_z[tri_idx][node_idx] = new_z

            # Update positions and enforce constraints
            self.square_positions, self.triangle_positions_list, self.square_z, self.triangle_z = \
                self.node_connections.update_positions(
                    self.square_positions,
                    self.triangle_positions_list,
                    self.square_z,
                    self.triangle_z
                )
            self.square_positions, self.triangle_positions_list, self.square_z, self.triangle_z = \
                self.constraints.enforce_constraints(
                    self.square_positions,
                    self.triangle_positions_list,
                    self.square_z,
                    self.triangle_z
                )
        # If 10 seconds have passed, start specific node movement
        elif not self.specific_movement_started:
            self.specific_movement_started = True
            print("\nPit phase complete, starting specific node movement...")

            # Save current positions of all nodes
            self.fixed_positions = {
                'square': [(x, y, z) for (x, y), z in zip(self.square_positions, self.square_z)],
                'triangles': [[(x, y, z) for (x, y), z in zip(tri_pos, tri_z)]
                            for tri_pos, tri_z in zip(self.triangle_positions_list, self.triangle_z)]
            }
            
        # Movement phase after 10 seconds - Ignoring rigid body collisions
        if self.specific_movement_started:
            # Create node mapping and define target positions...
            node_mapping = {}
            current_node = 1

            # Map nodes...
            for i, pos in enumerate(self.square_positions):
                node_mapping[current_node] = ('square', i)
                current_node += 1

            for tri_idx, tri_pos in enumerate(self.triangle_positions_list):
                for node_idx, pos in enumerate(tri_pos):
                    node_mapping[current_node] = ('triangle', tri_idx, node_idx)
                    current_node += 1

            # Create reverse mapping
            reverse_mapping = {v: k for k, v in node_mapping.items()}

            # Define target positions
            node_targets = {
                6: (-0.75, -1.5, -1.5),
                4: (-1, -1, 2),
                3: (-0.4, -1.5, 1)
            }

            # Process connected node groups
            moved_nodes = set()
            for group in self.node_connections.node_groups:
                target_nodes = []
                for node in group:
                    if node['source'] == 'square':
                        node_num = reverse_mapping[('square', node['local_index'])]
                    else:
                        tri_idx = int(node['source'].split('_')[1])
                        node_num = reverse_mapping[('triangle', tri_idx, node['local_index'])]
                    if node_num in node_targets:
                        target_nodes.append((node_num, node))

                # If the group contains target nodes, calculate movement
                if target_nodes:
                    main_node = target_nodes[0]
                    target_pos = node_targets[main_node[0]]
                    group_key = tuple(sorted([(n['source'], n['local_index']) for n in group]))

                    # Initialize movement direction for new node group
                    if group_key not in self.movement_directions:
                        # Store initial positions and calculate unit direction vectors
                        start_positions = {}
                        movement_directions = {}

                        for node in group:
                            if node['source'] == 'square':
                                start_pos = (
                                    self.square_positions[node['local_index']][0],
                                    self.square_positions[node['local_index']][1],
                                    self.square_z[node['local_index']]
                                )
                            else:
                                tri_idx = int(node['source'].split('_')[1])
                                start_pos = (
                                    self.triangle_positions_list[tri_idx][node['local_index']][0],
                                    self.triangle_positions_list[tri_idx][node['local_index']][1],
                                    self.triangle_z[tri_idx][node['local_index']]
                                )

                            node_key = (node['source'], node['local_index'])
                            start_positions[node_key] = start_pos

                            # Calculate direction vector
                            dx = target_pos[0] - start_pos[0]
                            dy = target_pos[1] - start_pos[1]
                            dz = target_pos[2] - start_pos[2]
                            total_distance = math.sqrt(dx*dx + dy*dy + dz*dz)

                            if total_distance > 0:
                                movement_directions[node_key] = (
                                    dx/total_distance,
                                    dy/total_distance,
                                    dz/total_distance,
                                    total_distance  # Store total distance
                                )

                        self.movement_directions[group_key] = movement_directions
                        self.start_positions[group_key] = start_positions

                    # Move nodes using stored directions
                    speed = 0.01 * self.time_acceleration  # Reduce movement speed

                    for node in group:
                        node_key = (node['source'], node['local_index'])
                        direction = self.movement_directions[group_key][node_key]
                        start_pos = self.start_positions[group_key][node_key]

                        if node['source'] == 'square':
                            current_pos = (
                                self.square_positions[node['local_index']][0],
                                self.square_positions[node['local_index']][1],
                                self.square_z[node['local_index']]
                            )
                        else:
                            tri_idx = int(node['source'].split('_')[1])
                            current_pos = (
                                self.triangle_positions_list[tri_idx][node['local_index']][0],
                                self.triangle_positions_list[tri_idx][node['local_index']][1],
                                self.triangle_z[tri_idx][node['local_index']]
                            )

                        # Calculate moved distance
                        dx_current = current_pos[0] - start_pos[0]
                        dy_current = current_pos[1] - start_pos[1]
                        dz_current = current_pos[2] - start_pos[2]
                        moved_distance = math.sqrt(dx_current*dx_current + dy_current*dy_current + dz_current*dz_current)

                        # Check if the target is reached
                        if moved_distance < direction[3]:  # Compare with total distance
                            # Move along the fixed direction
                            new_x = current_pos[0] + direction[0] * speed * dt
                            new_y = current_pos[1] + direction[1] * speed * dt
                            new_z = current_pos[2] + direction[2] * speed * dt

                            # Prevent overshooting
                            if math.sqrt((new_x - start_pos[0])**2 +
                                        (new_y - start_pos[1])**2 +
                                        (new_z - start_pos[2])**2) > direction[3]:
                                # Reach the endpoint
                                new_x = start_pos[0] + direction[0] * direction[3]
                                new_y = start_pos[1] + direction[1] * direction[3]
                                new_z = start_pos[2] + direction[2] * direction[3]

                            # Update position
                            if node['source'] == 'square':
                                self.square_positions[node['local_index']] = (new_x, new_y)
                                self.square_z[node['local_index']] = new_z
                            else:
                                tri_idx = int(node['source'].split('_')[1])
                                self.triangle_positions_list[tri_idx][node['local_index']] = (new_x, new_y)
                                self.triangle_z[tri_idx][node['local_index']] = new_z

                        # Mark moved nodes
                        if node['source'] == 'square':
                            moved_nodes.add(('square', node['local_index']))
                        else:
                            tri_idx = int(node['source'].split('_')[1])
                            moved_nodes.add(('triangle', tri_idx, node['local_index']))

            # Restore fixed positions for nodes that have not moved
            for i in range(len(self.square_positions)):
                if ('square', i) not in moved_nodes:
                    fixed_pos = self.fixed_positions['square'][i]
                    self.square_positions[i] = (fixed_pos[0], fixed_pos[1])
                    self.square_z[i] = fixed_pos[2]

            for tri_idx in range(len(self.triangle_positions_list)):
                for node_idx in range(3):
                    if ('triangle', tri_idx, node_idx) not in moved_nodes:
                        fixed_pos = self.fixed_positions['triangles'][tri_idx][node_idx]
                        self.triangle_positions_list[tri_idx][node_idx] = (fixed_pos[0], fixed_pos[1])
                        self.triangle_z[tri_idx][node_idx] = fixed_pos[2]

            # Apply only node connection constraints
            self.square_positions, self.triangle_positions_list, self.square_z, self.triangle_z = \
                self.node_connections.update_positions(
                    self.square_positions,
                    self.triangle_positions_list,
                    self.square_z,
                    self.triangle_z
                )

        # Always draw the pit
        # Draw the environment
        pit_x = np.linspace(pit_center[0] - pit_radius, pit_center[0] + pit_radius, 20)
        pit_y = np.linspace(pit_center[1] - pit_radius, pit_center[1] + pit_radius, 20)
        pit_X, pit_Y = np.meshgrid(pit_x, pit_y)
        pit_Z = np.zeros_like(pit_X)

        for i in range(pit_X.shape[0]):
            for j in range(pit_X.shape[1]):
                pit_Z[i, j] = self._calculate_pit_surface_height(
                    pit_X[i, j], pit_Y[i, j],
                    pit_center, pit_radius, pit_depth
                )

        # Plot the pit surface with distinct color and transparency
        self.ax.plot_surface(pit_X, pit_Y, pit_Z, alpha=0.3, color='lightblue', edgecolor='blue')

        # Check if all target nodes have reached their target positions
        all_targets_reached = True
        if self.specific_movement_started:
            for node_num, target_pos in node_targets.items():
                if node_num not in node_mapping:
                    continue
                node_type = node_mapping[node_num]
                if node_type[0] == 'square':
                    current_pos = (
                        self.square_positions[node_type[1]][0],
                        self.square_positions[node_type[1]][1],
                        self.square_z[node_type[1]]
                    )
                else:
                    current_pos = (
                        self.triangle_positions_list[node_type[1]][node_type[2]][0],
                        self.triangle_positions_list[node_type[1]][node_type[2]][1],
                        self.triangle_z[node_type[1]][node_type[2]]
                    )

                distance = math.sqrt(
                    (target_pos[0] - current_pos[0])**2 +
                    (target_pos[1] - current_pos[1])**2 +
                    (target_pos[2] - current_pos[2])**2
                )

                if distance > 0.01:
                    all_targets_reached = False
                    break

        return self.specific_movement_started and all_targets_reached
    
    def _get_connected_nodes(self):
        """
        Retrieve all connected node groups.

        This method iterates through the node groups defined in `self.node_connections.node_groups`
        and constructs a list of connected nodes, identifying each node by its type and index.

        Returns:
            list: A list of connected node groups, where each group is a list of tuples.
                Each tuple represents a node in the format ('source_type', index) for squares
                or ('triangle', triangle_index, local_index) for triangles.
        """
        connected_nodes = []
        for group in self.node_connections.node_groups:
            node_group = []
            for node in group:
                if node['source'] == 'square':
                    node_group.append(('square', node['local_index']))
                else:
                    tri_idx = int(node['source'].split('_')[1])
                    node_group.append(('triangle', tri_idx, node['local_index']))
            connected_nodes.append(node_group)
        return connected_nodes


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