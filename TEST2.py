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
#initial_length = 1.5
#target_length = 2.5
max_speed = 0.02
edge_mass = 1.92

class AdvancedTriangleSolver:
    def __init__(self, memberlengths, target_length):
        # Initialize members
        self.memberids = [1, 2, 3]
        self.nodeids = [1, 2, 3]
        self.memberends = [[1, 2], [2, 3], [3, 1]]
        self.memberlengths = list(memberlengths)
        self.memberspeeds = [0.0, 0.0, 0.0]
        self.extension_phase = 0
        self.phase_complete = False
        self.node_velocities = [(0.0, 0.0) for _ in range(3)]
        self.memberam = len(self.memberids)
        self.nodeam = len(self.nodeids)
        self.membertoidx = {i: i-1 for i in self.memberids}
        self.nodetoidx = {i: i-1 for i in self.nodeids}
        self.nodetomembers = self._init_node_members()
        self.last_debug_frame = 0
        self.target_length = target_length  
        self.initial_length = memberlengths[0]  

    def _init_node_members(self):
        nodetomembers = [[] for _ in self.nodeids]
        for i, v in enumerate(self.memberids):
            nodetomembers[self.nodetoidx[self.memberends[i][0]]].append((v, 0))
            nodetomembers[self.nodetoidx[self.memberends[i][1]]].append((v, 1))
        return nodetomembers

    def update_motion(self, positions, dt):
        """Update both member lengths and node positions"""
        if self.phase_complete:
            return self.memberlengths, positions, self.node_velocities

        # Calculate target member speeds based on current phase
        target_speeds = [0.0, 0.0, 0.0]
        
        # Debug output
        if not hasattr(self, 'last_debug_frame'):
            self.last_debug_frame = 0
            
        if self.last_debug_frame % 100 == 0:
            print(f"\nCurrent phase: {self.extension_phase}")
            print(f"Current lengths: {[f'{l:.6f}' for l in self.memberlengths]}")
            print(f"Target length: {self.target_length:.6f}")
            print(f"Node velocities: {[(f'{vx:.6f}', f'{vy:.6f}') for vx, vy in self.node_velocities]}")
        self.last_debug_frame += 1
        
        if self.extension_phase == 0:
            # Two sides extending
            if self.memberlengths[1] < self.target_length:
                target_speeds[1] = max_speed
            if self.memberlengths[2] < self.target_length:
                target_speeds[2] = max_speed
                
            if (abs(self.memberlengths[1] - self.target_length) < 1e-4 and 
                abs(self.memberlengths[2] - self.target_length) < 1e-4):
                print("\nPhase 0 complete, entering phase 1")
                print(f"Lengths at phase transition: {[f'{l:.6f}' for l in self.memberlengths]}")
                self.extension_phase = 1
                
        elif self.extension_phase == 1:
            # Third side extending
            if self.memberlengths[0] < self.target_length:
                target_speeds[0] = max_speed

            # Debug print to understand what's happening
            print("\nDebug info:")
            print(f"Lengths check: {[abs(l - self.target_length) < 1e-4 for l in self.memberlengths]}")
            velocities_check = [math.sqrt(vx*vx + vy*vy) for vx, vy in self.node_velocities]
            print(f"Velocities magnitudes: {velocities_check}")
            print(f"Velocity check: {[v < 0.2 for v in velocities_check]}")
                        
            # Modify judgment logic
            all_extended = True  # Check if all edges have reached the target length
            for l in self.memberlengths:
                if abs(l - self.target_length) > 1e-4:  # Allow small numerical error tolerance
                    all_extended = False
                    break

            # Relax velocity condition
            max_allowed_velocity = 0.5  # Increase the allowed maximum velocity
            all_stable = True  # Check if all nodes are stable enough
            for vx, vy in self.node_velocities:
                if math.sqrt(vx*vx + vy*vy) > max_allowed_velocity:  # Check node velocity magnitude
                    all_stable = False
                    break

            # If edge lengths reach the target, force them to the target to avoid numerical errors
            if all_extended:
                self.memberlengths = [self.target_length] * 3

            # State transition judgment
            if all_extended:
                if not hasattr(self, 'extension_complete_time'):  # Initialize extension complete time
                    self.extension_complete_time = 0
                self.extension_complete_time += 1  # Increment the counter
                
                # Transition to the next phase once edges are at target length for a while, ignoring velocity
                if self.extension_complete_time > 30:  # Wait for 30 frames
                    print("\nPhase 1 complete, entering stabilization phase")
                    print(f"Current lengths: {[f'{l:.6f}' for l in self.memberlengths]}")
                    print(f"Node velocities: {[(f'{vx:.6f}', f'{vy:.6f}') for vx, vy in self.node_velocities]}")
                    self.extension_phase = 2  # Move to stabilization phase
                    self.stabilization_counter = 50  # Set stabilization counter
                    self.memberlengths = [self.target_length] * 3  # Ensure all lengths are exactly the target
            else:
                if hasattr(self, 'extension_complete_time'):  # Reset the extension complete time if condition is not met
                    self.extension_complete_time = 0
                
        elif self.extension_phase == 2:
            if self.stabilization_counter > 0:
                self.stabilization_counter -= 1
                return self.memberlengths, positions, self.node_velocities
            else:
                print("\nStabilization complete, starting contraction")
                self.extension_phase = 3
                    
        elif self.extension_phase == 3:
            # All sides contracting simultaneously
            any_needs_contraction = False
            
            for i in range(3):
                if self.memberlengths[i] > self.initial_length + 1e-4:
                    any_needs_contraction = True
                    target_speeds[i] = -max_speed
            
            if not hasattr(self, 'contraction_started') and any_needs_contraction:
                self.contraction_started = True
                print("\nStarting synchronized contraction")
                print(f"Initial contraction lengths: {[f'{l:.6f}' for l in self.memberlengths]}")
            
            if not any_needs_contraction:
                print("\nContraction complete")
                print(f"Final lengths: {[f'{l:.6f}' for l in self.memberlengths]}")
                self.phase_complete = True
                return self.memberlengths, positions, self.node_velocities

        # Calculate normal forces
        normal_forces = self.solvenormals()

        # Calculate pad positions and their velocities
        pad_info = []
        for i in range(3):
            n1 = self.memberends[i][0] - 1
            n2 = self.memberends[i][1] - 1
            p1 = positions[n1]
            p2 = positions[n2]
            v1 = self.node_velocities[n1]
            v2 = self.node_velocities[n2]
            
            ratio = memberpadoffset / self.memberlengths[i]
            
            # Pad position
            pad_x = p1[0] + (p2[0] - p1[0]) * ratio
            pad_y = p1[1] + (p2[1] - p1[1]) * ratio
            
            # Pad velocity
            edge_dir_x = (p2[0] - p1[0]) / self.memberlengths[i]
            edge_dir_y = (p2[1] - p1[1]) / self.memberlengths[i]
            pad_vx = (v1[0] + (v2[0] - v1[0]) * ratio + 
                     target_speeds[i] * edge_dir_x * ratio)
            pad_vy = (v1[1] + (v2[1] - v1[1]) * ratio +
                     target_speeds[i] * edge_dir_y * ratio)
            
            pad_info.append({
                'position': (pad_x, pad_y),
                'velocity': (pad_vx, pad_vy),
                'normal_force': normal_forces[i],
                'nodes': (n1, n2),
                'ratio': ratio
            })

        # Calculate friction forces
        edge_forces = []
        for pad in pad_info:
            vx, vy = pad['velocity']
            speed = math.sqrt(vx*vx + vy*vy)
            
            if speed > 1e-6:
                friction_mag = friction_dynamic * pad['normal_force']
                fx = -friction_mag * vx / speed
                fy = -friction_mag * vy / speed
            else:
                fx, fy = 0.0, 0.0
            edge_forces.append((fx, fy))

        # Update node velocities based on edge forces
        new_velocities = [(0.0, 0.0) for _ in range(3)]

        for i, (fx, fy) in enumerate(edge_forces):
            pad = pad_info[i]
            n1, n2 = pad['nodes']
            ratio = pad['ratio']
            
            # Mass distribution: edge mass split unevenly at the pad
            m1 = edge_mass * 0.65 # Mass on the 0.54 end (larger mass, 4/5 of total edge_mass)
            m2 = edge_mass * 0.35  # Mass on the other end (smaller mass, 1/5 of total edge_mass)

            # Calculate acceleration for each node based on force and mass
            ax1 = fx * (1 - ratio) / m1
            ay1 = fy * (1 - ratio) / m1

            ax2 = fx * ratio / m2
            ay2 = fy * ratio / m2
            
            # Apply to connected nodes
            damping = 0.95
            # Update velocity for node1
            current_vx1, current_vy1 = self.node_velocities[n1]
            new_vx1 = (current_vx1 + ax1 * dt) * damping
            new_vy1 = (current_vy1 + ay1 * dt) * damping

            # Update velocity for node2
            current_vx2, current_vy2 = self.node_velocities[n2]
            new_vx2 = (current_vx2 + ax2 * dt) * damping
            new_vy2 = (current_vy2 + ay2 * dt) * damping

            # Assign updated velocities back to the corresponding nodes
            new_velocities[n1] = (new_vx1, new_vy1)
            new_velocities[n2] = (new_vx2, new_vy2)

        self.node_velocities = new_velocities

        # Update positions
        new_positions = []
        for i in range(3):
            x = positions[i][0] + self.node_velocities[i][0] * dt
            y = positions[i][1] + self.node_velocities[i][1] * dt
            new_positions.append((x, y))

        # Enforce length constraints
        new_positions = self.enforce_length_constraints(new_positions)

        # Update member lengths
        for i in range(3):
            if target_speeds[i] != 0:
                new_length = self.memberlengths[i] + target_speeds[i] * dt
                self.memberlengths[i] = np.clip(new_length, self.initial_length, self.target_length)

        # Re-enforce constraints with new lengths
        new_positions = self.enforce_length_constraints(new_positions)

        return self.memberlengths, new_positions, self.node_velocities

    def enforce_length_constraints(self, positions, max_iterations=10, tolerance=1e-6):
        current_positions = list(positions)
        edges = [(0,1), (1,2), (2,0)]
        
        for _ in range(max_iterations):
            max_error = 0
            for edge_idx, (i, j) in enumerate(edges):
                p1 = current_positions[i]
                p2 = current_positions[j]
                
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                current_length = math.sqrt(dx*dx + dy*dy)
                target_length = self.memberlengths[edge_idx]
                
                error = abs(current_length - target_length)
                max_error = max(max_error, error)
                
                if current_length > 1e-6:
                    correction = (target_length - current_length) / current_length / 2
                    delta_x = dx * correction
                    delta_y = dy * correction
                    
                    current_positions[i] = (p1[0] - delta_x, p1[1] - delta_y)
                    current_positions[j] = (p2[0] + delta_x, p2[1] + delta_y)
            
            if max_error < tolerance:
                break
                
        return current_positions

    def solvenormals(self):
        """Calculate normal forces using matrix equations"""
        solvematrix = np.zeros((self.memberam * 3 + self.nodeam, self.memberam * 3))
        ansmatrix = np.zeros(self.memberam * 3 + self.nodeam)
        
        count = 0
        for i, val in enumerate(self.memberids):
            # eq1: N + Rs + Re = W
            solvematrix[count, i] = 1
            solvematrix[count, self.memberam + i * 2] = 1
            solvematrix[count, self.memberam + i * 2 + 1] = 1
            ansmatrix[count] = weight
            count += 1
            
            # eq2: AN + LRe = BW
            solvematrix[count, i] = memberpadoffset
            solvematrix[count, self.memberam + i * 2 + 1] = self.memberlengths[i]
            ansmatrix[count] = self.memberlengths[i] / 2 * weight
            count += 1
            
            # eq3: (L-A)N + LRs = (L-B)W
            solvematrix[count, i] = self.memberlengths[i] - memberpadoffset
            solvematrix[count, self.memberam + i * 2] = self.memberlengths[i]
            ansmatrix[count] = (self.memberlengths[i] - self.memberlengths[i]/2) * weight
            count += 1
            
        for i, val in enumerate(self.nodeids):
            for v2 in self.nodetomembers[i]:
                solvematrix[count, self.memberam + self.membertoidx[v2[0]] * 2 + v2[1]] = 1
            count += 1
            
        res = np.linalg.lstsq(solvematrix, ansmatrix, rcond=None)
        return res[0][:self.memberam]

class ContinuousVisualizer:
    def __init__(self, num_cycles=10, start_pos=(0, 0), initial_angle=0):
        """
        Initializes the simulator for continuous visualization of a triangle in 3D space.

        :param num_cycles: Number of cycles to run the simulation.
        :param start_pos: Initial position of the triangle's center, given as (x, y).
        :param initial_angle: Initial rotation angle of the triangle in radians.
        """
        # Create a 3D plot figure and axes
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Initialize a solver for the triangle, using an array of initial side lengths
        self.solver = AdvancedTriangleSolver([self.initial_length] * 3)
        
        self.num_cycles = num_cycles         # Total number of simulation cycles
        self.current_cycle = 0              # Tracks the current cycle number
        
        # Calculate initial triangle vertex positions assuming it is centered at the origin
        height = self.initial_length * math.sqrt(3) / 2  # Height of an equilateral triangle
        base_positions = [
            (-self.initial_length / 2, -height / 3),  # Bottom left vertex
            (self.initial_length / 2, -height / 3),   # Bottom right vertex
            (0, 2 * height / 3)                  # Top vertex
        ]
        
        # Apply rotation to the triangle vertices if an initial angle is provided
        if initial_angle != 0:
            rot_matrix = np.array([
                [np.cos(initial_angle), -np.sin(initial_angle)],  # Rotation matrix for 2D rotation
                [np.sin(initial_angle), np.cos(initial_angle)]
            ])
            base_positions = [np.dot(rot_matrix, np.array(pos)) for pos in base_positions]
        
        # Translate the triangle to the specified start position
        self.positions = [
            (pos[0] + start_pos[0], pos[1] + start_pos[1])
            for pos in base_positions
        ]
        
        # Verify the center of the triangle is calculated correctly
        center = np.mean(self.positions, axis=0)
        print(f"Initial center position: ({center[0]:.6f}, {center[1]:.6f})")
        
        # Track the trajectory of the triangle throughout the simulation
        self.trajectory = []
        self.time_acceleration = 5.0  # Factor to accelerate the simulation time

    def init_3d_parameters(self):
        """
        Initializes the 3D plot parameters for the simulation, including axis labels, titles, and display ranges.
        """
        # Clear the 3D axes for fresh plotting
        self.ax.clear()
        self.ax.set_xlabel('X Position')     # Label for the X-axis
        self.ax.set_ylabel('Y Position')     # Label for the Y-axis
        self.ax.set_zlabel('Z Position')     # Label for the Z-axis
        self.ax.set_title('Continuous Physical Simulation in 3D')  # Set the plot title

        # Set limits for the 3D axes to define the display range
        self.ax.set_xlim([-10, 10])  # Expand the X-axis range
        self.ax.set_ylim([-10, 10])  # Expand the Y-axis range
        self.ax.set_zlim([-10, 10])  # Expand the Z-axis range
        self.ax.grid(True)           # Enable grid lines for better visualization

        
    def update_frame(self, frame):
        self.init_3d_parameters()
        
        # Execute multiple physics steps
        for _ in range(int(self.time_acceleration)):
            dt = 0.05
            lengths, new_positions, velocities = self.solver.update_motion(self.positions, dt)
            self.positions = new_positions
            
            # Record trajectory
            center = np.mean(self.positions, axis=0)
            self.trajectory.append(center)
            
            if self.solver.phase_complete:
                self.current_cycle += 1
                if self.current_cycle < self.num_cycles:
                    # Reset solver but keep positions
                    self.solver = AdvancedTriangleSolver([self.initial_length] * 3)
                else:
                    print(f"\nCompleted {self.num_cycles} cycles")
                    return True
        
        # Draw current state
        positions_3d = np.array([[x, y, 0] for x, y in self.positions])
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            trajectory_array = np.array(self.trajectory)
            self.ax.plot(trajectory_array[:,0], trajectory_array[:,1], 
                        np.zeros_like(trajectory_array[:,0]), 
                        'g-', alpha=0.5, label='Trajectory')
        
        # Draw edges
        for i in range(3):
            j = (i + 1) % 3
            self.ax.plot([positions_3d[i,0], positions_3d[j,0]],
                        [positions_3d[i,1], positions_3d[j,1]],
                        [positions_3d[i,2], positions_3d[j,2]], 'b-', linewidth=2)
        
        # Draw nodes
        self.ax.scatter(positions_3d[:,0], positions_3d[:,1], positions_3d[:,2],
                       c='r', s=50)
        
        # Draw pads
        for i in range(3):
            n1 = self.solver.memberends[i][0] - 1
            n2 = self.solver.memberends[i][1] - 1
            p1 = positions_3d[n1]
            p2 = positions_3d[n2]
            ratio = memberpadoffset / lengths[i]
            pad_pos = p1 + (p2 - p1) * ratio
            self.ax.scatter([pad_pos[0]], [pad_pos[1]], [pad_pos[2]], 
                           c='g', s=50, marker='s', alpha=1.0)
        # Add info
        phase_names = ['Extension Phase 1', 'Extension Phase 2', 
                      'Stabilization', 'Contraction']
        current_phase = phase_names[self.solver.extension_phase]
        
        # Calculate total distance traveled
        total_distance = 0
        if len(self.trajectory) > 1:
            traj_array = np.array(self.trajectory)
            distances = np.sqrt(np.sum(np.diff(traj_array, axis=0)**2, axis=1))
            total_distance = np.sum(distances)
        
        self.ax.text2D(0.05, 0.95, 
                      f'Cycle: {self.current_cycle + 1}/{self.num_cycles}\n'
                      f'Phase: {current_phase}\n'
                      f'Length: {lengths[0]:.3f}\n'
                      f'Total Distance: {total_distance:.3f}m',
                      transform=self.ax.transAxes)
        
        return False

def run_continuous_simulation(start_pos=(0, 0), initial_angle=0, num_cycles=10):
    """
    Runs a continuous simulation of a moving triangle.
    
    :param start_pos: Starting position of the triangle, given as (x, y).
    :param initial_angle: Initial angle of the triangle in radians.
    :param num_cycles: Number of cycles to run the simulation.
    """
    # Initialize the visualizer with the specified parameters
    visualizer = ContinuousVisualizer(
        num_cycles=num_cycles,       # Number of cycles for the simulation
        start_pos=start_pos,         # Starting position of the triangle
        initial_angle=initial_angle  # Initial rotation angle of the triangle
    )
    
    # Create an animated visualization with the visualizer's update function
    ani = animation.FuncAnimation(
        visualizer.fig,              # Figure object for the animation
        visualizer.update_frame,     # Frame update function
        frames=None,                 # Number of frames (None implies infinite loop)
        interval=10,                 # Time interval between frames in milliseconds
        repeat=False                 # Do not repeat the animation
    )
    
    # Display the animation
    plt.show()


if __name__ == "__main__":
    # Run the simulation with a specified starting position and initial angle
    run_continuous_simulation(
        start_pos=(0, 0),            # Starting position of the triangle at (0, 0)
        initial_angle=np.pi / 4      # Initial angle of 45 
    )