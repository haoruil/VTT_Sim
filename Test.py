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
initial_length = 2
target_length = 2.5
max_speed = 0.02
edge_mass = 1.92

class SquareTrussSolver:
    def __init__(self, memberlengths):
        # Initialize with 5 members (4 square edges + 1 diagonal)
        self.memberids = [1, 2, 3, 4, 5]  # 5 is the diagonal
        self.nodeids = [1, 2, 3, 4]  # Square vertices
        # Define the connections: [bottom, right, top, left, diagonal]
        self.memberends = [[1, 2], [2, 3], [3, 4], [4, 1], [1, 3]]
        self.memberlengths = list(memberlengths)
        self.memberspeeds = [0.0] * 5
        self.phase_complete = False
        self.node_velocities = [(0.0, 0.0) for _ in range(4)]
        self.memberam = len(self.memberids)
        self.nodeam = len(self.nodeids)
        self.membertoidx = {i: i-1 for i in self.memberids}
        self.nodetoidx = {i: i-1 for i in self.nodeids}
        self.nodetomembers = self._init_node_members()
        self.phase = 0  # 0: diagonal contraction, 1: edge extension
        
    def _init_node_members(self):
        nodetomembers = [[] for _ in self.nodeids]
        for i, v in enumerate(self.memberids):
            nodetomembers[self.nodetoidx[self.memberends[i][0]]].append((v, 0))
            nodetomembers[self.nodetoidx[self.memberends[i][1]]].append((v, 1))
        return nodetomembers

    def solvenormals(self):
        """Calculate normal forces using matrix equations for square configuration"""
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

    def enforce_length_constraints(self, positions, max_iterations=10, tolerance=1e-6):
        current_positions = list(positions)
        edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]  # Square edges + diagonal
        
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

    def update_motion(self, positions, dt):
        """
        In this process, simultaneously:
        - The diagonal (index=4) should shrink to half of its initial length.
        - Edge1 (index=1) and Edge2 (index=2) should extend to 1.5 times their initial length.
        """

        # If the target state has already been achieved, exit early.
        if self.phase_complete:
            return self.memberlengths, positions, self.node_velocities

        # Prepare an array for target speeds to store contraction/extension speeds for each member.
        target_speeds = [0.0] * 5

        # Define the initial lengths for the diagonal and edges.
        # Assumes `self.memberlengths[i]` represents the current length, and `initial_length` is predefined.
        initial_length_diag = math.sqrt(2) * initial_length  # Assume a square, diagonal length is sqrt(2) * edge length
        initial_length_edge = initial_length               # Square edge length equals initial_length

        # Calculate target lengths for diagonal and edges.
        diag_target = 0.5 * initial_length_diag
        edge_target = 1.5 * initial_length_edge

        # 1. Shrink the diagonal
        current_diag = self.memberlengths[4]
        if current_diag > diag_target:
            target_speeds[4] = -max_speed  # Negative speed indicates contraction
            print(f"Contracting diagonal: current={current_diag:.3f}, target={diag_target:.3f}")

        # 2. Extend Edge1 and Edge2
        #    Assumes indices 1 and 2 correspond to Edge1 and Edge2.
        for i in [1, 2]:
            current_edge = self.memberlengths[i]
            if current_edge < edge_target:
                target_speeds[i] = max_speed  # Positive speed indicates extension
                print(f"Extending edge {i}: current={current_edge:.3f}, target={edge_target:.3f}")

        # Check if the diagonal and both edges are close enough to their target lengths.
        if (
            abs(self.memberlengths[4] - diag_target) < 1e-3
            and abs(self.memberlengths[1] - edge_target) < 1e-3
            and abs(self.memberlengths[2] - edge_target) < 1e-3
        ):
            print("Motion complete!")
            self.phase_complete = True
            return self.memberlengths, positions, self.node_velocities

        # ================= Update member lengths =================

        # 1) Update lengths for controlled members.
        for i in [1, 2, 4]:  # Only update Edge1, Edge2, and the diagonal.
            if abs(target_speeds[i]) > 1e-8:  # Update only if there is motion.
                new_length = self.memberlengths[i] + target_speeds[i] * dt
                if i == 4:
                    # Constrain the diagonal length to [diag_target, initial_length_diag].
                    self.memberlengths[i] = np.clip(new_length, diag_target, initial_length_diag)
                elif i in [1, 2]:
                    # Constrain edge lengths to [initial_length_edge, edge_target].
                    self.memberlengths[i] = np.clip(new_length, initial_length_edge, edge_target)
                print(f"Member {i} updated length: {self.memberlengths[i]:.3f}")

        # 2) Update node positions (geometric/mechanical constraints).
        #    This assumes an external function `enforce_length_constraints` updates node positions.
        new_positions = self.enforce_length_constraints(positions)

        # 3) Update lengths of uncontrolled members (e.g., indices 0 and 3).
        for i in [0, 3]:
            n1, n2 = self.memberends[i][0] - 1, self.memberends[i][1] - 1
            dx = new_positions[n2][0] - new_positions[n1][0]
            dy = new_positions[n2][1] - new_positions[n1][1]
            self.memberlengths[i] = math.sqrt(dx * dx + dy * dy)
            print(f"Uncontrolled member {i} updated length: {self.memberlengths[i]:.3f}")

        # ================= Return updated results =================
        return self.memberlengths, new_positions, self.node_velocities


class SquareVisualization:
    def __init__(self, num_cycles=1):
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize perfect square positions centered at origin
        side_length = initial_length
        half_side = side_length / 2
        self.positions = [
            (-half_side, -half_side),  # Bottom left (1)
            (half_side, -half_side),   # Bottom right (2)
            (half_side, half_side),    # Top right (3)
            (-half_side, half_side)    # Top left (4)
        ]
        
        # Initialize solver with correct initial lengths
        diagonal_length = math.sqrt(2) * side_length
        initial_lengths = [side_length] * 4  # Four equal sides
        initial_lengths.append(diagonal_length)  # Diagonal
        self.solver = SquareTrussSolver(initial_lengths)
        
        self.num_cycles = num_cycles
        self.current_cycle = 0
        self.trajectory = []
        
    def update_frame(self, frame):
        self.ax.clear()
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Z Position')
        self.ax.set_title('Square Truss Robot Simulation')
        
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([-5, 5])
        
        # Update positions
        dt = 0.05
        lengths, new_positions, velocities = self.solver.update_motion(self.positions, dt)
        self.positions = new_positions
        
        # 在这里打印四个节点的位置
        for i, (x, y) in enumerate(self.positions):
            print(f"Node {i} position: (x={x:.3f}, y={y:.3f})")
            
        # Draw current state
        positions_3d = np.array([[x, y, 0] for x, y in self.positions])
        
        # Draw square edges
        for i in range(4):
            j = (i + 1) % 4
            self.ax.plot([positions_3d[i,0], positions_3d[j,0]],
                        [positions_3d[i,1], positions_3d[j,1]],
                        [positions_3d[i,2], positions_3d[j,2]], 'b-', linewidth=2)
            
            # Draw pad for this edge
            n1, n2 = i, j
            p1 = positions_3d[n1]
            p2 = positions_3d[n2]
            ratio = memberpadoffset / lengths[i]
            pad_pos = p1 + (p2 - p1) * ratio
            self.ax.scatter([pad_pos[0]], [pad_pos[1]], [pad_pos[2]], 
                          c='g', s=50, marker='s', label='Pad' if i == 0 else "")
        
        # Draw diagonal and its pad
        self.ax.plot([positions_3d[0,0], positions_3d[2,0]],
                    [positions_3d[0,1], positions_3d[2,1]],
                    [positions_3d[0,2], positions_3d[2,2]], 'r-', linewidth=2)
        
        # Draw diagonal pad
        p1 = positions_3d[0]  # Bottom left
        p2 = positions_3d[2]  # Top right
        ratio = memberpadoffset / lengths[4]
        pad_pos = p1 + (p2 - p1) * ratio
        self.ax.scatter([pad_pos[0]], [pad_pos[1]], [pad_pos[2]], 
                      c='y', s=50, marker='s', label='Diagonal Pad')
        
        # Draw nodes
        self.ax.scatter(positions_3d[:,0], positions_3d[:,1], positions_3d[:,2],
                       c='r', s=50, label='Nodes')
        
        # Add status information
        phase_names = ['Diagonal Contraction', 'Edge Extension']
        current_phase = phase_names[self.solver.phase]
        diagonal_length = lengths[4]
        diagonal_ratio = diagonal_length / initial_length
        
        self.ax.text2D(0.02, 0.98, 
                      f'Phase: {current_phase}\n'
                      f'Diagonal Length: {diagonal_length:.3f}\n'
                      f'Diagonal Ratio: {diagonal_ratio:.3f}\n'
                      f'Bottom Edge: {lengths[0]:.3f}\n'
                      f'Top Edge: {lengths[2]:.3f}',
                      transform=self.ax.transAxes)
        
        self.ax.legend()
        
        return self.ax

def run_square_simulation(num_cycles=1):
    visualizer = SquareVisualization(num_cycles=num_cycles)
    max_frames = 1500  # Adjust this number based on your needs
    ani = animation.FuncAnimation(
        visualizer.fig, 
        visualizer.update_frame,
        frames=range(max_frames),
        interval=50,
        repeat=False,
        blit=False,
        cache_frame_data=False
    )
    plt.show()

if __name__ == "__main__":
    run_square_simulation()