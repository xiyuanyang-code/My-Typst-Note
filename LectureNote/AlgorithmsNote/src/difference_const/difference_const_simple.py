
"""
Difference Constraint System Modeling Example -- Simplified Delivery Scheduling
-------------------------------------------------
Goal: Find a set of arrival times x_i such that the following constraints are satisfied:
  1. Time windows: a_i <= x_i <= b_i
  2. Partial travel constraints: x_j >= x_i + travel_time(i,j)

Method:
  - Convert each constraint to difference constraint form x_i - x_j <= c
  - Build a weighted directed graph
  - Use Bellman-Ford to check feasibility and find shortest path solution

Author: iFlow CLI
"""

import math


class DifferenceConstraintSystem:
    """Difference constraint system solver"""

    def __init__(self):
        """Initialize difference constraint system"""
        # Node definition (0 is virtual source node)
        self.nodes = [0, 1, 2, 3, 4]
        self.source = 0

        # Location mapping
        self.locations = {1: "A", 2: "B", 3: "C", 4: "D"}

        # Travel times (minutes) - only keep unidirectional constraints to avoid negative cycles
        self.travel = {
            (1, 2): 25, 
            (1, 3): 35, 
            (2, 4): 45,
            (3, 4): 30,
        }

        # Feasible time windows
        self.time_windows = {
            1: (30, 150),
            2: (20, 180),
            3: (80, 220),
            4: (10, 260),
        }

        # Edge list
        self.edges = []

    def build_graph(self):
        """Build difference constraint graph"""
        self.edges = []

        # Travel constraints: x_j >= x_i + t => x_i - x_j <= -t => edge j->i weight -t
        for (i, j), t in self.travel.items():
            self.edges.append((j, i, -t))
            print(f"Adding travel constraint edge: {j} -> {i}, weight {-t} (corresponds to x_{i} - x_{j} <= {-t})")

        # Time window constraints: add virtual source node
        for i, (a, b) in self.time_windows.items():
            # Upper bound: x_i <= b => x_i - x_s <= b => edge s->i weight b
            self.edges.append((self.source, i, b))
            # Lower bound: x_i >= a => x_s - x_i <= -a => edge i->s weight -a
            self.edges.append((i, self.source, -a))

    def bellman_ford(self, source):
        """
        Bellman-Ford algorithm to solve difference constraint system
        
        Args:
            source: Source node
            
        Returns:
            tuple: (distance dictionary, whether negative cycle exists)
        """
        dist = {v: math.inf for v in self.nodes}
        dist[source] = 0.0
        n = len(self.nodes)
        
        # Perform n-1 iterations
        for i in range(n - 1):
            updated = False
            for (u, v, w) in self.edges:
                if dist[u] != math.inf and dist[v] > dist[u] + w:
                    dist[v] = dist[u] + w
                    updated = True
            if not updated:
                print(f"Converged after {i+1} iterations")
                break

        # Detect negative cycles: perform one more iteration, if updates still happen, negative cycle exists
        for (u, v, w) in self.edges:
            if dist[u] != math.inf and dist[v] > dist[u] + w:
                print(f"Negative cycle detected: node {u} to node {v}, current distance {dist[v]} > {dist[u]} + {w}")
                return dist, True  # Negative cycle exists

        return dist, False

    def solve(self):
        """Solve difference constraint system"""
        self.build_graph()
        return self.bellman_ford(self.source)

    def print_solution(self, dist, neg_cycle):
        """Print solution"""
        if neg_cycle:
            print("❌ System has no solution (negative weight cycle exists)")
            return False
        else:
            print("✅ System is feasible. The following is a set of arrival times (minutes) that satisfy the constraints:")
            for i in sorted(self.locations):
                print(f"  {self.locations[i]}: x_{i} = {dist[i]:.1f}  (allowed interval {self.time_windows[i]})")

            # Check slack for each travel constraint
            print("\nTravel constraint check (x_j >= x_i + t):")
            for (i, j), t in self.travel.items():
                lhs = dist[j]
                rhs = dist[i] + t
                slack = lhs - rhs
                print(f"  {self.locations[i]} -> {self.locations[j]}: "
                      f"{lhs:.1f} ≥ {rhs:.1f}  (slack={slack:+.1f})")
            return True

    def visualize_solution(self, dist, neg_cycle):
        """Visualize solution (requires matplotlib and networkx)"""
        if neg_cycle:
            print("❌ Cannot visualize, system has no solution (negative weight cycle exists)")
            return

        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("⚠️  Missing visualization dependencies (matplotlib, networkx). Please run 'pip install matplotlib networkx' to install.")
            return

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 1. Timeline visualization
        self._plot_timeline(ax1, dist, plt)
        
        # 2. Constraint graph visualization
        self._plot_constraint_graph(ax2, dist, plt, nx)
        
        plt.tight_layout()
        plt.show()

    def _plot_timeline(self, ax, dist, plt):
        """Plot timeline"""
        # Extract locations and times
        locations = [self.locations[i] for i in sorted(self.locations)]
        times = [dist[i] for i in sorted(self.locations)]
        min_times = [self.time_windows[i][0] for i in sorted(self.locations)]
        max_times = [self.time_windows[i][1] for i in sorted(self.locations)]
        
        y_pos = range(len(locations))
        
        # Plot time windows
        ax.hlines(y_pos, min_times, max_times, colors='lightblue', linewidth=10, alpha=0.5, label='Allowed time window')
        
        # Plot actual time points
        ax.scatter(times, y_pos, color='red', s=100, zorder=5, label='Optimal arrival time')
        
        # Add value labels
        for i, (time, min_t, max_t) in enumerate(zip(times, min_times, max_times)):
            ax.text(time, i, f'{time:.1f}', ha='left', va='bottom', fontsize=9)
            ax.text(min_t, i, f'[{min_t}', ha='right', va='top', fontsize=8, color='gray')
            ax.text(max_t, i, f'{max_t}]', ha='left', va='top', fontsize=8, color='gray')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(locations)
        ax.set_xlabel('Time (minutes)')
        ax.set_title('Optimal arrival time at each location')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_constraint_graph(self, ax, dist, plt, nx):
        """Plot constraint graph"""
        G = nx.DiGraph()
        
        # Add nodes
        for i in self.nodes:
            if i == 0:
                G.add_node(i, label='S')
            else:
                G.add_node(i, label=self.locations[i])
        
        # Add edges (only show important edges to avoid overly complex graph)
        # Add time window constraint edges
        for i, (a, b) in self.time_windows.items():
            G.add_edge(self.source, i, weight=b, color='blue')  # Upper bound constraint
            G.add_edge(i, self.source, weight=-a, color='green')  # Lower bound constraint
        
        # Add partial travel constraint edges
        travel_edges = list(self.travel.items())  # Show all travel constraints
        for (i, j), t in travel_edges:
            G.add_edge(j, i, weight=-t, color='red')
        
        # Node position layout
        pos = {
            0: (0, 0),  # Virtual source node at center
            1: (-1, 1),  # A
            2: (1, 1),   # B
            3: (-1, -1), # C
            4: (1, -1)   # D
        }
        
        # Draw graph
        node_colors = ['lightgreen' if i == 0 else 'lightblue' for i in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000, ax=ax)
        nx.draw_networkx_labels(G, pos, {i: G.nodes[i]['label'] for i in G.nodes()}, ax=ax)
        
        # Draw edges by color
        edges_by_color = {}
        for u, v, data in G.edges(data=True):
            color = data.get('color', 'black')
            if color not in edges_by_color:
                edges_by_color[color] = []
            edges_by_color[color].append((u, v))
        
        colors = ['red', 'blue', 'green']
        labels = ['Travel constraint', 'Upper bound constraint', 'Lower bound constraint']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            if color in edges_by_color:
                edges = edges_by_color[color]
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color, 
                                       width=2, arrows=True, ax=ax, 
                                       arrowsize=20, arrowstyle='->', 
                                       connectionstyle=f'arc3,rad={0.1*i}')
        
        # Draw edge weights
        edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
        
        ax.set_title('Difference Constraint System Graph')
        ax.axis('off')


def main():
    """Main function"""
    # Create difference constraint system
    dcs = DifferenceConstraintSystem()
    
    # Solve system
    dist, neg_cycle = dcs.solve()
    
    # Print solution
    solution_exists = dcs.print_solution(dist, neg_cycle)
    
    # Visualize solution if it exists
    if solution_exists:
        dcs.visualize_solution(dist, neg_cycle)


if __name__ == "__main__":
    main()

