from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import math
import copy
from scipy.optimize import fsolve
import numpy as np

class Node:
    def __init__(self, id, node_type='cell', connections=None, flow_rate=0):
        self.id = id
        self.node_type = node_type
        self.connections = connections if connections else []
        self.flow_rate = flow_rate  # Default flow rate is 0

class Cell(Node):
    def __init__(self, id, species=None, flow_behavior=None, connections=None, flow_rate=0, petribution=None):
        super().__init__(id, node_type='cell', connections=connections, flow_rate=flow_rate)
        self.species = species
        self.flow_behavior = flow_behavior
        self.petribution = petribution if petribution else set()

class Pump(Node):
    def __init__(self, id, flow_behavior=None, connections=None, flow_rate=0):
        super().__init__(id, node_type='pump', connections=connections, flow_rate=flow_rate)
        self.flow_behavior = flow_behavior

class Valve(Node):
    def __init__(self, id, state=None, connections=None, flow_rate=0):
        super().__init__(id, node_type='valve', connections=connections, flow_rate=flow_rate)
        self.state = state

class Media(Node):
    def __init__(self, id, name=None, connections=None, flow_rate=0):
        super().__init__(id, node_type='media', connections=connections, flow_rate=flow_rate)
        self.name = name

class MUX(Node):
    def __init__(self, id, num_inputs, connections=None, flow_rate=0, petribution=None):
        super().__init__(id, node_type='mux', connections=connections, flow_rate=flow_rate)
        self.num_inputs = num_inputs
        self.name = "V"
        self.petribution = petribution if petribution else set()

class Output(Node):
    def __init__(self, id, connections=None, flow_rate=0):
        super().__init__(id, node_type='output', connections=connections, flow_rate=flow_rate)
        
class Controller(Node):
    def __init__(self, id, connections=None, flow_rate=0):
        super().__init__(id, node_type='controller', connections=connections, flow_rate=flow_rate)


class MillifluidicSystem:
    def __init__(self):
        self.nodes = {}
        self.connections = defaultdict(list)
        self.inverse_connections = defaultdict(list)  # To keep track of inverse connections for back-propagation
        self.warnings = []

    def add_node(self, node):
        self.nodes[node.id] = node

    def get_flow_rate(self, node_id):
        if node_id in self.nodes:
            return self.nodes[node_id].flow_rate
        else:
            return None

    def get_node_by_id(self, node_id):
        if node_id in self.nodes:
            return self.nodes[node_id]
        else:
            return None

    def add_connection(self, from_node, to_node, flow_rate=None):
        if flow_rate is None:
            flow_rate = 0  # Default flow rate is 0 if not specified
        self.connections[from_node].append((to_node, flow_rate))
        self.inverse_connections[to_node].append((from_node, flow_rate))

    def set_flow_rates(self):
        # Create a list of node IDs to iterate over
        node_ids_to_check = list(self.nodes.keys())

        # Set initial flow rates for cells
        for node_id in node_ids_to_check:
            node = self.nodes[node_id]
            if isinstance(node, Cell):
                cell_flow_rate = node.flow_rate
                print(f"Initial Flow Rate for {node_id} (Cell): {cell_flow_rate}")

        # Set flow rates for pumps and MUXes based on connected cells
        for node_id in node_ids_to_check:
            node = self.nodes[node_id]
            if isinstance(node, Pump):
                total_output_flow = sum(self.nodes[to_node_id].flow_rate for to_node_id, _ in self.connections[node_id])
                node.flow_rate = total_output_flow
                print(f"Set Flow Rate for {node_id} (Pump): {node.flow_rate}")

        # Balance flow rates for media sources
        for node_id in node_ids_to_check:
            node = self.nodes[node_id]
            if isinstance(node, Media):
                total_output_flow = sum(self.nodes[to_node_id].flow_rate for to_node_id, _ in self.connections[node_id])
                node.flow_rate = total_output_flow
                print(f"Set Flow Rate for {node_id} (Media): {node.flow_rate}")

        # Check if a cell's flow rate is higher than the sum of its output connections
        for node_id in node_ids_to_check:
            node = self.nodes[node_id]
            if isinstance(node, Cell):
                total_output_flow = 0
                output_is_only_cells = True

                for to_node_id, _ in self.connections[node_id]:
                    if isinstance(self.nodes[to_node_id], Cell):
                        total_output_flow += self.nodes[to_node_id].flow_rate
                    else:
                        output_is_only_cells = False

                if output_is_only_cells and node.flow_rate > total_output_flow:
                    print(f"Warning: The flow rate of cell {node_id} ({node.flow_rate}) "
                          f"is higher than the sum of the flow rates of its output cells ({total_output_flow}).")
                    
                    # Calculate warning value
                    warning_value = (node.flow_rate / total_output_flow) / total_output_flow
                    self.warnings.append((node_id, total_output_flow, warning_value))

                    # Add an output node named "waste" with flow rate equal to the excess flow
                    waste_node_id = f"W_{node_id}"
                    waste_node = Output(waste_node_id, flow_rate=node.flow_rate - total_output_flow)
                
                    self.add_node(waste_node)
                    self.add_connection(node_id, waste_node_id, flow_rate=node.flow_rate - total_output_flow)

                    print(f"Added output node {waste_node_id} with flow rate {node.flow_rate - total_output_flow}.")
                    
def generate_layout(system):
    layout = {"media": [], "output": [], "other": []}
    for node_id, node in system.nodes.items():
        if isinstance(node, Media):
            layout["media"].append((node_id, system.connections[node_id]))
        elif isinstance(node, Output):
            layout["output"].append((node_id, system.connections[node_id]))
        else:
            layout["other"].append((node_id, system.connections[node_id]))
    return layout


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

def determine_columns(system):
    # Separate nodes by type
    cells = [node_id for node_id, node in system.nodes.items() if node.node_type == 'cell']
    pumps = [node_id for node_id, node in system.nodes.items() if node.node_type == 'pump']
    muxes = [node_id for node_id, node in system.nodes.items() if node.node_type == 'mux']
    wastes = [node_id for node_id, node in system.nodes.items() if node.node_type == 'output' and node_id.startswith('W_')]
    outputs = [node_id for node_id, node in system.nodes.items() if node.node_type == 'output' and not node_id.startswith('W_')]
    medias = [node_id for node_id, node in system.nodes.items() if node.node_type == 'media']

    max_columns = 15  # Increased to allow more spacing
    columns = {i: [] for i in range(max_columns)}

    # Place media nodes in the first column
    columns[0] = medias

    # Start distributing cells from column 2
    cell_column_start = 2

    # Distribute cells across columns and ensure left-to-right flow
    for idx, cell_id in enumerate(cells):
        target_col = cell_column_start + (idx % (max_columns - 4))  # Distribute cells evenly across available columns
        columns[target_col].append(cell_id)

        # Place connected pumps to the left of the cell
        connected_pumps = [
            pump_id for pump_id in pumps
            if any(to_node == cell_id for to_node, _ in system.connections.get(pump_id, []))
        ]
        for pump_id in connected_pumps:
            pump_col = max(target_col - 1, 1)  # Ensure it doesn't overlap with media column
            if pump_id not in columns[pump_col]:
                columns[pump_col].append(pump_id)

            # Place connected muxes to the left of the pump
            connected_muxes = [
                mux_id for mux_id in muxes
                if any(to_node == pump_id for to_node, _ in system.connections.get(mux_id, []))
            ]
            for mux_id in connected_muxes:
                mux_col = max(pump_col - 1, 1)  # Ensure it doesn't overlap with media column
                if mux_id not in columns[mux_col]:
                    columns[mux_col].append(mux_id)

        # Place the waste node to the right of the cell
        waste_node_id = f"W_{cell_id}"
        if waste_node_id in wastes:
            waste_col = min(target_col + 1, max_columns - 2)  # Ensure it doesn't overlap with output column
            if waste_node_id not in columns[waste_col]:
                columns[waste_col].append(waste_node_id)

    # Place remaining pumps, muxes, and wastes not linked to cells
    for pump_id in pumps:
        if not any(pump_id in col_nodes for col_nodes in columns.values()):
            columns[1].append(pump_id)  # Place unlinked pumps in the second column

    for mux_id in muxes:
        if not any(mux_id in col_nodes for col_nodes in columns.values()):
            columns[1].append(mux_id)  # Place unlinked muxes in the second column

    for waste_id in wastes:
        if not any(waste_id in col_nodes for col_nodes in columns.values()):
            columns[max_columns - 2].append(waste_id)  # Place unlinked wastes in the second-to-last column

    # Place output nodes in the last column
    columns[max_columns - 1] = outputs

    # Debug: Print final column assignments
    print("\nFinal column assignments after optimization:")
    for col, nodes in columns.items():
        print(f"Column {col}: {nodes}")

    return columns

# import random

# import random

# import random


import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import numpy as np
import random


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

def visualize_layout_with_click(system, image_scale=0.1, edge_width=2.0):
    G = nx.DiGraph()

    # Define edge color mapping based on start and end node types
    edge_color_map = {
        ('media', 'pump'): 'blue',
        ('media', 'mux'): 'blue',
    }

    # Add nodes with layers
    for node_id, node in system.nodes.items():
        G.add_node(node_id, layer=node.node_type)

    # Add edges with specified colors
    for from_node, connections in system.connections.items():
        from_node_type = system.nodes[from_node].node_type
        for to_node, flow_rate in connections:
            to_node_type = system.nodes[to_node].node_type
            edge_type = (from_node_type, to_node_type)
            edge_color = edge_color_map.get(edge_type, 'black')
            G.add_edge(from_node, to_node, weight=flow_rate, color=edge_color)

    # Determine columns for intermediate nodes
    columns = determine_columns(system)

    # Define node positions
    pos = {}
    column_width = 1.0
    min_y, max_y = -5.0, 5.0

    # Spread nodes evenly across the full vertical range in each column
    for col_idx, node_ids in columns.items():
        num_nodes = len(node_ids)
        if num_nodes > 1:
            vertical_positions = np.linspace(min_y, max_y, num_nodes)
        else:
            vertical_positions = [(min_y + max_y) / 2]

        for i, node_id in enumerate(node_ids):
            random_offset = random.uniform(-0.01, 0.01)
            pos[node_id] = (col_idx * column_width, vertical_positions[i] + random_offset)

    # Define node images for each type
    node_images = {
        'cell': 'cell_image.png',
        'pump': 'pump_image.png',
        'valve': 'valve_image.png',
        'media': 'media_image.png',
        'mux': 'mux_image.png',
        'output': 'output_image.png',
        'controller': 'controller_image.webp'
    }

    fig, ax = plt.subplots(figsize=(15, 10))

    # Store original plot limits after initial draw
    original_xlim, original_ylim = None, None
    last_clicked_node = None

    # Draw the graph or subgraph
    def draw_graph(highlighted_nodes=None, highlighted_edges=None):
        nonlocal original_xlim, original_ylim

        ax.clear()

        if highlighted_nodes is None:
            highlighted_nodes = set(G.nodes)
        if highlighted_edges is None:
            highlighted_edges = set(G.edges())

        # Draw nodes with images
        for node_id in G.nodes:
            node = system.nodes[node_id]
            node_type = node.node_type
            img_path = node_images.get(node_type, 'default_image.png')
            img = plt.imread(img_path)

            alpha = 1.0 if node_id in highlighted_nodes else 0.2

            plt.imshow(img, extent=[pos[node_id][0] - image_scale, pos[node_id][0] + image_scale,
                                    pos[node_id][1] - image_scale, pos[node_id][1] + image_scale],
                       aspect='auto', alpha=alpha)
            plt.text(pos[node_id][0], pos[node_id][1], node_id, ha='center', va='center',
                     fontsize=10, fontweight='bold', color='black', alpha=alpha)

        # Draw edges
        for from_node, to_node, data in G.edges(data=True):
            color = data['color']
            alpha = 1.0 if (from_node, to_node) in highlighted_edges else 0.2
            nx.draw_networkx_edges(G, pos, edgelist=[(from_node, to_node)], connectionstyle="arc3,rad=0.2",
                                   edge_color=[color], width=edge_width, alpha=alpha, arrows=True)

        # Set original plot limits if not already set
        if original_xlim is None or original_ylim is None:
            original_xlim, original_ylim = ax.get_xlim(), ax.get_ylim()

        # Apply original plot limits to maintain scale
        ax.set_xlim(original_xlim)
        ax.set_ylim(original_ylim)

        plt.axis('off')
        plt.tight_layout()
        plt.draw()

    # Get directly connected nodes and edges
    def get_highlighted_elements(node):
        highlighted_nodes = {node}
        highlighted_edges = set()
        for neighbor in G.neighbors(node):
            highlighted_nodes.add(neighbor)
            highlighted_edges.add((node, neighbor))
        for predecessor in G.predecessors(node):
            highlighted_nodes.add(predecessor)
            highlighted_edges.add((predecessor, node))

        return highlighted_nodes, highlighted_edges

    # Update graph to highlight selected node and connected elements
    def update_graph(selected_node):
        nonlocal last_clicked_node

        if last_clicked_node == selected_node:  # Double-click to reset
            draw_graph()
            last_clicked_node = None
        else:
            highlighted_nodes, highlighted_edges = get_highlighted_elements(selected_node)
            draw_graph(highlighted_nodes, highlighted_edges)
            last_clicked_node = selected_node

    # Handle clicks on the plot
    def on_click(event):
        if event.inaxes != ax:
            return

        x, y = event.xdata, event.ydata
        nearest_node = None
        min_distance = float('inf')

        # Find the nearest node to the click position
        for node, (x_node, y_node) in pos.items():
            dist = (x - x_node) ** 2 + (y - y_node) ** 2
            if dist < min_distance:
                min_distance = dist
                nearest_node = node

        # Update the graph if a node is clicked
        if nearest_node is not None and min_distance < 0.1:
            update_graph(nearest_node)

    # Initial draw of the full graph
    draw_graph()

    # Connect the event handler to the figure
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()


def create_millifluidic_system():
    
    cells_info = [
        {
            "id": "C1",
            "species": "A",
            "flow_behavior": "X",
            "inputs": ["t", "b", "f"],
            "outputs": ["C2", "C3"],
            "flow_rate": 50,
            "perturbation": {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"},
            "volume": 0.02,
            "mu_max": 0.5,
            "Ks": 5,
            "Y_x_s": 60000000,
            "mu_death": 0.01,
            "OD_desired": 2,
            "K": 1e-9,
            "S_in": 200
        },
        {
            "id": "C2",
            "species": "B",
            "flow_behavior": "X",
            "inputs": ["a", "d", "f", "C1"],
            "outputs": ["C5"],
            "flow_rate": 100,
            "perturbation": {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"},
            "volume": 0.02,
            "mu_max": 0.6,
            "Ks": 4,
            "Y_x_s": 50000000,
            "mu_death": 0.01,
            "OD_desired": 1,
            "K": 1e-9,
            "S_in": 200
        },
        {
            "id": "C3",
            "species": "C",
            "flow_behavior": "X",
            "inputs": ["b", "C1"],
            "outputs": ["C6", "C9"],
            "flow_rate": 120,
            "perturbation": set(),
            "volume": 0.02,
            "mu_max": 0.4,
            "Ks": 6,
            "Y_x_s": 70000000,
            "mu_death": 0.01,
            "OD_desired": 0.6,
            "K": 1e-9,
            "S_in": 200
        },
        {
            "id": "C4",
            "species": "D",
            "flow_behavior": "X",
            "inputs": ["t", "b", "f"],
            "outputs": ["C8", "C10", "C12"],
            "flow_rate": 120,
            "perturbation": {"function = 1/4M1+1/4M2+1/2M3 for time = 0 to 6", "function = M3 for time = 6 to 7", "function = M2 for time = 7 to end"},
            "volume": 0.02,
            "mu_max": 0.45,
            "Ks": 3,
            "Y_x_s": 65000000,
            "mu_death": 0.01,
            "OD_desired": 0.7,
            "K": 1e-9,
            "S_in": 200
        },
        {
            "id": "C5",
            "species": "E",
            "flow_behavior": "X",
            "inputs": ["C2", "b", "f"],
            "outputs": ["C2", "C6", "C7", "C9", "C11"],
            "flow_rate": 50,
            "perturbation": {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"},
            "volume": 0.02,
            "mu_max": 0.55,
            "Ks": 5,
            "Y_x_s": 60000000,
            "mu_death": 0.01,
            "OD_desired": 1.5,
            "K": 1e-9,
            "S_in": 200
        },
        {
            "id": "C6",
            "species": "F",
            "flow_behavior": "X",
            "inputs": ["C3", "f", "C5"],
            "outputs": ["C7"],
            "flow_rate": 100,
            "perturbation": {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"},
            "volume": 0.02,
            "mu_max": 0.47,
            "Ks": 4.5,
            "Y_x_s": 65000000,
            "mu_death": 0.01,
            "OD_desired": 0.9,
            "K": 1e-9,
            "S_in": 200
        }
    ]
 
    # cells_info = [
        
    #       {"id": "C1", "species": "A", "flow_behavior": "X", "inputs": ["t", "b","f"], "outputs": ["C2","C3"], "flow_rate": 50, "perturbation":  {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"}},
    #       {"id": "C2", "species": "B", "flow_behavior": "X", "inputs": ["a", "d","f", "C1"], "outputs": ["C5"], "flow_rate": 100, "perturbation": {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"}},
    #       {"id": "C3", "species": "C", "flow_behavior": "X", "inputs": ["b", "C1"], "outputs": ["C6","C9"], "flow_rate": 120, "perturbation": set()},
    #       {"id": "C4", "species": "D", "flow_behavior": "X", "inputs": ["t", "b", "f"], "outputs": ["C8","C10","C12"], "flow_rate": 120, "perturbation": {"function = 1/4M1+1/4M2+1/2M3 for time = 0 to 6", "function = M3 for time = 6 to 7", "function = M2 for time = 7 to end"}},
    #       {"id": "C5", "species": "E", "flow_behavior": "X", "inputs": ["C2", "b","f"], "outputs": ["C2","C6","C7","C9","C11"], "flow_rate": 50, "perturbation":  {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"}},
    #       {"id": "C6", "species": "F", "flow_behavior": "X", "inputs": [ "C3", "f", "C5"], "outputs": ["C7"], "flow_rate": 100, "perturbation": {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"}},
    #       {"id": "C7", "species": "G", "flow_behavior": "X", "inputs": ["f","C5", "C6"], "outputs": ["Out1","C9"], "flow_rate": 120, "perturbation": set()},
    #       {"id": "C8", "species": "H", "flow_behavior": "X", "inputs": ["C4", "b"], "outputs": ["C10"], "flow_rate": 120, "perturbation": {"function = 1/4M1+1/4M2+1/2M3 for time = 0 to 6", "function = M3 for time = 6 to 7", "function = M2 for time = 7 to end"}},
    #       {"id": "C9", "species": "E", "flow_behavior": "X", "inputs": ["C3", "C5","b","h","C7"], "outputs": ["Out2"], "flow_rate": 50, "perturbation":  {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"}},
    #       {"id": "C10", "species": "F", "flow_behavior": "X", "inputs": [ "C4", "a", "C8"], "outputs": ["C11","C12"], "flow_rate": 100, "perturbation": {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"}},
    #       {"id": "C11", "species": "G", "flow_behavior": "X", "inputs": ["f","C5", "C10"], "outputs": ["C12"], "flow_rate": 120, "perturbation": set()},
    #       {"id": "C12", "species": "H", "flow_behavior": "X", "inputs": ["C4", "C10","C11"], "outputs": ["Out3"], "flow_rate": 120, "perturbation": {"function = 1/4M1+1/4M2+1/2M3 for time = 0 to 6", "function = M3 for time = 6 to 7", "function = M2 for time = 7 to end"}}


    # ]
    # cells_info = [
    #       {"id": "C1", "species": "C", "flow_behavior": "X", "inputs": ["b1"], "outputs": ["C2","C5","C62"], "flow_rate": 1000, "perturbation": set()},
    #       {"id": "C2", "species": "C", "flow_behavior": "X", "inputs": ["C1","C47"], "outputs": ["C14","C3"], "flow_rate": 980, "perturbation": set()},
    #       {"id": "C3", "species": "C", "flow_behavior": "X", "inputs": ["C2", "C45"], "outputs": ["C4"], "flow_rate": 800, "perturbation": set()},
    #       {"id": "C4", "species": "C", "flow_behavior": "X", "inputs": ["C3", "C16"], "outputs": ["C10"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C5", "species": "C", "flow_behavior": "X", "inputs": ["C1", "C46"], "outputs": ["C6"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C6", "species": "C", "flow_behavior": "X", "inputs": ["C5", "C66"], "outputs": ["C7"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C7", "species": "C", "flow_behavior": "X", "inputs": ["C6", "C48"], "outputs": ["C8"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C8", "species": "C", "flow_behavior": "X", "inputs": ["C7", "C10"], "outputs": ["C9"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C9", "species": "C", "flow_behavior": "X", "inputs": ["C8", "C11"], "outputs": ["C58"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C10", "species": "C", "flow_behavior": "X", "inputs": ["C4", "C21"], "outputs": ["C8"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C11", "species": "C", "flow_behavior": "X", "inputs": ["C12", "C46"], "outputs": ["C9"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C12", "species": "C", "flow_behavior": "X", "inputs": ["C45", "C48"], "outputs": ["C11"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C13", "species": "C", "flow_behavior": "X", "inputs": ["C48", "C49"], "outputs": ["C22"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C14", "species": "C", "flow_behavior": "X", "inputs": ["C48", "C2"], "outputs": ["C21"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C15", "species": "C", "flow_behavior": "X", "inputs": ["C48", "C51"], "outputs": ["C65"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C16", "species": "C", "flow_behavior": "X", "inputs": ["r0", "r1"], "outputs": ["C4","C20"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C17", "species": "C", "flow_behavior": "X", "inputs": ["C50", "C51"], "outputs": ["C18"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C18", "species": "C", "flow_behavior": "X", "inputs": ["C17", "C64"], "outputs": ["C24"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C19", "species": "C", "flow_behavior": "X", "inputs": ["C49", "C51"], "outputs": ["C20"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C20", "species": "C", "flow_behavior": "X", "inputs": ["C16", "C19"], "outputs": ["C25"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C21", "species": "C", "flow_behavior": "X", "inputs": ["C14", "C66"], "outputs": ["C10"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C22", "species": "C", "flow_behavior": "X", "inputs": ["C13", "C50"], "outputs": ["C23"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C23", "species": "C", "flow_behavior": "X", "inputs": ["C22", "C24"], "outputs": ["C26"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C24", "species": "C", "flow_behavior": "X", "inputs": ["C18", "C48"], "outputs": ["C23"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C25", "species": "C", "flow_behavior": "X", "inputs": ["C20", "C65"], "outputs": ["C26"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C26", "species": "C", "flow_behavior": "X", "inputs": ["C23", "C25"], "outputs": ["C27","C28","C29"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C27", "species": "C", "flow_behavior": "X", "inputs": ["C26"], "outputs": ["C30","C31"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C28", "species": "C", "flow_behavior": "X", "inputs": ["C26", "C53"], "outputs": ["C33"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C29", "species": "C", "flow_behavior": "X", "inputs": ["C26", "C53"], "outputs": ["C32"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C30", "species": "C", "flow_behavior": "X", "inputs": ["C27", "C53"], "outputs": ["C32"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C31", "species": "C", "flow_behavior": "X", "inputs": ["C27", "C53"], "outputs": ["C33"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C32", "species": "C", "flow_behavior": "X", "inputs": ["C29", "C30"], "outputs": ["C37"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C33", "species": "C", "flow_behavior": "X", "inputs": ["C28", "C31"], "outputs": ["C35","C38"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C34", "species": "C", "flow_behavior": "X", "inputs": ["C58", "C59"], "outputs": ["C35","C36","C37"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C35", "species": "C", "flow_behavior": "X", "inputs": ["C34", "C33"], "outputs": ["C40"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C36", "species": "C", "flow_behavior": "X", "inputs": ["C34"], "outputs": ["C38","C39"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C37", "species": "C", "flow_behavior": "X", "inputs": ["C34", "C32"], "outputs": ["C41"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C38", "species": "C", "flow_behavior": "X", "inputs": ["C33", "C36"], "outputs": ["C41"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C39", "species": "C", "flow_behavior": "X", "inputs": ["C32", "C36"], "outputs": ["C40"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C40", "species": "C", "flow_behavior": "X", "inputs": ["C35", "C39"], "outputs": ["C42"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C41", "species": "C", "flow_behavior": "X", "inputs": ["C37", "C38"], "outputs": ["C61"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C42", "species": "C", "flow_behavior": "X", "inputs": ["C40", "C60"], "outputs": ["C43"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C43", "species": "C", "flow_behavior": "X", "inputs": ["C42", "C61"], "outputs": ["C54"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C44", "species": "C", "flow_behavior": "X", "inputs": ["C54", "C60"], "outputs": ["C56"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C45", "species": "C", "flow_behavior": "X", "inputs": ["d1","b1"], "outputs": ["C3","C12","C47"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C46", "species": "C", "flow_behavior": "X", "inputs": ["d1", "c1","C47"], "outputs": ["C5","C66","C11"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C47", "species": "C", "flow_behavior": "X", "inputs": ["C1", "C45"], "outputs": ["C2","C46"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C48", "species": "C", "flow_behavior": "X", "inputs": ["r0", "r1"], "outputs": ["C12","C13","C14","C15","C24"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C49", "species": "C", "flow_behavior": "X", "inputs": ["d0", "c0","b0"], "outputs": ["C50","C51","C19","C13"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C50", "species": "C", "flow_behavior": "X", "inputs": ["c0", "d0","C49"], "outputs": ["C17","C64"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C51", "species": "C", "flow_behavior": "X", "inputs": ["b0", "C49"], "outputs": ["C15","C19"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C52", "species": "C", "flow_behavior": "X", "inputs": ["m0", "t0"], "outputs": ["C53"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C53", "species": "C", "flow_behavior": "X", "inputs": ["a0", "C52"], "outputs": ["C28","C29","C30","C31"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C54", "species": "C", "flow_behavior": "X", "inputs": ["s0", "s1","C43"], "outputs": ["C44","C55"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C55", "species": "C", "flow_behavior": "X", "inputs": ["s1", "C54", "C60"], "outputs": ["C62"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C56", "species": "C", "flow_behavior": "X", "inputs": ["s1", "C43","C44"], "outputs": ["C57"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C57", "species": "C", "flow_behavior": "X", "inputs": ["C56", "C51","b0"], "outputs": ["C63"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C58", "species": "C", "flow_behavior": "X", "inputs": ["a0", "C9"], "outputs": ["C34","C59"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C59", "species": "C", "flow_behavior": "X", "inputs": ["C58", "m1"], "outputs": ["C34","C60"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C60", "species": "C", "flow_behavior": "X", "inputs": ["t1", "C59"], "outputs": ["C42","C61","C55","C44"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C61", "species": "C", "flow_behavior": "X", "inputs": ["C41", "C60"], "outputs": ["C43"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C62", "species": "C", "flow_behavior": "X", "inputs": ["C1", "C55","b1"], "outputs": ["Out1","C63"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C63", "species": "C", "flow_behavior": "X", "inputs": ["C57", "C62"], "outputs": ["Out0"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C64", "species": "C", "flow_behavior": "X", "inputs": ["b0", "C50"], "outputs": ["C65","C18"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C65", "species": "C", "flow_behavior": "X", "inputs": ["C15", "C64"], "outputs": ["C25"], "flow_rate": 90, "perturbation": set()},
    #       {"id": "C66", "species": "C", "flow_behavior": "X", "inputs": ["b1", "C46"], "outputs": ["C6","C21"], "flow_rate": 90, "perturbation": set()},

    # ]
    
    
    system = MillifluidicSystem()
    existing_nodes = {}

    for cell_info in cells_info:
        cell_id = cell_info["id"]
        species = cell_info["species"]
        flow_behavior = cell_info["flow_behavior"]
        perturbation = cell_info["perturbation"]
        cell_flow_rate = cell_info.get("flow_rate", 0)

        # Create the cell node
        cell = Cell(cell_id, species=species, flow_behavior=flow_behavior, flow_rate=cell_flow_rate, petribution=perturbation)
        system.add_node(cell)
        existing_nodes[cell_id] = 'cell'

        # Add a waste node for each cell
        waste_id = f"W_{cell_id}"
        waste_node = Output(waste_id)
        system.add_node(waste_node)
        system.add_connection(cell_id, waste_id, flow_rate=0)

        # Separate cell inputs and media inputs
        cell_inputs = [input_node for input_node in cell_info["inputs"] if input_node.startswith('C')]
        media_inputs = [input_node for input_node in cell_info["inputs"] if not input_node.startswith('C')]

        # If all inputs are cells, add a single pump and remove direct connections
        if cell_inputs and not media_inputs:
            combined_pump_id = f"P_{cell_id}"
            combined_pump = Pump(combined_pump_id)
            system.add_node(combined_pump)

            # Connect all cell inputs to the combined pump
            for input_node in cell_inputs:
                if input_node not in existing_nodes:
                    system.add_node(Cell(input_node))
                    existing_nodes[input_node] = 'cell'

                # Remove direct connections from input_node to cell_id
                if cell_id in [conn[0] for conn in system.connections[input_node]]:
                    system.connections[input_node] = [(to_node, flow) for to_node, flow in system.connections[input_node] if to_node != cell_id]

                # Connect input_node to the combined pump
                system.add_connection(input_node, combined_pump_id, 0)

            # Connect the combined pump to the cell
            system.add_connection(combined_pump_id, cell_id, 0)

        # Handle media inputs
        elif media_inputs:
            if len(media_inputs) > 1:
                # Create a MUX for multiple media inputs
                mux_id = f"V_{cell_id}"
                mux = MUX(mux_id, num_inputs=len(media_inputs), petribution=perturbation)
                system.add_node(mux)

                # Connect media inputs to MUX
                for input_node in media_inputs:
                    if input_node not in existing_nodes:
                        system.add_node(Media(input_node))
                        existing_nodes[input_node] = 'media'
                    system.add_connection(input_node, mux_id, 0)

                # Connect MUX to a pump and then to the cell
                pump_id = f"P_{cell_id}"
                pump = Pump(pump_id)
                system.add_node(pump)
                system.add_connection(mux_id, pump_id, 0)
                system.add_connection(pump_id, cell_id, 0)

            else:
                # Add a pump for single media input
                pump_id = f"P_{cell_id}"
                pump = Pump(pump_id)
                system.add_node(pump)

                input_node = media_inputs[0]
                if input_node not in existing_nodes:
                    system.add_node(Media(input_node))
                    existing_nodes[input_node] = 'media'
                system.add_connection(input_node, pump_id, 0)

                # Connect the pump to the cell
                system.add_connection(pump_id, cell_id, 0)

        # Handle outputs from the cell
        for output_node in cell_info["outputs"]:
            if output_node not in existing_nodes:
                if output_node.startswith('Out'):
                    system.add_node(Output(output_node))
                    existing_nodes[output_node] = 'output'
                else:
                    system.add_node(Media(output_node))
                    existing_nodes[output_node] = 'media'
            system.add_connection(cell_id, output_node, 0)

    return system




# Define the Millifluidic System with flow computations
# class MillifluidicSystemWithFlow:
#     def __init__(self, cells_info):
#         self.cells_info = cells_info
#         self.connections = {i: cell["outputs"] for i, cell in enumerate(cells_info)}
#         self.precomputed_fractions = None

#     def compute_fractions(self, connections, sender_status, receiver_status):
#         if not connections:  # No connections, 100% to waste
#             return {"waste": 1.0}

#         weights = {}
#         for receiver in connections:
#             if receiver == "waste":
#                 continue
#             if sender_status == "Good" and receiver_status.get(receiver) == "Good":
#                 weights[receiver] = 1
#             elif (sender_status == "Good" and receiver_status.get(receiver) == "Bad") or (
#                 sender_status == "Bad" and receiver_status.get(receiver) == "Good"
#             ):
#                 weights[receiver] = 2
#             elif sender_status == "Bad" and receiver_status.get(receiver) == "Bad":
#                 weights[receiver] = 4

#         # Normalize weights to calculate flow fractions
#         total_weight = sum(weights.values())
#         fractions = {}
#         if total_weight > 0:
#             for receiver, weight in weights.items():
#                 fractions[receiver] = (weight / total_weight) * 0.2
#         fractions["waste"] = 0.8 if total_weight > 0 else 1.0

#         return fractions

#     def precompute_fractions(self):
#         fractions = {}
#         for i, cell in enumerate(self.cells_info):
#             sender_status = cell["status"]["sender"]
#             receiver_status = {j: self.cells_info[j]["status"]["receiver"] for j in cell["outputs"] if j != "waste"}
#             fractions[i] = self.compute_fractions(cell["outputs"], sender_status, receiver_status)
#         return fractions

#     def equations_with_splits(self, vars):
#         n = len(self.cells_info)
#         F_out = vars[:n]  # Outflow rates
#         S = vars[n:]      # Substrate concentrations
#         eqs = []

#         for i, cell in enumerate(self.cells_info):
#             # Retrieve precomputed fractions
#             fractions = self.precomputed_fractions[i]

#             # Reactor constants
#             V, mu_max, Ks, Y_x_s, mu_death, OD_desired, K, S_in = (
#                 cell["volume"], cell["mu_max"], cell["Ks"], cell["Y_x_s"],
#                 cell["mu_death"], cell["OD_desired"], cell["K"], cell["S_in"]
#             )

#             # Total inflow to the reactor
#             inflow = sum(F_out[src] * fractions.get(src, 0) for src in range(n))
#             F_media = 0.01  # Assume constant media inflow rate
#             total_inflow = inflow + F_media

#             # Effective substrate inflow
#             S_in_eff = (F_media * S_in + sum(F_out[src] * fractions.get(src, 0) * S[src] for src in range(n))) / V

#             # Growth rate
#             mu = mu_max * S[i] / (Ks + S[i])

#             # Growth-dilution balance
#             eqs.append(mu - (mu_death + F_out[i] / V))

#             # Substrate balance
#             eqs.append(S[i] - (S_in_eff - (mu * OD_desired / K * V * S[i]) /
#                                (Y_x_s * F_out[i] * (Ks + S[i]))))

#         return eqs

#     def solve_flow_behavior(self):
#         self.precomputed_fractions = self.precompute_fractions()

#         # Initial guesses for F_out and S
#         n = len(self.cells_info)
#         F_out_guess = [0.1] * n
#         S_guess = [100] * n  # Start with substrate concentration halfway to S_in
#         initial_guess = F_out_guess + S_guess

#         # Solve the system with precomputed fractions
#         solution = fsolve(self.equations_with_splits, initial_guess)

#         # Extract results
#         F_out_sol = solution[:n]
#         S_sol = solution[n:]

#         # Update cells_info with solved flow rates and substrate concentrations
#         for i, cell in enumerate(self.cells_info):
#             cell["flow_rate_solved"] = max(0, F_out_sol[i])  # Ensure non-negative
#             cell["substrate_concentration_solved"] = max(0, S_sol[i])  # Ensure non-negative

#             # Print results for debugging
#             print(f"Cell {cell['id']}:")
#             print(f"  Solved flow rate: {cell['flow_rate_solved']:.4f} mL/min")
#             print(f"  Solved substrate concentration: {cell['substrate_concentration_solved']:.4f} mg/mL")

# # Example setup with cells_info
# cells_info = [
#     {"id": "C1", "species": "A", "flow_behavior": "X", "inputs": [], "outputs": [1, "waste"],
#      "flow_rate": 50, "perturbation": {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"},
#      "volume": 0.02, "mu_max": 0.5, "Ks": 5, "Y_x_s": 60000000, "mu_death": 0.01, "OD_desired": 2, "K": 1e-9, "S_in": 200,
#      "status": {"sender": "Good", "receiver": "Good"}},
#     {"id": "C2", "species": "B", "flow_behavior": "X", "inputs": [0], "outputs": [2, 3],
#      "flow_rate": 100, "perturbation": {"function = M1 for time = 0 to 4", "function = M3 for time = 4 to end"},
#      "volume": 0.02, "mu_max": 0.6, "Ks": 4, "Y_x_s": 50000000, "mu_death": 0.01, "OD_desired": 1, "K": 1e-9, "S_in": 200,
#      "status": {"sender": "Bad", "receiver": "Bad"}},
#     {"id": "C3", "species": "C", "flow_behavior": "X", "inputs": [1], "outputs": [3, "waste"],
#      "flow_rate": 120, "perturbation": set(),
#      "volume": 0.02, "mu_max": 0.4, "Ks": 6, "Y_x_s": 70000000, "mu_death": 0.01, "OD_desired": 0.6, "K": 1e-9, "S_in": 200,
#      "status": {"sender": "Good", "receiver": "Bad"}},
#     {"id": "C4", "species": "D", "flow_behavior": "X", "inputs": [1, 2], "outputs": ["waste"],
#      "flow_rate": 120, "perturbation": {"function = 1/4M1+1/4M2+1/2M3 for time = 0 to 6",
#                                         "function = M3 for time = 6 to 7", "function = M2 for time = 7 to end"},
#      "volume": 0.02, "mu_max": 0.45, "Ks": 3, "Y_x_s": 65000000, "mu_death": 0.01, "OD_desired": 0.7, "K": 1e-9, "S_in": 200,
#      "status": {"sender": "Bad", "receiver": "Good"}}
# ]

# # Initialize and solve flow behavior
# system = MillifluidicSystemWithFlow(cells_info)
# system.solve_flow_behavior()


# Example usage
system = create_millifluidic_system()
system.set_flow_rates()
layout = generate_layout(system)
visualize_layout_with_click(system, image_scale=0.2, edge_width=1.0)
