from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, id, node_type='cell', connections=None):
        self.id = id
        self.node_type = node_type
        self.connections = connections if connections else []

class Cell(Node):
    def __init__(self, id, species=None, flow_behavior=None, connections=None):
        super().__init__(id, node_type='cell', connections=connections)
        self.species = species
        self.flow_behavior = flow_behavior

class Pump(Node):
    def __init__(self, id, flow_behavior=None, connections=None):
        super().__init__(id, node_type='pump', connections=connections)
        self.flow_behavior = flow_behavior

class Valve(Node):
    def __init__(self, id, state=None, connections=None):
        super().__init__(id, node_type='valve', connections=connections)
        self.state = state

class Media(Node):
    def __init__(self, id, name=None, connections=None):
        super().__init__(id, node_type='media', connections=connections)
        self.name = name
        
class MUX(Node):
    def __init__(self, id, num_inputs, connections=None):
        super().__init__(id, node_type='mux', connections=connections)
        self.num_inputs = num_inputs
        self.name = f"MUX{num_inputs}->1"


        
class Output(Node):
    def __init__(self, id, connections=None):
        super().__init__(id, node_type='output', connections=connections)

class MillifluidicSystem:
    def __init__(self):
        self.nodes = {}
        self.connections = defaultdict(list)

    def add_node(self, node):
        self.nodes[node.id] = node

    def add_connection(self, from_node, to_node, flow_rate):
        self.connections[from_node].append((to_node, flow_rate))
        # If bidirectional, also add the reverse connection
        # self.connections[to_node].append((from_node, flow_rate))

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
def visualize_layout(system):
    G = nx.DiGraph()

    for node_id, node in system.nodes.items():
        G.add_node(node_id, layer=node.node_type)  

    for from_node, connections in system.connections.items():
        for to_node, flow_rate in connections:
            G.add_edge(from_node, to_node, weight=flow_rate)

    # Set the size of the figure
    plt.figure(figsize=(14, 6))

    # Assign positions for nodes using grid layout
    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")

    # Customizing positions for media, pump, cell, MUX, and output nodes
    for node_id, node_type in pos.items():
        if system.nodes[node_id].node_type == 'media':
            pos[node_id][0] = 1  # Move media nodes to the leftmost column
        elif system.nodes[node_id].node_type == 'mux':
            pos[node_id][0] = 2  # Move MUX nodes to the second column
        elif system.nodes[node_id].node_type == 'pump':
            pos[node_id][0] = 3  # Move pump nodes to the third column
        elif system.nodes[node_id].node_type == 'cell':
            pos[node_id][0] = 4  # Move cell nodes to the fourth column
        elif system.nodes[node_id].node_type == 'output':
            pos[node_id][0] = 5  # Move output nodes to the fifth column

    # Draw the images of components and display node names
    node_images = {
        'cell': 'cell_image.png', 
        'pump': 'pump_image.png', 
        'valve': 'valve_image.png',  
        'media': 'media_image.png',   
        'mux': 'mux_image.png',  
        'output': 'media_image.png'  
    }

    for node_id, node in system.nodes.items():
        node_type = node.node_type
        img_path = node_images.get(node_type, 'default_image.png')  
        img = plt.imread(img_path)
        plt.imshow(img, extent=[pos[node_id][0] - 0.05, pos[node_id][0] + 0.05,
                                pos[node_id][1] - 0.05, pos[node_id][1] + 0.05], aspect='auto')

        # Display node names above the images
        if(node_type!="mux"):
            plt.text(pos[node_id][0], pos[node_id][1] + 0.08, node_id, ha='center', fontsize=10, fontweight='bold')
        else:
            plt.text(pos[node_id][0], pos[node_id][1] + 0.08, node.name, ha='center', fontsize=10, fontweight='bold')
      

    # Draw the edges between nodes with curved lines
    nx.draw_networkx_edges(
        G, pos,
        connectionstyle="arc3,rad=0.2",  # This creates curved edges
        arrows=True
    )

    plt.axis('off')
    plt.tight_layout()  # Adjust layout t
    plt.show()






# Example usage
# system = MillifluidicSystem()
# system.add_node(Cell(1, species='SpeciesA', flow_behavior='FlowBehaviorA'))
# system.add_node(Cell(2, species='SpeciesB', flow_behavior='FlowBehaviorB'))
# system.add_node(Cell(3, species='SpeciesC', flow_behavior='FlowBehaviorC'))
# system.add_node(Pump(4, flow_behavior='PumpFlowBehavior'))
# system.add_node(Valve(5, state='open'))
# system.add_node(Media(6, name='MediaName'))
# system.add_node(Output(7))
# system.add_connection(1, 2, 10)
# system.add_connection(2, 3, 5)
# system.add_connection(1, 4, 15)
# system.add_connection(4, 2, 10)
# system.add_connection(4, 5, 8)
# system.add_connection(5, 3, 7)
# system.add_connection(6, 1, 20)
# system.add_connection(3, 7, 10)  # Adding output connection

# layout = generate_layout(system)
# visualize_layout(system)


# Define inputs based on the provided example
def create_millifluidic_system():
    cells_info = [
        {"id": "C1", "species": "A", "flow_behavior": "X", "inputs": ["M1", "M2"], "outputs": ["C2"], "perturbation": True},
        {"id": "C2", "species": "B", "flow_behavior": "X", "inputs": ["M1", "M3", "C1"], "outputs": ["C3"], "perturbation": True},
        {"id": "C3", "species": "C", "flow_behavior": "X", "inputs": ["M1", "C2"], "outputs": ["Out1"], "perturbation": False},
        {"id": "C4", "species": "D", "flow_behavior": "X", "inputs": ["M1", "M2", "M3"], "outputs": ["Out2"], "perturbation": True}
    ]

    medias_info = ["M1", "M2", "M3"]

    system = MillifluidicSystem()
    existing_nodes = {}

    for cell_info in cells_info:
        cell_id = cell_info["id"]
        species = cell_info["species"]
        flow_behavior = cell_info["flow_behavior"]
        perturbation_flag = cell_info["perturbation"]
        media_inputs = [input_node for input_node in cell_info["inputs"] if input_node.startswith('M')]
        cell = Cell(cell_id, species=species, flow_behavior=flow_behavior)

        if len(media_inputs) > 1 and perturbation_flag:
            num_media_inputs = len(media_inputs)
            mux_id = f"MUX{num_media_inputs}_{cell_id}"
            mux = MUX(mux_id, num_inputs=num_media_inputs)
            pump_id = f"P_{cell_id}"
            pump = Pump(pump_id)
            system.add_node(mux)
            system.add_node(pump)
            system.add_connection(mux_id, pump_id, 0)
            system.add_connection(pump_id, cell_id, 0)
            for input_node in media_inputs:
                if input_node not in existing_nodes:
                    system.add_node(Media(input_node))
                    existing_nodes[input_node] = 'media'
                system.add_connection(input_node, mux_id, 0)
        else:
            for input_node in media_inputs:
                if input_node not in existing_nodes:
                    system.add_node(Media(input_node))
                    existing_nodes[input_node] = 'media'
                pump_id = f"P_{cell_id}" if len(media_inputs) == 1 else None
                if pump_id:
                    pump = Pump(pump_id)
                    system.add_node(pump)
                    system.add_connection(input_node, pump_id, 0)
                    system.add_connection(pump_id, cell_id, 0)
                else:
                    system.add_connection(input_node, cell_id, 0)

        for input_node in cell_info["inputs"]:
            if input_node.startswith('C'):
                if input_node not in existing_nodes:
                    system.add_node(Cell(input_node))
                    existing_nodes[input_node] = 'cell'
                system.add_connection(input_node, cell_id, 0)

        for output_node in cell_info["outputs"]:
            if output_node.startswith('M'):
                if output_node not in existing_nodes:
                    system.add_node(Media(output_node))
                    existing_nodes[output_node] = 'media'
                system.add_connection(cell_id, output_node, 0)
            elif output_node.startswith('Out'):
                if output_node not in existing_nodes:
                    system.add_node(Output(output_node))
                    existing_nodes[output_node] = 'output'
                system.add_connection(cell_id, output_node, 0)

        system.add_node(cell)

    for media_name in medias_info:
        if media_name not in existing_nodes:
            system.add_node(Media(media_name))
            existing_nodes[media_name] = 'media'
        else:
            print(f"Error: {media_name} has already been defined as a node.")

    return system

# Example usage
system = create_millifluidic_system()
layout = generate_layout(system)
visualize_layout(system)

