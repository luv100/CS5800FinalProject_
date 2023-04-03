

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import heapdict as hd
from itertools import chain, combinations
import time
import glob
import utils
import sys
import plotly.graph_objs as go
import plotly
import os

def data_parser(input_data):
    number_of_locations = int(input_data[0][0])
    number_of_houses = int(input_data[1][0])
    list_of_locations = input_data[2]
    list_of_houses = input_data[3]
    starting_location = input_data[4][0]

    adjacency_matrix = [[entry if entry == 'x' else float(entry) for entry in row] for row in input_data[5:]]
    return number_of_locations, number_of_houses, list_of_locations, list_of_houses, starting_location, adjacency_matrix



def plotGraph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=20,  # Increased size of the node markers
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])  # Changed the way 'x' and 'y' coordinates are being appended
        node_trace['y'] += tuple([y])

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'Node {node}<br># of connections: {len(adjacencies[1])}')

    node_trace['marker']['color'] = node_adjacencies
    node_trace['text'] = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph made with Plotly',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    plotly.offline.plot(fig, filename='C:/gurobi1001/win64/examples/RaoTA-master/algs/semitreeTSPTrial_v2/lifeExp.html')
    fig.show()
    
    
def adjacency_matrix_to_graph(adjacency_matrix):

    node_weights = [adjacency_matrix[i][i] for i in range(len(adjacency_matrix))]
    adjacency_matrix_formatted = [[0 if entry == 'x' else entry for entry in row] for row in adjacency_matrix]

    for i in range(len(adjacency_matrix_formatted)):
        adjacency_matrix_formatted[i][i] = 0

    G = nx.convert_matrix.from_numpy_matrix(np.matrix(adjacency_matrix_formatted))

    message = ''

    for node, datadict in G.nodes.items():
        if node_weights[node] != 'x':
            message += 'The location {} has a road to itself. This is not allowed.\n'.format(node)
        datadict['weight'] = node_weights[node]

    return G, message

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):   

    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    plotGraph(G)
    

def solve_from_file(input_file, output_directory, params=[]):
    #print('Processing {}...'.format(input_file), end="")
    sys.stdout.flush()

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    
    
    

def plot_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')
    
    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)
        
        
        
if __name__=="__main__":
    
    class Args:
        def __init__(self, all=False, input=None, output_directory='.', params=None):
            self.all = all
            self.input = input
            self.output_directory = output_directory
            self.params = params
    
    args = Args(all=False, input="Inputs_to_test_current_code", output_directory="output", params=None)
    #args = parser.parse_args()
    #("args.all",args.all)
    output_directory = args.output_directory
    #print("args.all",args.all)
    if args.all:
        
        input_directory = args.input
        f = glob.glob(args.input + "/*.in")
       
        input_file = f[0]
        plot_all(input_directory, output_directory, params=args.params)
    else:
        #print("Inside Solver Trial TSP ==================>")
        
        #print("args.input",args.input)
        #print("glob.glob(args.input + ",glob.glob(args.input + "/*.in"))
        f = glob.glob(args.input + "/*.in")
        
       
        input_file = f[0]
        #input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
        
    outputs = "1_output_practiceInsideFolder"
    outfile = os.path.splitext(os.path.basename(input_file))[0] + ".out"
    oldfile = outputs + "/" + outfile
    

