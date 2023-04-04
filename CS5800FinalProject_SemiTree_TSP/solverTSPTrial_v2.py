import os
import sys
import subprocess
import argparse
from functools import reduce
sys.path.append("libs")
import utils
import optitsp
import brute
import gurobilp
import glob

import output_validator as ov
import plotly.graph_objs as go
import plotly
from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

def get_costs(infile, outfile): # driving, walking, total

    #print("infile",infile)
    #print("outfile",outfile)
    indat = ov.utils.read_file(infile)
    outdat = ov.utils.read_file(outfile)
    cost, msg = ov.tests(indat, outdat)
    return [float(s.split(" ")[-1][:-1]) for s in msg.split('\n')[:-1]]

def get_naive(infile):
    # read file
    f = open(infile)
    l = f.readlines()
    home, dropoffs = l[4], l[4][:-1] + " " + l[3]
    f.close()
    # write file
    name = infile.split("/")[-1][:-3]
    f = open("temp/naive.out", 'w')
    f.write(home)
    f.write("1\n")
    f.write(dropoffs)
    f.close()
    _, _, nt = get_costs(infile, "temp/naive.out")
    subprocess.run("rm temp/naive.out", shell=True)
    return nt


    
# =============================================================================
# def plotGraph(G):
#     pos = nx.spring_layout(G)
#     labels = nx.get_edge_attributes(G, 'weight')
# 
#     edge_trace = go.Scatter(
#         x=[],
#         y=[],
#         line=dict(width=1, color='#888'),
#         hoverinfo='none',
#         mode='lines')
#     
#     
#     #edge_trace['x'] = list(edge_trace['x'])
#     #edge_trace['y'] = list(edge_trace['y'])
#     print("edge_trace['x']",edge_trace['x'])
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_trace['x'] += tuple([x0, x1, None])
#         edge_trace['y'] += tuple([y0, y1, None])
# 
#     node_trace = go.Scatter(
#         x=[],
#         y=[],
#         text=[],
#         mode='markers',
#         hoverinfo='text',
#         marker=dict(
#             showscale=False,
#             colorscale='YlGnBu',
#             reversescale=True,
#             color=[],
#             size=10,
#             colorbar=dict(
#                 thickness=15,
#                 title='Node Connections',
#                 xanchor='left',
#                 titleside='right'
#             ),
#             line=dict(width=2)))
# 
#     for node in G.nodes():
#         x, y = pos[node]
#         node_trace['x'] +=x
#         node_trace['y']+=y
#         #node_trace['x'].append(x)
#         #node_trace['y'].append(y)
# 
#     node_adjacencies = []
#     node_text = []
#     for node, adjacencies in enumerate(G.adjacency()):
#         node_adjacencies.append(len(adjacencies[1]))
#         node_text.append(f'Node {node}<br># of connections: {len(adjacencies[1])}')
# 
#     node_trace['marker']['color'] = node_adjacencies
#     node_trace['text'] = node_text
#     
#     print("edge_trace, node_trace",edge_trace, node_trace)
#     fig = go.Figure(data=[edge_trace, node_trace],
#                     layout=go.Layout(
#                         title='<br>Network graph made with Plotly',
#                         titlefont=dict(size=16),
#                         showlegend=False,
#                         hovermode='closest',
#                         margin=dict(b=20,l=5,r=5,t=40),
#                         annotations=[dict(
#                             text="",
#                             showarrow=False,
#                             xref="paper", yref="paper",
#                             x=0.005, y=-0.002)],
#                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
# 
#     plotly.offline.plot(fig, filename='C:/gurobi1001/win64/examples/RaoTA-master/algs/semitreeTSPTrial/lifeExp.html')
#     fig.show()
# =============================================================================
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

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A list of (location, [homes]) representing drop-offs
    """

    t0 = time.perf_counter()

    # Create graph
    print("Line 213: adjacency_matrix",adjacency_matrix)
    G, _ = adjacency_matrix_to_graph(adjacency_matrix)
    #print("Line 215, plotGraph commented.")
    plotGraph(G)
    # Convert locations to indices
    stas = set([list_of_locations.index(h) for h in list_of_homes])
    sloc = list_of_locations.index(starting_car_location)

    artmap = getArtmap(sloc, stas, G)
    
    print("line 220, articulation map", artmap)
    listlocs, listdropoffs = treesolve(sloc, stas, G, artmap)

    t1 = time.perf_counter() - t0
    #print(" done! Time: {}s".format(t1))

    return listlocs, listdropoffs

def getArtmap(sloc, stas, G):
    
    # Find biconnected components and get in nice form
    bccs = list(nx.biconnected_components(G))
    
    #print("\nLine 233: biconnected components in the graph",bccs )
    
    artpts = set(nx.articulation_points(G))
    #print("\nLine 236: Articulation Points in the graph", artpts)

    # articulation point -> (biconnected component, entire subgraph involving bcc, artpts in that bcc)
    
    # working of the line below:
    """
    In the big picture, this line of code initializes the artmap dictionary, which is used to store information about the relationship between articulation points and biconnected components in the given graph G.

    The artmap dictionary has the following structure:
    
    Each key in the dictionary is an articulation point from the graph.
    
    The value associated with each key is a list of tuples. Each tuple 
    represents a biconnected component that contains the corresponding articulation point.
    
    The tuples contain three elements:
        
    The biconnected component itself (a set of nodes).
    
    An empty set, which will later be populated with the entire subgraph 
    involving the biconnected component.
    A set of other articulation points that are part of the biconnected 
    component, excluding the current articulation point (the key).
    """
    artmap = {i:[(b, set(), b.intersection(artpts) - {i}) for b in bccs if i in b] for i in artpts}
    
    print("\n Line 264: Articulation Map", artmap)
    
    
    """
    In the big picture, this block of code ensures that the starting location 
    sloc is included in the artmap dictionary, even if it is not an articulation point. 
    
    In summary, adding sloc to the artmap ensures that the starting location 
    is treated as an essential part of the graph structure and the TSP variant
    solution. This approach allows the algorithm to effectively break down the 
    problem into smaller subproblems, analyze the graph's connectivity, and 
    ultimately construct an optimal solution.
    
    """
    
    if sloc not in artmap:
        
        cc = [i for i in bccs if sloc in i][0] # Find the biconnected component sloc is part of
        artmap[sloc] = [(cc, set(), cc.intersection(artpts))]

    # gets rid of parent in each set using BFS
    parents = {sloc}
    while parents:
        newparents = set()
        for p in parents:
            childs = set() # get all the children
            for t in artmap[p]:
                childs.update(t[2])
            for c in childs:
                artmap[c] = [i for i in artmap[c] if p not in i[0]] # remove the parent from all of them
                for t in artmap[c]:
                    t[2].difference_update({p})
            newparents.update(childs)
        parents = newparents

    generateSubgraphsDFS(artmap, sloc) # avoid too much recomputation and also pretty fun
    
    #print("Line 29: artmap while returning, artmap: ",           artmap)
    return artmap

def generateSubgraphsDFS(artmap, sloc):
    for bcc, subg, artps in artmap[sloc]:
        subg.update(bcc)
        for c in artps:
            generateSubgraphsDFS(artmap, c)
            for _, c_subg, _ in artmap[c]:
                subg.update(c_subg)

def treesolve(sloc, stas, G, artmap):
    #print("Inside treesolve, sloc", sloc)
    #print("Inside treesolve, stas", stas)
    #print("Inside treesolve, G", G)
    #print("Inside treesolve, artmap", artmap)
    
    # stas (a set of locations) ==> to be dropped off. ; given in the input file
    # in the input file(1000_50.in), set of location
    # given  2,3,5,6 and starting point is 4. 
    # this is converted in the code as 1,2,4,5 (because of indices change)
    # and starting point is 3. 
    lst_listlocs, lst_listdropoffs = [], []
    
    #print("artmap[sloc]",artmap[sloc])
    for bccset, subgset, artpts in artmap[sloc]:
        
        
        subg_tas = stas.intersection(subgset)
        
        if len(subg_tas) == 0:
            
            lst_listlocs.append([sloc])
            
            lst_listdropoffs.append({})
            
        elif len(subg_tas) == 1 or (len(subg_tas) == 2 and sloc in subg_tas):
            
            lst_listlocs.append([sloc])
            
            lst_listdropoffs.append({sloc:subg_tas})
            
        else:
            
            # Generate biconnected graph to run optitsp on
            #nx.draw(G)
            #=============================================================================================================>
            #plotGraph(G)
            #=============================================================================================================>
            #plt.savefig("graph_50_input.png")
            bccg = G.subgraph(bccset).copy()
            
            bccorigtas = stas.intersection(bccset) # so we don't dropoff fake tas
            
            bcctas = bccorigtas.copy()
            
            bcctasmap = {} # for converting fake ta back to orig
            
            bccforceta = set()
            bccinsert = {} # Stores recursive calls to stitch in

            # Convert artpts to homes and perform recursive calls if necessary
            for ap in artpts:
                ap_subg = reduce(lambda a,b: a.union(b), [i[1] for i in artmap[ap]])
                ap_tas = stas.intersection(ap_subg)
                if len(ap_tas) == 1:
                    bcctas.add(ap)
                    bcctasmap[ap] = ap_tas.pop()
                elif len(ap_tas) >= 2: # if more than 2 tas then always optimal to enter subgraph
                    bcctas.add(ap)
                    bccforceta.add(ap)
                    bccinsert[ap] = treesolve(ap, ap_tas, G.subgraph(ap_subg).copy(), artmap)
            
            print("Line 369: bcctas", bcctas)
            print("Line 370: bccg", bccg)
            # Run optitsp (or brute)
            if len(bcctas) >= 3:
                print("=====")
                optitsp_locs, optitsp_dropoffs = optitsp.solve(sloc, bcctas, bccg, donttouch=bccforceta)
            else:
                #print("Line 374, bcctas", bcctas)
                optitsp_locs, optitsp_dropoffs = brute.solve(sloc, bcctas, bccg, donttouch=bccforceta)
            
            # Stitch everything together
            bcclocs = []
            bcc_inserted = []
            for i in optitsp_locs:
                if i in bccinsert and i not in bcc_inserted:
                    bcclocs.extend(bccinsert[i][0])
                    bcc_inserted.append(i) # don't double insert
                else:
                    bcclocs.append(i)
            bccdropoffs = {}
            for i in optitsp_dropoffs:
                newset = {bcctasmap[t] if t in bcctasmap else t for t in optitsp_dropoffs[i]}.intersection(stas)
                if len(newset) > 0: # possible for newset to be empty
                    bccdropoffs[i] = newset
            for i in bccinsert:
                for dppt in bccinsert[i][1]:
                    if dppt in bccdropoffs:
                        bccdropoffs[dppt].update(set(bccinsert[i][1][dppt]))
                    else:
                        bccdropoffs[dppt] = set(bccinsert[i][1][dppt])

            # Add to our running lists
            lst_listlocs.append(bcclocs)
            lst_listdropoffs.append(bccdropoffs)

    # Reconstruct listlocs and listdropoffs
    listlocs = lst_listlocs[0]
    listdropoffs = lst_listdropoffs[0]
    for l in lst_listlocs[1:]:
        listlocs.extend(l[1:])
    for dps in lst_listdropoffs[1:]:
        for d in dps:
            if d in listdropoffs:
                listdropoffs[d].update(dps[d])
            else:
                listdropoffs[d] = dps[d]

    # Convert all dropoff sets to lists
    for d in listdropoffs:
        listdropoffs[d] = list(listdropoffs[d])

    return listlocs, listdropoffs
                
"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    #print('Processing {}...'.format(input_file), end="")
    sys.stdout.flush()

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')
    print("Line 470, input_files", input_files)

    for input_file in input_files:
        print("Line 473 input_file", input_file)
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    
    class Args:
        def __init__(self, all=False, input=None, output_directory='.', params=None):
            self.all = all
            self.input = input
            self.output_directory = output_directory
            self.params = params
    
    args = Args(all=True, input="1_input_practiceInsideFolder", output_directory="1_output_practiceInsideFolder", params=None)
    #args = parser.parse_args()
    #("args.all",args.all)
    output_directory = args.output_directory
    #print("args.all",args.all)
    if args.all:
        
        input_directory = args.input
        
        f = glob.glob(args.input + "/*.in")
        
       
        input_file = f[0]
        
        print("Line 498: input_file", input_file)
        print("Line 499: f", f)
        print("Line500: input_directory", input_directory)
        
        
        
        solve_all(input_directory, output_directory, params=args.params)
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
    
    _, _, t = get_costs(input_file,  oldfile)
    nt = get_naive(input_file)
    ot = get_costs(input_file, oldfile)[2] if os.path.exists(oldfile) else float('inf')
    print("new: {}% old: {}%".format(round((nt-t)/nt*100,2), round((nt-ot)/nt*100, 2)), end="")


        
