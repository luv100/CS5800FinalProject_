Details about the folders:

1. All_inputs: All inputs that code can be tested on. 

2. Inputs_to_test_current_code: These are the specific inputs, which should cover all the 
	extreme/edge cases. 
	
3. readInputGetGraph.py: code to get an read input files, create an adjacency matrix, graphs and plotting of the graphs.
	3a: Line 153: args = Args(all=False, input="Inputs_to_test_current_code", output_directory="output", params=None)
		all = False will plot the graph, and generate adjacency matrix for only 1 input file. 
		all = True, does the above work for all of them. 
		
	