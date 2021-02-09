#Project 1
#Kyle Rood

#How to run:
#python project1_rood.py <filename with graph>
# ex) python project1_rood.py test_graph.txt

import pandas as pd
import heapq
from collections import deque
import sys
import os
import time

#Graph object to contain different functions within it
class Graph:
    def __init__(self, graph, vertices, edges):
        self.graph = graph
        self.vertices = vertices
        self.edges = edges

    def dijkstra_shortest_path(self, starting_vertex, destination_vertex):
        #set all initial distances as inifinity (this finds shortest path to all nodes)
        distances = {vertex: float('infinity') for vertex in self.graph}
        distances[starting_vertex] = 0

        #priority queue used to iterate over all of the vertexs
        pq = [(0, starting_vertex)]
        while len(pq) > 0:
            #remove vertex and weight from pqueue --> only want each one to be start once
            current_distance, current_vertex = heapq.heappop(pq)

            #go to next iteration if not less than current distance to node
            if current_distance > distances[current_vertex]:
                continue

            #loop over each neighbor, looking for the minimum distance
            for neighbor, weight in self.graph[current_vertex]:
                distance = current_distance + weight

                # Only consider this new path if it's shorter
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        return distances[destination_vertex]

    # heuristic function --> using the squares as our conversion (stored in self.vertices)
    def heuristic(self, node):
        try:
            return self.vertices[node]
        except:
            return 1

    def get_weight(self, node_list, node):
        for node_id, node_weight in node_list:
            if(node_id == node):
                return node_weight

    def a_star_algorithm(self, start_node, stop_node):
        # visited_not_inspected -->list of nodes which have been visited, but who's neighbors haven't all been inspected
        # visited_inspected --> list of nodes which have been visited and who's neighbors have been inspected
        visited_not_inspected = set([start_node])
        visited_inspected = set([])

        # initialize graph
        graph = {}
        graph[start_node] = 0

        #initialize adjacency map of all nodes
        adjacency_map = {}
        adjacency_map[start_node] = start_node

        #loop while we still have nodes that have uninspected neighbors 
        while(len(visited_not_inspected) > 0):
            node = None

            # find a node with the lowest value for evaluation function
            for vert in visited_not_inspected:
                if(node == None or graph[vert] + self.heuristic(vert) < graph[node] + self.heuristic(node)):
                    node = vert

            if(node == None):
                print('No path exists...')
                return None

            # if the current node is the stop_node reconstruct path from stop_node to the start_node
            if(node == stop_node):
                reconst_path = []

                while(adjacency_map[node] != node):
                    reconst_path.append(node)
                    node = adjacency_map[node]

                reconst_path.append(start_node)
                reconst_path.reverse()

                #get weight of path found
                weight = 0
                for i in range(0, len(reconst_path)-1):
                    weight += self.get_weight(self.graph[reconst_path[i]], reconst_path[i+1])

                return weight

            #looping 
            neighbors = self.graph[node]
            for (neighbor_id, weight) in neighbors:
                # if the current node isn't in both visited_not_inspected and visited_inspected
                # add it to visited_not_inspected and note n as it's parent
                if(neighbor_id not in visited_not_inspected and neighbor_id not in visited_inspected):
                    visited_not_inspected.add(neighbor_id)
                    adjacency_map[neighbor_id] = node
                    graph[neighbor_id] = graph[node] + weight

                # otherwise, check if it's quicker to first visit n, then m
                else:
                    # update parent data and g data
                    if(graph[neighbor_id] > graph[node] + weight):
                        graph[neighbor_id] = graph[node] + weight
                        adjacency_map[neighbor_id] = node

                        # if node is in visited_inspected, move it to visited_not_inspected
                        if(neighbor_id in visited_inspected):
                            visited_inspected.remove(neighbor_id)
                            visited_not_inspected.add(neighbor_id)

            # remove node from visited_not_inspected and add to visited_inspected --> all neighbors visited
            visited_not_inspected.remove(node)
            visited_inspected.add(node)

        print('No path exists...')
        return None


def parse_file(filename):
    #load in data and parse
    data = ""
    with open(filename, 'r') as file:
        data = file.read()

    lines = data.split("\n")
    collect = 0 #0 = no, 1 = vert, 2 = edges, 3 = start/end

    #dataframes for different pieces
    vertex_cols = ["vertex_id", "square_id"]
    vertices = pd.DataFrame(index = [0], columns = vertex_cols)

    edge_cols = ["from", "to", "distance"]
    edges = pd.DataFrame(index = [0], columns = edge_cols)

    graph = {}

    #default values
    start_node = "0"
    dest_node = "99"

    for line in lines:
        if(line == ""):
            collect = 0

        #means we start collecting vert
        if(collect == 1):
            split_line = line.split(",")
            new_entry = {
                "vertex_id": split_line[0], 
                "square_id": split_line[1]
            }
            vertices = vertices.append(new_entry, ignore_index=True)

        #edges
        elif(collect == 2):
            split_line = line.split(",")
            new_entry = {
                "from": split_line[0], 
                "to": split_line[1],
                "distance": split_line[2]
            }
            edges = edges.append(new_entry, ignore_index=True)

            #need both combinations (both directions)
            try:
                graph[split_line[0]].append((split_line[1], int(split_line[2])))
            except:
                graph[split_line[0]] = [(split_line[1], int(split_line[2]))]

            try:
                graph[split_line[1]].append((split_line[0], int(split_line[2])))
            except:
                graph[split_line[1]] = [(split_line[0], int(split_line[2]))]

        elif(collect == 3):
            split_line = line.split(",")
            if(split_line[0] == "S"):
                start_node = split_line[1]
            elif(split_line[0] == "D"):
                dest_node = split_line[1]

        if(line == "# Vertex ID, Square ID"):
            collect = 1
        elif(line == "# From, To, Distance"):
            collect = 2
        elif(line == "# Source and Dest"):
            collect = 3

    #Drop nans
    vertices.dropna(inplace=True)
    edges.dropna(inplace=True)
    return graph, vertices, edges, start_node, dest_node


#RUNNING TESTS#
#load in data using argument
try:
    input_filename = str(sys.argv[1])
except:
    input_filename = "./test_graph.txt"

raw_graph, vert_dict, edge_dict, start, dest = parse_file(input_filename)
graph = Graph(raw_graph, vert_dict, edge_dict)

start_time = time.time()
print("Uninformed search result (Dijkstra's Shortest Path): ")
print(graph.dijkstra_shortest_path(start, dest))
end_time = time.time()
print("Time to execute Dijkstra: " + str(end_time - start_time))
print("")

start_time = time.time()
print("Informed search result (A* Search Algorithm): ")
print(graph.a_star_algorithm(start, dest))
end_time = time.time()
print("Time to execute A*: " +  str(end_time - start_time))
