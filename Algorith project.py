import heapq
import tkinter as tk
import time
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

graph = {
    'A': {'B': 3, 'C': 2},
    'B': {'D': 2},
    'C': {'B': -1, 'D': 1},
    'D': {}
}

def Mainbell():
    vertices = 5
    graph = []

    def addEdge(u, v, w):
        graph.append([u, v, w])

    def printArr(dist):
        print("Vertex\tDistance from Source")
        for i in range(vertices):
            print("{0}\t\t{1}".format(i, dist[i]))

    def BellmanFord(src):
        dist = [float("Inf")] * vertices
        dist[src] = 0

        for _ in range(vertices - 1):
            for u, v, w in graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        for u, v, w in graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                print("Graph contains a negative weight cycle")
                return

        printArr(dist)

    addEdge(0, 1, -1)
    addEdge(0, 2, 4)
    addEdge(1, 2, 3)
    addEdge(1, 3, 2)
    addEdge(1, 4, 2)
    addEdge(3, 2, 5)
    addEdge(3, 1, 1)
    addEdge(4, 3, -3)

    BellmanFord(0)


# Call the function
Mainbell()

def dijkstra(graph, source):
  
  distances = {vertex: float('inf') for vertex in graph}
  distances[source] = 0
  visited = set()

  priority_queue = []
  priority_queue.append((0, source))

  while priority_queue:
    
    current_distance, current_vertex = priority_queue.pop(0)
    
    if current_vertex in visited:
      continue

    visited.add(current_vertex)
    
    for neighbor in graph[current_vertex]:
      if neighbor not in visited:
        new_distance = current_distance + graph[current_vertex][neighbor]
        
        if new_distance < distances[neighbor]:
          distances[neighbor] = new_distance
          priority_queue.append((new_distance, neighbor))

  
  return distances


def BREADTH_FIRST():
    def bfs(graph, start):
        visited = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        return visited

    def calculate_complexity():
        times = []
        for n in range(1, 100):
            start_time = time.time()
            bfs(graph, 'A')
            times.append((time.time() - start_time) * (10 ** 9))
        f, ax = plt.subplots()
        ax.plot(range(1, 100), times, label='$n$')
        ax.set_xlabel('$n$')
        ax.set_ylabel('Time: nano seconds')
        ax.set_title('Complexity of BFS')
        ax.legend(loc=0)

        # Display the plot in a new window
        plt.show()
    
    def draw_graph():
        
        G = nx.Graph(graph)
        pos = nx.spring_layout(G)  # Determine node positions using a spring layout algorithm
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, edge_color='gray')
        plt.show()

    
    root = tk.Tk()
    root.geometry("700x700")
    root.title("BFS Algorithm")

    frame = tk.Frame(root)
    frame.pack(pady=200)
    bfs_output_text = tk.Text(frame, height=1, width=30)
    bfs_output_text.pack()

    def perform_bfs():
        bfs_output_text.delete('1.0', tk.END)  # Clear the text widget before performing BFS
        visited = bfs(graph, 'A')
        bfs_output_text.insert(tk.END, str(visited))

    bfs_button = tk.Button(frame, text="Perform BFS", height=2, width=15, command=perform_bfs)
    bfs_button.pack()

    complexity_button = tk.Button(frame, text='Calculate Complexity', command=calculate_complexity)
    complexity_button.pack()
    
    graph_button = tk.Button(frame, text='Show the Graph', command=draw_graph)
    graph_button.pack()

    root.mainloop()

def DEPTH_FIRST():
    def dfs(graph, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        dfs_output_text.insert(tk.END, start + " ")
        for neighbor in graph[start]:
            if neighbor not in visited:
                dfs(graph, neighbor, visited)

    def calculate_complexity():
        times = []
        for n in range(1, 100):
            start_time = time.time()
            dfs(graph, 'A')
            times.append((time.time() - start_time) * (10 ** 9))
        f, ax = plt.subplots()
        ax.plot(range(1, 100), times, label='$n$')
        ax.set_xlabel('$n$')
        ax.set_ylabel('Time: nano seconds')
        ax.set_title('Complexity of DFS')
        ax.legend(loc=0)

        # Display the plot in a new window
        plt.show()

    def draw_graph():
        
        G = nx.Graph(graph)
        pos = nx.spring_layout(G)  # Determine node positions using a spring layout algorithm
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, edge_color='gray')
        plt.show()

    root = tk.Tk()
    root.geometry("700x700")
    root.title("DFS Algorithm")

    frame = tk.Frame(root)
    frame.pack(pady=200)
    dfs_output_text = tk.Text(frame, height=1, width=30)
    dfs_output_text.pack()

    def perform_dfs():
        dfs_output_text.delete('1.0', tk.END)  # Clear the text widget before performing DFS
       
        dfs(graph, 'A')

    dfs_button = tk.Button(frame, text="Perform DFS", height=2, width=15, command=perform_dfs)
    dfs_button.pack()

    complexity_button = tk.Button(frame, text='Calculate Complexity', command=calculate_complexity)
    complexity_button.pack()

    graph_button = tk.Button(frame, text='Show the Graph', command=draw_graph)
    graph_button.pack()

    root.mainloop() 
    
    root = tk.Tk()
    root.geometry("700x700")
    root.title("DFS Algorithm")

    frame = tk.Frame(root)
    frame.pack(pady=200)
    dfs_output_text = tk.Text(frame, height=1, width=30)
    dfs_output_text.pack()

    def perform_dfs():
        dfs_output_text.delete('1.0', tk.END)  # Clear the text widget before performing DFS

        dfs(graph, 'A')

    dfs_button = tk.Button(frame, text="Perform DFS", height=2, width=15, command=perform_dfs)
    dfs_button.pack()

    complexity_button = tk.Button(frame, text='Calculate Complexity', command=calculate_complexity)
    complexity_button.pack()
    graph_button = tk.Button(frame, text='Show the Graph', command=draw_graph)
    graph_button.pack()

    root.mainloop()

def DIJKSTRA():
    def calculate_complexity():
        times = []
        for n in range(1, 100):
            start_time = time.time()
            for _ in range(n):
                dijkstra(graph, 'A')
            times.append((time.time() - start_time) * (10 ** 9))
        f, ax = plt.subplots()
        ax.plot(range(1, 100), times, label='$n$')
        ax.set_xlabel('$n$')
        ax.set_ylabel('Time: nano seconds')
        ax.set_title('Complexity of Dijkstra\'s Algorithm')
        ax.legend(loc=0)

        # Display the plot in a new window
        plt.show()
    
    def draw_graph():
        
        G = nx.Graph(graph)
        pos = nx.spring_layout(G)
        start_node = 'A'
        distances = dijkstra(graph,start_node )
        labels = {node: f"{node} ({distance})" for node, distance in distances.items()}

        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos, labels)
        plt.axis('off')
        plt.show()

    
    root = tk.Tk()
    root.geometry("700x700")
    root.title("Dijkstra's Algorithm")

    frame = tk.Frame(root)
    frame.pack(pady=200)
    dijkstra_output_text = tk.Text(frame, height=1, width=30)
    dijkstra_output_text.pack()

    def perform_dijkstra():
        dijkstra_output_text.delete('1.0', tk.END)  # Clear the text widget before performing Dijkstra's Algorithm
        distances = dijkstra(graph, 'A')
        dijkstra_output_text.insert(tk.END, str(distances))

    dijkstra_button = tk.Button(frame, text="Perform Dijkstra's", height=2, width=20, command=perform_dijkstra)
    dijkstra_button.pack()

    complexity_button = tk.Button(frame, text='Calculate Complexity', command=calculate_complexity)
    complexity_button.pack()
    graph_button = tk.Button(frame, text='Show the Graph', command=draw_graph)
    graph_button.pack()

    root.mainloop()

def BELLMAN_FORD():
    class Graph:
        def __init__(self, vertices):
            self.V = vertices
            self.graph = []

        def addEdge(self, src, dest, weight):
            self.graph.append((src, dest, weight))


    def draw_graph(graph):
        G = nx.DiGraph()

        for edge in graph:
            src, dest, weight = edge
            G.add_edge(src, dest, weight=weight)

        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'weight')

        nx.draw_networkx_nodes(G, pos, node_color='lightblue')
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.axis('off')
        plt.show()


    def Mainbell(vertices, graph):
        def BellmanFord(src):
            dist = [float("Inf")] * vertices
            dist[src] = 0

            for _ in range(vertices - 1):
                for u, v, w in graph:
                    if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w

            for u, v, w in graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    return "Graph contains a negative weight cycle"

            return dist

        distances = BellmanFord(0)
        return distances


    root = tk.Tk()
    root.geometry("800x800")
    root.title("Bellman-Ford Algorithm")

    frame = tk.Frame(root)
    frame.pack(pady=200)
    bellman_ford_output_text = tk.Text(frame, height=1, width=30)
    bellman_ford_output_text.pack()


    def perform_bellman_ford():
        bellman_ford_output_text.delete('1.0', tk.END)  # Clear the text widget before performing Bellman-Ford Algorithm
        vertices = 5
        g = Graph(vertices)
        g.addEdge(0, 1, -1)
        g.addEdge(0, 2, 4)
        g.addEdge(1, 2, 3)
        g.addEdge(1, 3, 2)
        g.addEdge(1, 4, 2)
        g.addEdge(3, 2, 5)
        g.addEdge(3, 1, 1)
        g.addEdge(4, 3, -3)
        distances = Mainbell(vertices, g.graph)
        bellman_ford_output_text.insert(tk.END, str(distances))


    bellman_ford_button = tk.Button(frame, text="Perform Bellman-Ford", height=2, width=20, command=perform_bellman_ford)
    bellman_ford_button.pack()


    def calculate_complexity():
        times = []
        vertices = 5
        g = Graph(vertices)
        g.addEdge(0, 1, -1)
        g.addEdge(0, 2, 4)
        g.addEdge(1, 2, 3)
        g.addEdge(1, 3, 2)
        g.addEdge(1, 4, 2)
        g.addEdge(3, 2, 5)
        g.addEdge(3, 1, 1)
        g.addEdge(4, 3, -3)
        for n in range(1, 100):
            start_time = time.time()
            for _ in range(n):
                Mainbell(vertices, g.graph)
            times.append((time.time() - start_time) * (10 ** 9))
        f, ax = plt.subplots()
        ax.plot(range(1, 100), times, label='$n$')
        ax.set_xlabel('$n$')
        ax.set_ylabel('Time: nano seconds')
        ax.set_title('Complexity of Bellman-Ford Algorithm')
        ax.legend(loc=0)

        # Display the plot in a new window
        plt.show()
    complexity_button = tk.Button(frame, text='Calculate Complexity', command=calculate_complexity)
    complexity_button.pack()    

    def show_graph():
        vertices = 5
        g = Graph(vertices)
        g.addEdge(0, 1, -1)
        g.addEdge(0, 2, 4)
        g.addEdge(1, 2, 3)
        g.addEdge(1, 3, 2)
        g.addEdge(1, 4, 2)
        g.addEdge(3, 2, 5)
        g.addEdge(3, 1, 1)
        g.addEdge(4, 3, -3)
        draw_graph(g.graph)


    graph_button = tk.Button(frame, text='Show the Graph', command=show_graph)
    graph_button.pack()

    root.mainloop()

import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

def A_star():
    graph = {
        'S': [('A', 1), ('B', 4)],
        'A': [('B', 2), ('C', 5), ('G', 12)],
        'B': [('C', 2)],
        'C': [('G', 3)]
    }

    Huristic_table = {
        'S': 7,
        'A': 6,
        'B': 4,
        'C': 2,
        'G': 0
    }

    def F_cost(path):
        g_cost = 0
        for (node, cost) in path:
            g_cost += cost
        last_node = path[-1][0]
        h_cost = Huristic_table[last_node]
        f_cost = g_cost + h_cost
        return f_cost, last_node

    def draw_graph(graph, path=None):
        G = nx.Graph()
        labels = {}

        for node, edges in graph.items():
            G.add_edges_from([(node, edge[0]) for edge in edges])
            labels.update({(node, edge[0]): str(edge[1]) for edge in edges})

        pos = nx.spring_layout(G)

        plt.figure()
        nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')

        # Highlight the path
        if path:
            path_edges = [(path[i][0], path[i + 1][0]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=2.0)

        plt.axis('off')
        plt.show()

    def star_search(graph, start, goal):
        visited = []
        queue = [[(start, 0)]]
        while queue:
            queue.sort(key=F_cost)
            path = queue.pop(0)
            node = path[-1][0]
            if node in visited:
                continue
            visited.append(node)
            if node == goal:
                return path
            else:
                adjacent_nodes = graph.get(node, [])
                for (node2, cost) in adjacent_nodes:
                    new_path = path.copy()
                    new_path.append((node2, cost))
                    queue.append(new_path)

    def calculate_complexity(graph, start, goal):
        start_time = time.time()
        result_path = star_search(graph, start, goal)
        end_time = time.time()
        complexity = end_time - start_time
        return result_path, complexity

    def show_result_path():
        result_path, _ = calculate_complexity(graph, 'S', 'G')
        result_text.delete(1.0, tk.END)  # Clear previous text
        result_text.insert(tk.END, "Path: {}\n".format(result_path))

    def show_complexity_graph():
        times = []
        for n in range(1, 100):
            start_time = time.time()
            star_search(graph, 'S', 'G')
            times.append((time.time() - start_time) * (10 ** 9))
        f, ax = plt.subplots()
        ax.plot(range(1, 100), times, label='$n$')
        ax.set_xlabel('$n$')
        ax.set_ylabel('Time: nano seconds')
        ax.set_title('Complexity of Star Search')
        ax.legend(loc=0)
        plt.show()

    # Create GUI window
    root = tk.Tk()
    root.title("Graph Visualization")
    root.geometry("800x600")

    # Text widget to display the output
    result_text = tk.Text(root, height=10, width=60)
    result_text.pack()

    # Button to show the result path
    result_path_button = tk.Button(root, text="Show Result Path", command=show_result_path)
    result_path_button.pack()

    # Button to show the complexity graph
    complexity_graph_button = tk.Button(root, text="Show Complexity Graph", command=show_complexity_graph)
    complexity_graph_button.pack()

    # Button to show the graph visualization
    graph_button = tk.Button(root, text="Show Graph", command=lambda: draw_graph(graph))
    graph_button.pack()

    # Start the GUI event loop
    tk.mainloop()

# Run the A_star function


def main_account_screen():
    root = tk.Tk()
    root.geometry("700x700")
    root.title("Graph Representation")
    frame = tk.Frame(root)
    frame.pack(pady=200)

    breadth_first_button = tk.Button(frame, text='Breadth First', command=BREADTH_FIRST)
    breadth_first_button.pack()

    depth_first_button = tk.Button(frame, text='Depth First', command=DEPTH_FIRST)
    depth_first_button.pack()

    dijkstra_button = tk.Button(frame, text='Dijkstra\'s Algorithm', command=DIJKSTRA)
    dijkstra_button.pack()

    bellman_ford_button = tk.Button(frame, text='Bellman-Ford Algorithm', command=BELLMAN_FORD)
    bellman_ford_button.pack()

    a_star_button = tk.Button(frame, text='A* Algorithm', command=A_star)
    a_star_button.pack()

    root.mainloop()
main_account_screen()
