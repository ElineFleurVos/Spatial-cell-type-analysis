import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_node_positions(slide_data):
    """Get node (i.e. tile) positions according to original histopathological slide"""
    return slide_data[["Coord_X", "Coord_Y"]].apply(
        lambda coord: tuple([coord[0], coord[1]]), axis=1
    )


# Create coordinate system and add tiles (created index)
def create_grid_old(slide_data):
    """Create a grid (i.e. a large matrix) and assign the tiles to the grid
    Because the next step is to look at the neighbors, add padding, i.e. add additional columns to the borders of the matrix filled with NaNs

    Args:
        slide_data (DataFrame)

    Returns:
        node_grid (array)
    """

    coord_Y = slide_data["Coord_Y"].unique()
    coord_X = slide_data["Coord_X"].unique()
    coord_Y.sort()
    coord_X.sort()
    node_grid = pd.DataFrame(columns=coord_X, index=coord_Y)

    # Assign the tiles (nodes) by using the index of the node to the grid
    for x in coord_X:
        for y in coord_Y:
            mask = (slide_data["Coord_X"] == x) & (slide_data["Coord_Y"] == y)
            if mask.sum() > 0:
                node_grid.loc[y, x] = slide_data[mask].index.values[0]
    node_grid = node_grid.to_numpy()

    # add padding (row/col) for dealing with borders
    node_grid = np.concatenate(
        [np.full((node_grid.shape[0], 1), np.nan), node_grid], axis=1
    )
    node_grid = np.concatenate(
        [node_grid, np.full((node_grid.shape[0], 1), np.nan)], axis=1
    )

    node_grid = np.concatenate(
        [node_grid, np.full((1, node_grid.shape[1]), np.nan)], axis=0
    )
    node_grid = np.concatenate(
        [np.full((1, node_grid.shape[1]), np.nan), node_grid], axis=0
    )

    return node_grid

# Create coordinate system and add tiles (created index)
def create_grid(slide_data):
    """Create a grid (i.e. a large matrix) and assign the tiles to the grid
    Because the next step is to look at the neighbors, add padding, i.e. add additional columns to the borders of the matrix filled with NaNs

    Args:
        slide_data (DataFrame)

    Returns:
        node_grid (array)
    """
    #get step size by finding the minimum distance between Y-coordinates. Step size is 
    #dependent on the magnitude of the image. 
    coord_Y = slide_data["Coord_Y"].unique()
    coord_X = slide_data["Coord_X"].unique()
    coord_Y.sort()
    coord_X.sort()

    step = float('inf') 
    # Iterate through the array, comparing adjacent elements
    for i in range(len(coord_Y) - 1):
        diff = coord_Y[i+1] - coord_Y[i]
        if diff < step:
            step = diff

    max_coord_Y = slide_data['Coord_Y'].max()
    min_coord_Y = slide_data['Coord_Y'].min()
    max_coord_X = slide_data['Coord_X'].max()
    min_coord_X = slide_data['Coord_X'].min()
    coord_X_all = np.arange(min_coord_X, max_coord_X + step, step)
    coord_Y_all = np.arange(min_coord_Y, max_coord_Y + step, step)
    coord_Y_all.sort()
    coord_X_all.sort()
    node_grid = pd.DataFrame(columns=coord_X_all, index=coord_Y_all)

    # Assign the tiles (nodes) by using the index of the node to the grid
    for x in coord_X_all:
        for y in coord_Y_all:
            mask = (slide_data["Coord_X"] == x) & (slide_data["Coord_Y"] == y)
            if mask.sum() > 0:
                node_grid.loc[y, x] = slide_data[mask].index.values[0]
    node_grid = node_grid.to_numpy()

    # add padding (row/col) for dealing with borders
    node_grid = np.concatenate(
        [np.full((node_grid.shape[0], 1), np.nan), node_grid], axis=1
    )
    node_grid = np.concatenate(
        [node_grid, np.full((node_grid.shape[0], 1), np.nan)], axis=1
    )

    node_grid = np.concatenate(
        [node_grid, np.full((1, node_grid.shape[1]), np.nan)], axis=0
    )
    node_grid = np.concatenate(
        [np.full((1, node_grid.shape[1]), np.nan), node_grid], axis=0
    )

    return node_grid

def get_edges(slide_data):
    """ Determine the edges by looking at the neighbors
    Neighbors: adjacent nodes (top, right, bottom, left) + diagonal nodes (top-right, bottom-right etc.) = 8 neighbors (or nodes)

    Args:
        slide_data (DataFrame)

    Returns:
        all_edges (list): list of tuples containing the edges for constructing a graph
    """
    node_grid = create_grid(slide_data)
    all_edges = []

    coord_X = node_grid.shape[1]
    coord_Y = node_grid.shape[0]
    for i in range(coord_Y):
        for j in range(coord_X):
            selected_tile = node_grid[i, j]

            if ~np.isnan(selected_tile):

                # 1. Direct neighbors
                top_neighbor = node_grid[i - 1, j]
                left_neighbor = node_grid[i, j - 1]
                right_neighbor = node_grid[i, j + 1]
                bottom_neighbor = node_grid[i + 1, j]
                # Collect neighbors
                neighbors = [
                    top_neighbor,
                    left_neighbor,
                    right_neighbor,
                    bottom_neighbor,
                ]

                # 2. Diagonal neighbors
                top_left_neighbor = node_grid[i - 1, j - 1]
                top_right_neighbor = node_grid[i - 1, j + 1]
                bottom_left_neighbor = node_grid[i + 1, j - 1]
                bottom_right_neighbor = node_grid[i + 1, j + 1]
                # Collect neighbors
                neighbors += [
                    top_left_neighbor,
                    top_right_neighbor,
                    bottom_left_neighbor,
                    bottom_right_neighbor,
                ]

                # Remove non-tiles (border/padding, gaps/holes in image)
                neighbors = [x for x in neighbors if ~np.isnan(x)]

                all_edges += list(zip([selected_tile] * len(neighbors), neighbors))

    # Remove duplicates: edge (1,2) == edge (2,1)
    all_edges = {tuple(sorted(edge)) for edge in all_edges}

    return all_edges

def construct_graph(predictions, slide_submitter_id=None, draw_graph=False):
    """
    Construct a graph using NetworkX.

    Args:
        predictions (DataFrame)
        slide_submitter_id (str): optional
        draw_graph (bool): optional

    Returns:
        NetworkX graph (within a dictionary)
    """
    if len(predictions.slide_submitter_id.unique()) > 1:
        if slide_submitter_id is None:
            raise Exception("If slide_submitter_id is not specified then predictions should contain only data for one slide")

        else:
            predictions = predictions[predictions.slide_submitter_id == slide_submitter_id]
    slide_data = predictions.reset_index(drop=True)
    
    nodes = slide_data.index
    all_edges = get_edges(slide_data)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(all_edges)

    if draw_graph:
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_size=50, node_color='red', font_size=8)
        
    if slide_submitter_id is None:
        return G
    else:
        return {slide_submitter_id: G}
