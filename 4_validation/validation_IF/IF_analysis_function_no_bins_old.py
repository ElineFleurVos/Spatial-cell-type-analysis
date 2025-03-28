import pandas as pd 
from validate_functions import create_value_grid
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

def IF_scoring(predictions, cell_type, marker, tile_size):

    single_cell_dir = "/home/evos/Data/CRC/Orion/Orion_single_cell_tables"
    single_cell_files = sorted(os.listdir(single_cell_dir))
    single_cell_metadata_dir = "/home/evos/extra_code/IF_analysis/single_cell_metadata.txt"
    single_cell_metadata = pd.read_csv(single_cell_metadata_dir, sep="\t")

    # Preload metadata into a dictionary for fast lookups
    metadata_dict = dict(zip(single_cell_metadata['single_cell_file'], single_cell_metadata['slide_id']))

    # Initialize lists for fluorescence and probability values
    fluorescence_values = []
    probability_values = []

    all_scores = {}

    for file in single_cell_files:

        # Read the single cell data
        single_cell_data = pd.read_csv(f"{single_cell_dir}/{file}", sep=",", index_col=0)

        # Convert columns of interest to numeric, setting invalid values to NaN
        single_cell_data[marker] = pd.to_numeric(single_cell_data[marker], errors='coerce')
        single_cell_data['X_centroid'] = pd.to_numeric(single_cell_data['X_centroid'], errors='coerce')
        single_cell_data['Y_centroid'] = pd.to_numeric(single_cell_data['Y_centroid'], errors='coerce')

        # Drop rows with NaN in the required columns
        single_cell_data = single_cell_data.dropna(subset=[marker, 'X_centroid', 'Y_centroid'])

        # Retrieve the slide ID
        slide_id = metadata_dict[file]

        # Filter predictions for the current slide
        predictions_slide = predictions[predictions.slide_id == slide_id]
        predictions_slide = predictions_slide.reset_index(drop=True)

        # Create the probability grid
        prob_grid = create_value_grid(predictions_slide, cell_type)

        # Adjust centroids for scaling factor
        scaling_factor = 0.325 / 0.5  # Resolution adjustment
        single_cell_data['X_centroid_adjusted'] = single_cell_data['X_centroid'] * scaling_factor
        single_cell_data['Y_centroid_adjusted'] = single_cell_data['Y_centroid'] * scaling_factor

        # Determine grid shapes
        grid_shape_y = prob_grid.shape[0]
        grid_shape_x = prob_grid.shape[1]

        # Initialize grids
        average_fluorescence_grid = np.zeros((grid_shape_y, grid_shape_x))
        cell_count_grid = np.zeros((grid_shape_y, grid_shape_x))

        # Vectorized computation for grid updates
        single_cell_data['X_tile_index'] = (single_cell_data['X_centroid_adjusted'] // tile_size).astype(int)
        single_cell_data['Y_tile_index'] = (single_cell_data['Y_centroid_adjusted'] // tile_size).astype(int)

        # Filter valid cells within grid bounds
        valid_cells = single_cell_data[
            (single_cell_data['X_tile_index'] < grid_shape_x) &
            (single_cell_data['Y_tile_index'] < grid_shape_y)
        ]

        # Convert to NumPy arrays for efficient updates
        x_indices = valid_cells['X_tile_index'].to_numpy()
        y_indices = valid_cells['Y_tile_index'].to_numpy()
        fluorescence = valid_cells[marker].to_numpy()

        # Combine indices for bincount
        flat_indices = y_indices * grid_shape_x + x_indices
        # Sum fluorescence and count cells per tile
        tile_sums = np.bincount(flat_indices, weights=fluorescence, minlength=grid_shape_x * grid_shape_y)
        tile_counts = np.bincount(flat_indices, minlength=grid_shape_x * grid_shape_y)
        # Reshape back to grid dimensions
        average_fluorescence_grid = tile_sums.reshape(grid_shape_y, grid_shape_x)
        cell_count_grid = tile_counts.reshape(grid_shape_y, grid_shape_x)
        # Avoid divide-by-zero
        average_fluorescence_grid[cell_count_grid > 0] /= cell_count_grid[cell_count_grid > 0]

        # # Update grids
        # np.add.at(average_fluorescence_grid, (y_indices, x_indices), fluorescence)
        # np.add.at(cell_count_grid, (y_indices, x_indices), 1)
        # # Calculate average fluorescence per tile
        # average_fluorescence_grid[cell_count_grid > 0] /= cell_count_grid[cell_count_grid > 0]

        # Append current file's data to the combined lists (store intermediate arrays)
        fluorescence_values.append(average_fluorescence_grid.flatten())
        probability_values.append(prob_grid.values.flatten())

        # Create scatter data for correlation
        scatter_data = pd.DataFrame({
            'Fluorescence': average_fluorescence_grid.flatten(),
            'Probability': prob_grid.values.flatten()
        })
        scatter_data = scatter_data.dropna()

        # Calculate Spearman correlation and p-value
        cor_spearman, p_value_corr_spearman = spearmanr(scatter_data['Fluorescence'], scatter_data['Probability'])
        cor_pearson, p_value_corr_pearson = pearsonr(scatter_data['Fluorescence'], scatter_data['Probability'])

        # Store results for the current slide
        all_scores[slide_id] = [cor_spearman, p_value_corr_spearman, cor_pearson, p_value_corr_pearson]

    # Combine all fluorescence and probability values after the loop
    all_fluorescence_values = np.concatenate(fluorescence_values)
    all_probability_values = np.concatenate(probability_values)

    print('Finished IF score analysis')

    return all_scores, all_fluorescence_values, all_probability_values


