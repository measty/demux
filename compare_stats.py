"""Compares cell detection stats from two different csv files."""
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--csv1', type=str, help='Path to first csv file.', default=r"E:\PRISMATIC\demux_restained\81RKRK_HE_cell_stats.csv") 
    parser.add_argument('--csv2', type=str, help='Path to second csv file.', default=r"E:\PRISMATIC\demux_restained\81RKRK_PHH3_HE_restained_HE_cell_stats.csv")
    parser.add_argument('--n_types', type=int, default=5, help='Number of cell types.')
    parser.add_argument('--save_dir', type=str, help='Optional path to save figures in.')

    args = parser.parse_args()
    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)
    # drop 'Unnamed: 0' column
    df1 = df1.drop(columns=['Unnamed: 0'])
    df2 = df2.drop(columns=['Unnamed: 0'])

    # each row of df is stats of cell detections from a single patch
    # columns are cell type counts, n_cells, and area_hist in some number of bins
    # make plots to illustrate differences in distributions of these stats

    # cell type counts
    # plot box and whisker plots of cell type counts with shared y axis
    n_types = args.n_types
    df1_type_counts = df1.iloc[:, :n_types]
    df2_type_counts = df2.iloc[:, :n_types]
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].boxplot(df1_type_counts)
    ax[0].set_ylabel('Number of cells')
    ax[0].set_title('Real HE')
    ax[1].boxplot(df2_type_counts)
    ax[1].set_ylabel('Number of cells')
    ax[1].set_title('Restained HE')
    plt.show()


    

    # box and whisker plonts on cell nums
    df1_n_cells = df1.iloc[:, n_types]
    df2_n_cells = df2.iloc[:, n_types]
    fig, ax = plt.subplots(1, 2)
    ax[0].boxplot(df1_n_cells)
    ax[0].set_ylabel('Number of cells')
    ax[1].boxplot(df2_n_cells)
    ax[1].set_ylabel('Number of cells')
    plt.show()

    # area_hist - how should we illustrate diffences in distributions?
    # plot the mean histograms of area_hist across patches side by side
    df1_area_hist = df1.iloc[:, n_types+1:]
    df2_area_hist = df2.iloc[:, n_types+1:]
    df1_area_hist_mean = df1_area_hist.mean(axis=0)
    df2_area_hist_mean = df2_area_hist.mean(axis=0)
    fig, ax = plt.subplots(1, 2)
    ax[0].bar(np.arange(len(df1_area_hist_mean)), df1_area_hist_mean)
    ax[0].set_xticks(np.arange(len(df1_area_hist_mean)))
    ax[0].set_xticklabels(df1_area_hist.columns)
    ax[0].set_ylabel('Mean area_hist')
    ax[0].set_xlabel('Area bin')
    ax[1].bar(np.arange(len(df2_area_hist_mean)), df2_area_hist_mean)
    ax[1].set_xticks(np.arange(len(df2_area_hist_mean)))
    ax[1].set_xticklabels(df2_area_hist.columns)
    ax[1].set_ylabel('Mean area_hist')
    ax[1].set_xlabel('Area bin')
    plt.show()
