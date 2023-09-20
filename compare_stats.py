"""Compares cell detection stats from two different csv files."""
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--csv1', type=str, required=True, help='Path to first csv file.')
    parser.add_argument('--csv2', type=str, required=True, help='Path to second csv file.')
    parser.add_argument('--n_types', type=int, default=4, help='Number of cell types.')
    parser.add_argument('--save_dir', type=str, help='Optional path to save figures in.')

    args = parser.parse_args()
    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    # each row of df is stats of cell detections from a single patch
    # columns are cell type counts, n_cells, and area_hist in some number of bins
    # make plots to illustrate differences in distributions of these stats

    # cell type counts
    # get the mean and std of each cell type count across patches
    n_types = args.n_types
    df1_type_counts = df1.iloc[:, :n_types]
    df2_type_counts = df2.iloc[:, :n_types]
    df1_type_counts_mean = df1_type_counts.mean(axis=0)
    df1_type_counts_std = df1_type_counts.std(axis=0)
    df2_type_counts_mean = df2_type_counts.mean(axis=0)
    df2_type_counts_std = df2_type_counts.std(axis=0)

    # plot the mean and std of each cell type count
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(df1_type_counts_mean)), df1_type_counts_mean, yerr=df1_type_counts_std, label='csv1')
    ax.bar(np.arange(len(df2_type_counts_mean)), df2_type_counts_mean, yerr=df2_type_counts_std, label='csv2')
    ax.set_xticks(np.arange(len(df1_type_counts_mean)))
    ax.set_xticklabels(df1_type_counts.columns)
    ax.set_ylabel('Mean cell type count')
    ax.set_xlabel('Cell type')
    ax.legend()
    plt.show()

    # n_cells
    # get the mean and std of n_cells across patches
    df1_n_cells = df1.iloc[:, n_types]
    df2_n_cells = df2.iloc[:, n_types]
    df1_n_cells_mean = df1_n_cells.mean(axis=0)
    df1_n_cells_std = df1_n_cells.std(axis=0)
    df2_n_cells_mean = df2_n_cells.mean(axis=0)
    df2_n_cells_std = df2_n_cells.std(axis=0)

    # plot the mean and std of n_cells
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(df1_n_cells_mean)), df1_n_cells_mean, yerr=df1_n_cells_std, label='csv1')
    ax.bar(np.arange(len(df2_n_cells_mean)), df2_n_cells_mean, yerr=df2_n_cells_std, label='csv2')
    ax.set_xticks(np.arange(len(df1_n_cells_mean)))
    ax.set_xticklabels(['n_cells'])
    ax.set_ylabel('Mean n_cells')
    ax.set_xlabel('Cell type')
    ax.legend()
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
