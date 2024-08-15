import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



fdn_movies = pathlib.Path('../..') / 'data' / 'Neutrophil swarming project' / 'Example swarming and non-swarming neutrophil data' / 'single swarm data' 
fn_track_movie1 = fdn_movies / '140714_movie1' / '140714_movie 1 after Drift Cor Tracks.xls'
fn_track_movie2 = fdn_movies / '140707_movie 2' / '140707_movie 2 after Drfit Corr Tracks.xls'
fn_track_movie3 = fdn_movies / '140707_movie 3' / 'movie 3 after drift corrTracks.xls'


if __name__ == '__main__':


    ds2 = pd.read_excel(fn_track_movie3, sheet_name='Displacement^2')
    ds2_by_cell = ds2.groupby('Parent')


    colors = sns.color_palette('viridis', n_colors=ds2_by_cell.ngroups)
    fig, ax = plt.subplots()
    for i, (cellid, ds2_cell) in enumerate(ds2_by_cell):
        color = colors[i]
        x = ds2_cell['Time']
        y = ds2_cell['Value']

        ax.plot(x, y, color=color, label=cellid)
    ax.set_xlabel('Time [step ?]')
    ax.set_ylabel('$\Delta^2 [\\mu m^2]$')
    ax.grid(True)
    fig.tight_layout()
    plt.ion()
    plt.show()
