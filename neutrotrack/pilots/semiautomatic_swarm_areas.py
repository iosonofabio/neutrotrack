# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/24
content:    Test script to guess the swarm coordinates and volume.
'''
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    print('Read in video data from file')
    fn = '../data/Neutrophil swarming project/example_imaris/140707_movie_2.ims'
    with h5py.File(fn) as h5:
        h5_movie = h5['DataSet']['ResolutionLevel 0']
        ntp = len(h5_movie.keys())
        video = np.zeros((ntp, 4, 64, 512, 512), dtype=np.uint8)
        for i in range(video.shape[0]):
            key = 'TimePoint ' + str(i)
            for chi in range(video.shape[1]):
                keych = 'Channel ' + str(chi)
                video[i, chi] = h5_movie[key][keych]['Data'][:]

from matplotlib.backend_bases import MouseButton

injury_coords = dict(x=150, y=250, z=20)
xc, yc = injury_coords['x'], injury_coords['y']
timeframes = [5, 8, 11, 15, 22, 28, 33, 40, 45, 50, 54, 59][::-1]
zframes = [20, 15, 10, 5, 25, 30, 35]
image_index = {'time': None, 'z': None}
radii = []
artists = {}

fig, ax = plt.subplots()

def on_click(event):
    if event.button is MouseButton.LEFT:
        if 'current' in artists:
            artists['current'].remove()

        x, y = event.xdata, event.ydata
        r = np.sqrt((x - xc)**2 + (y - yc)**2)
        patch = plt.Circle((xc, yc), r, lw=2, facecolor='none', edgecolor='red')
        artists['current'] = patch
        ax.add_patch(patch)
        plt.show()
    elif event.button is MouseButton.RIGHT:
        if 'current' not in artists and 'old' not in artists:
            radii.append({'time': image_index['time'], 'z': image_index['z'], 'r': None})
        elif 'current' not in artists:
            radii.append({'time': image_index['time'], 'z': image_index['z'], 'r': float(artists['old'].get_radius())})
        else:
            radii.append({'time': image_index['time'], 'z': image_index['z'], 'r': float(artists['current'].get_radius())})
            if 'old' in artists:
                artists['old'].remove()
            artists['old'] = artists.pop('current')
        switch_image()
binding_id = plt.connect('button_press_event', on_click)

def switch_image():
    if image_index['time'] is None:
        image_index['time'] = timeframes[0]
        image_index['z'] = zframes[0]
    elif (image_index['time'] == timeframes[-1]) and (image_index['z'] == zframes[-1]):
        plt.close(fig)
        return
    elif (image_index['time'] == timeframes[-1]):
        image_index['z'] = zframes[zframes.index(image_index['z']) + 1]
        image_index['time'] = timeframes[0]
    else:
        image_index['time'] = timeframes[timeframes.index(image_index['time']) + 1]

    time = image_index['time']
    z = image_index['z']
    it = timeframes.index(time)

    image_t = video[time, 2, z]
    ax.set_title(f'timepoint {time}, {it+1} of {len(timeframes)}, z {z}')
    ax.imshow(image_t, interpolation='nearest')
    if 'old' in artists:
        rold = artists['old'].get_radius()
        artists['old'].remove()
        patch = plt.Circle((xc, yc), rold, lw=2, facecolor='none', edgecolor='red', ls='--')
        artists['old'] = patch
        ax.add_patch(patch)
    plt.show()

switch_image()
plt.show()

radii = pd.DataFrame(radii)
rmatrix = radii.set_index(['time', 'z'])['r'].unstack('time')

fig, ax = plt.subplots()
ax.imshow(rmatrix, interpolation='nearest')
ax.set_ylabel('z plane')
plt.show()

from scipy.interpolate import PchipInterpolator
rmax = rmatrix.values.max()
fig = plt.figure(figsize=(16, 12))
cmap = plt.get_cmap('viridis')
for it, time in enumerate(timeframes):
    ax = fig.add_subplot(3, 4, len(timeframes) - it, projection='3d')
    ax.set_title(f"{time * 45.0 / 60} mins")
    for iz, z in enumerate(zframes):
        radius = rmatrix.at[z, time]
        theta = np.linspace(0, 2 * np.pi, 100)
        x = injury_coords['x'] + radius * np.cos(theta)
        y = injury_coords['y'] + radius * np.sin(theta)
        ax.plot(x, y, zs=z, color=cmap(1.0 * iz / len(zframes)))

    tmp = rmatrix.loc[:, time].sort_index()
    radiuses = tmp.values
    zs = tmp.index.values
    zs_int = np.linspace(zs[0], zs[-1], 200)
    radiuses_int = PchipInterpolator(zs, radiuses)(zs_int)
    zs_int = np.append(zs_int, zs_int[-1])
    radiuses_int = np.append(radiuses_int,  0)
    for theta in np.arange(-6, 2) * 2 *np.pi / 15:
        xs_int = injury_coords['x'] + radiuses_int * np.cos(theta)
        ys_int = injury_coords['y'] + radiuses_int * np.sin(theta)
        ax.plot(xs_int, ys_int, zs=zs_int, color='grey', alpha=0.7)

for ax in fig.get_axes():
    ax.set_xlim(injury_coords['x'] - rmax * 1.1, injury_coords['x'] + rmax * 1.1)
    ax.set_ylim(injury_coords['y'] - rmax * 1.1, injury_coords['y'] + rmax * 1.1)
fig.tight_layout(pad=0)
plt.show()

vols = []
for it, time in enumerate(timeframes):
    tmp = rmatrix.loc[:, time].sort_index()
    radiuses = tmp.values
    zs = tmp.index.values
    zs_int = np.linspace(zs[0], zs[-1], 200)
    radiuses_int = PchipInterpolator(zs, radiuses)(zs_int)
    zs_int = np.append(zs_int, zs_int[-1])
    radiuses_int = np.append(radiuses_int,  0)
    vol = np.pi * (radiuses_int**2)[:-1] @ np.diff(zs_int)
    vols.append({'time': time, 'volume': vol})
vols = pd.DataFrame(vols).set_index('time').sort_index()['volume']

from scipy.stats import linregress
fit = linregress(vols.index * 45.0 / 60, vols.values / 1000.0)
m = fit.slope
q = fit.intercept
xx = np.linspace(*((vols.index * 45.0 / 60)[[0, -1]]), 100)
yy = q + m * xx

fig, ax = plt.subplots()
ax.plot(vols.index * 45.0 / 60, vols.values / 1000.0)
ax.plot(xx, yy, color='grey', label='{:.1f} neutrophils / min'.format(m))
ax.set_xlabel('time [mins]')
ax.set_ylabel('swarm volume\n[kilovoxels ~ # neutrophils]')
ax.grid(True)
ax.legend()
fig.tight_layout()
plt.show()

