# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/24
content:    Test script to guess the swarm coordinates and volume.
'''
import h5py
import numpy as np


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


    print('Plot one time point across z-slices')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(5, 7, figsize=(18, 12), sharex=True, sharey=True)
    axs = axs.ravel()
    i = 3
    injury_coords = dict(x=150, y=250)
    zlims = [4, video.shape[2] - 22]
    for ax, z in zip(axs, range(4, video.shape[2] - 22, 1)):
        time = ((i*6)*45)/60
        ax.imshow(video[i * 6, 2, z], interpolation='nearest')
        ax.set_title(f"z={z}, {time} minutes")         
        ax.scatter([injury_coords['x']], [injury_coords['y']], s=100, marker='+', color='red')
    fig.tight_layout(pad=0)
    plt.ion(); plt.show()

    print('Fit the swarm for each z slice')
    from scipy.optimize import minimize
    injury_coords = dict(x=150, y=250)
    xm, ym = np.meshgrid(np.arange(512), np.arange(512))
    tmpx = (xm - injury_coords['x'])**2
    tmpy = (ym - injury_coords['y'])**2
    def f_min(radii, image):
        radius = radii[0]
        # minimize negative average intensity inside ellipse 
        tmp = tmpx + tmpy < radius**2
        loss = -(image[tmp].mean())
        return loss
    
    zlims = [4, video.shape[2] - 22]
    radii = []
    for z in range(*zlims):
        image_z = video[i * 6, 2, z]
        radius = minimize(f_min, x0=(40,), args=(image_z,), bounds=[(0, 500)])
        radii.append(radius)
        break

    fig, axs = plt.subplots(5, 7, figsize=(18, 12), sharex=True, sharey=True)
    axs = axs.ravel()
    for ax, z, radius in zip(axs, range(*zlims), radii):
        time = ((i*6)*45)/60
        ax.imshow(video[i * 6, 2, z], interpolation='nearest')
        ax.set_title(f"z={z}, {time} minutes")         
        ax.scatter([injury_coords['x']], [injury_coords['y']], s=100, marker='+', color='red')
        ax.add_patch(plt.Circle(
            (injury_coords['x'], injury_coords['y']), radius,
            ec='red', lw=2, fc='none',
        ))
    fig.tight_layout(pad=0)

    i = 4
    zlims = [1, 36]
    rmin = 15
    avgs = []
    for z in range(*zlims):
        image_z = video[i * 6, 2, z]
        avgz = []
        for r in range(rmin, 75):
            res = image_z[tmpx + tmpy < r**2].mean()
            res -= image_z[(tmpx + tmpy >= r**2) & (tmpx + tmpy < (r + 30)**2)].mean()
            avgz.append(res)
        avgs.append(avgz)
    avgs = np.array(avgs)

    colors = plt.get_cmap('viridis')
    peaks = []
    for iz, avgz in enumerate(avgs):
        radius = (avgz).argmax() + rmin
        peaks.append([radius, avgz[radius - rmin]])
    peaks = np.array(peaks).T

    fig, axs = plt.subplots(1, 2)
    ax = axs[0]
    for iz, avgz in enumerate(avgs):
        ax.plot(np.arange(rmin, rmin + len(avgz)), avgz, color=colors(1.0 * iz / len(avgs)))
    ax.plot(peaks[0], peaks[1], color='black')
    axs[1].plot(np.arange(*zlims), peaks[0])
    axs[1].set_xlabel('z plane')
    axs[1].set_ylabel('radius')
    fig.tight_layout()
    plt.ion(); plt.show()

    fig, axs = plt.subplots(5, 7, figsize=(18, 12), sharex=True, sharey=True)
    axs = axs.ravel()
    for ax, z, radius in zip(axs, range(*zlims), peaks[0]):
        time = ((i*6)*45)/60
        ax.imshow(video[i * 6, 2, z], interpolation='nearest')
        ax.set_title(f"z={z}, {time} minutes")         
        ax.scatter([injury_coords['x']], [injury_coords['y']], s=100, marker='+', color='red')
        ax.add_patch(plt.Circle(
            (injury_coords['x'], injury_coords['y']), radius,
            ec='red', lw=2, fc='none',
        ))
    fig.tight_layout(pad=0)

    print('Plot 3d sphere')
    rmin = 15
    zlims = [1, 38]
    fig = plt.figure()
    timeframes = [5, 8, 11, 15, 22, 30, 40, 50, 59]
    rmax = 0
    for ii, i in enumerate(timeframes):
        avgs = []
        for z in range(*zlims):
            image_z = video[i, 2, z]
            avgz = []
            for r in range(rmin, 75):
                res = image_z[tmpx + tmpy < r**2].mean()
                res -= image_z[(tmpx + tmpy >= r**2) & (tmpx + tmpy < (r + 30)**2)].mean()
                avgz.append(res)
            avgs.append(avgz)
        avgs = np.array(avgs)

        peaks = []
        for iz, avgz in enumerate(avgs):
            radius = (avgz).argmax() + rmin
            peaks.append([radius, avgz[radius - rmin]])
        peaks = np.array(peaks).T
        rmax = max(rmax, peaks[0].max())

        ax = fig.add_subplot(1, len(timeframes), 1 + ii, projection='3d')
        time = (i*45.0)/60
        ax.set_title(f"{time} mins")
        cmap = plt.get_cmap('viridis')
        for iz, (z, radius) in enumerate(zip(range(*zlims), peaks[0])):  
            theta = np.linspace(0, 2 * np.pi, 100)
            x = injury_coords['x'] + radius * np.cos(theta)
            y = injury_coords['y'] + radius * np.sin(theta)
            ax.plot(x, y, zs=z, color=cmap(1.0 * iz / len(peaks[0])))
    for ax in fig.get_axes():
        ax.set_xlim(injury_coords['x'] - rmax * 1.1, injury_coords['x'] + rmax * 1.1)
        ax.set_ylim(injury_coords['y'] - rmax * 1.1, injury_coords['y'] + rmax * 1.1)
    fig.tight_layout()
    plt.ion(); plt.show()

    
    print('Fit and plot ellipsoid')
    injury_coords = dict(x=150, y=250, z=20)
    zm, xm, ym = np.meshgrid(np.arange(64), np.arange(512), np.arange(512), indexing='ij')
    tmpz = (zm - injury_coords['z'])**2
    tmpx = (xm - injury_coords['x'])**2
    tmpy = (ym - injury_coords['y'])**2

    def lossfun(r, rz, it):
        image_t = video[it, 2]
        loss = image_t[tmpx / r**2 + tmpy / r**2 + tmpz / rz**2 < 1].mean()
        loss -= image_t[(tmpx / r**2 + tmpy / r**2 + tmpz / rz**2 >= 1) & (tmpx / (r + 3)**2 + tmpy / (r + 3)**2 + tmpz / (rz + 3)**2 < 1)].mean()
        return -loss

    rmin = 12
    rs = np.arange(rmin, rmin + 14)
    rzs = np.arange(1, 14)
    timeframes = [5, 8, 11, 15, 22, 28, 33, 40, 45, 50, 54, 59]
    lossg = np.zeros((len(timeframes), len(rs), len(rzs)))
    for ii, i in enumerate(timeframes):
        for ir, r in enumerate(rs):
            for irz, rz in enumerate(rzs):
                lossg[ii, ir, irz] = lossfun(r, rz, ii)

    fig, axs = plt.subplots(1, len(timeframes))
    for ii, i in enumerate(timeframes):
        ax = axs[ii]
        ax.imshow(-lossg[ii], interpolation='nearest')
        time = (i*45.0)/60
        ax.set_title(f"{time} mins")
    fig.tight_layout()
