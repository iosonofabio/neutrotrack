import pathlib
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def load_data(
    cell_type="neutrophil",
    conditions=("swarming", "non"),
):
    data_cache_fdn = pathlib.Path(__file__).parent.parent.parent / "data" / "cache"

    data_dict = {}
    for subfdn in data_cache_fdn.iterdir():
        condition = subfdn.name
        if condition not in conditions:
            continue
        for video_fdn in subfdn.iterdir():
            video = video_fdn.name
            for ct_fdn in video_fdn.iterdir():
                ct = ct_fdn.name

                if ct != cell_type:
                    continue

                with open(ct_fdn / "cache.pkl", "rb") as f:
                    datum = pickle.load(f)
                data_dict[(condition, ct, video)] = datum

    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--celltype",
        choices=["neutrophil", "monocyte", "dendritic"],
        default="neutrophil",
        help="Cell type to analyze",
    )
    parser.add_argument("--conditions", default="swarming", help="Condition to analyze")
    parser.add_argument("--plot", action="store_true", help="Plot the data")
    parser.add_argument("--plot-fast", action="store_true", help="Plot the fast tracks")
    parser.add_argument("--simple", action="store_true", help="Use simple plot")
    parser.add_argument(
        "--dim", choices=["2d", "3d"], default="3d", help="Plot dimension"
    )
    parser.add_argument(
        "--color",
        choices=["time", "speed"],
        default="time",
        help="Color by time or speed",
    )
    args = parser.parse_args()

    conditions = args.conditions.replace(" ", "").split(",")
    data_dict = load_data(conditions=conditions, cell_type=args.celltype)

    for key, df in data_dict.items():
        break

    # df has "track" with the track/cell names etc.
    # The actual data is in df["step"]
    cols = [f"Position {x}" for x in ["X", "Y", "Z"]]

    # NOTE: dendritic cells have slightly different column names ??
    # ...sigh, but not consistently within each data frame
    col0 = cols[0] if cols[0] in df["step"] else cols[0].replace(" ", "_")
    df_pos = df["step"][col0].copy()
    if col0 != cols[0]:
        df_pos.rename(columns={col0: cols[0]}, inplace=True)
    for col in cols[1:] + ["Speed"]:
        col1 = col if col in df["step"] else col.replace(" ", "_")
        tmp = df["step"][col1]
        col2 = col if col in tmp.columns else col1
        df_pos[col] = tmp.loc[:, col2]

    # Plot the actual tracks
    if args.plot:
        if args.dim == "2d":
            kwargs = {}
        else:
            kwargs = dict(
                subplot_kw={"projection": "3d"},
            )
        fig, ax = plt.subplots(
            **kwargs,
        )
        for trackid, dfi in df_pos.groupby("TrackID"):
            size = len(dfi)
            if size < 10:
                continue

            x, y, z = dfi[cols].values.T
            if args.color == "time":
                colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
            elif args.color == "speed":
                speed_norm = dfi["Speed"] / dfi["Speed"].max()
                colors = plt.cm.viridis(speed_norm.values)

            if args.dim == "2d":
                if args.simple:
                    ax.plot(x, y, label=trackid, alpha=0.5)
                else:
                    for i in range(len(x) - 1):
                        ax.plot(x[i : i + 2], y[i : i + 2], alpha=0.5, color=colors[i])
            else:
                if args.simple:
                    ax.plot(x, y, z, label=trackid, alpha=0.5)
                else:
                    for i in range(size - 1):
                        ax.plot(
                            x[i : i + 2],
                            y[i : i + 2],
                            z[i : i + 2],
                            alpha=0.5,
                            color=colors[i],
                        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if args.dim == "3d":
            ax.set_zlabel("z")
        fig.tight_layout()

        plt.ion()
        plt.show()

    # The injury site is roughly in the middle of the image
    # (Cat did not send me, despite asking her 6+ months ago to get it written down)
    xmax, ymax, zmax = df_pos[cols].max().values
    xmin, ymin, zmin = df_pos[cols].min().values
    xmid, ymid, zmid = (xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2
    df_pos["distance_from_center"] = np.sqrt(
        ((df_pos[cols] - np.array([xmid, ymid, zmid])) ** 2).sum(axis=1)
    )

    if args.plot:
        fig, ax = plt.subplots()
        ax.scatter(
            df_pos["Time Index"],
            df_pos["distance_from_center"],
            color="k",
            alpha=0.01,
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Distance from center")
        fig.tight_layout()
        plt.ion()
        plt.show()

    # Identify tracks that become much closer to the center
    delta_dist = []
    for trackid, dfi in df_pos.groupby("TrackID"):
        size = len(dfi)

        # Cut shot tracks
        if size < 20:
            continue

        idx_dmax = dfi["distance_from_center"].idxmax()
        idx_dmin = dfi["distance_from_center"].idxmin()
        dmax = dfi.loc[idx_dmax, "distance_from_center"]
        dmin = dfi.loc[idx_dmin, "distance_from_center"]
        delta = dmax - dmin
        time_dmax = dfi.at[idx_dmax, "Time Index"]
        time_dmin = dfi.at[idx_dmin, "Time Index"]
        delta_time = time_dmax - time_dmin

        # Subsequenct frames can be mistrackings
        if -delta_time <= 5:
            continue

        delta_dist.append(
            {
                "trackid": trackid,
                "dmax": dmax,
                "dmin": dmin,
                "delta": delta,
                "time_dmax": time_dmax,
                "time_dmin": time_dmin,
                "delta_time": delta_time,
                "avg_speed": -delta / delta_time,
                "size": size,
            }
        )
    delta_dist = pd.DataFrame(delta_dist)

    if args.plot:
        fig, ax = plt.subplots()
        ax.scatter(
            delta_dist["delta"],
            delta_dist["avg_speed"],
        )
        ax.set_xlabel("Delta distance")
        ax.set_ylabel("Average speed")
        fig.tight_layout()
        plt.ion()
        plt.show()

    fast_tracks = delta_dist.nlargest(20, "delta")["trackid"].values

    if args.plot_fast:
        fig, ax = plt.subplots()
        ax.set_title(f"{args.celltype}, {args.conditions}")
        for trackid, dfi in df_pos.groupby("TrackID"):
            if trackid not in fast_tracks:
                continue
            ax.scatter(
                dfi["Time Index"],
                dfi["distance_from_center"],
                color="k",
                alpha=0.7,
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Distance from center")
        fig.tight_layout()
        plt.ion()
        plt.show()

    # Plot the actual tracks
    if args.plot_fast:
        if args.dim == "2d":
            kwargs = {}
        else:
            kwargs = dict(
                subplot_kw={"projection": "3d"},
            )
        fig, ax = plt.subplots(
            **kwargs,
        )
        ax.set_title(f"{args.celltype}, {args.conditions}")
        for trackid, dfi in df_pos.groupby("TrackID"):
            if trackid not in fast_tracks:
                continue
            x, y, z = dfi[cols].values.T
            if args.color == "time":
                colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
            elif args.color == "speed":
                speed_norm = dfi["Speed"] / dfi["Speed"].max()
                colors = plt.cm.viridis(speed_norm.values)

            if args.dim == "2d":
                if args.simple:
                    ax.plot(x, y, label=trackid, alpha=0.5)
                else:
                    for i in range(len(x) - 1):
                        ax.plot(x[i : i + 2], y[i : i + 2], alpha=0.9, color=colors[i])
            else:
                if args.simple:
                    ax.plot(x, y, z, label=trackid, alpha=0.5)
                else:
                    for i in range(len(x) - 1):
                        ax.plot(
                            x[i : i + 2],
                            y[i : i + 2],
                            z[i : i + 2],
                            alpha=0.5,
                            color=colors[i],
                        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if args.dim == "3d":
            ax.set_zlabel("z")
        fig.tight_layout()

        plt.ion()
        plt.show()
