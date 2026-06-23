import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def _as_list(x):
    if isinstance(x, list):
        return x
    return [x]


def _normalize_profile_groups(P):
    """
    Accepts:
      profile_dict
      [profile_dict, profile_dict, ...]
      [[profile_dict, ...], [profile_dict, ...], ...]
    Returns:
      [[profile_dict, ...], [profile_dict, ...], ...]
    """
    if isinstance(P, dict):
        return [[P]]

    if isinstance(P, list) and len(P) > 0 and isinstance(P[0], dict):
        return [P]

    return P


def _save_plot(fig, savedir=None, savename=None, dpi=300, bbox_inches="tight"):
    if savedir is None or savename is None:
        return

    savedir = Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    savename = str(savename)
    if "." not in Path(savename).name:
        savename += ".png"

    fig.savefig(savedir / savename, dpi=dpi, bbox_inches=bbox_inches)

def plot_slice(
    S,
    unit=None,
    scale=1,
    cmap="coolwarm",
    size=(10, 3.5),
    shared_colorbar=True,
    ixlabel='all',
    iylabel='all',
    nRound=None,
    flip=None,
    savedir=None,
    savename=None,
    dpi=300,
    bbox_inches="tight",
):
    if flip:
        S = S[::-1]

    S = _as_list(S)

    n = len(S)
    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(size[0], size[1] * n),
        constrained_layout=True,
    )

    if n == 1:
        axes = [axes]

    if shared_colorbar:
        vmin = min(np.nanmin(s["values"]) for s in S)
        vmax = max(np.nanmax(s["values"]) for s in S)
    else:
        vmin = vmax = None

    pcm = None
    a = 0

    for ax, s in zip(axes, S):
        A, B = np.meshgrid(s["a"], s["b"])

        pcm = ax.pcolormesh(
            A * scale,
            B * scale,
            s["values"],
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        if ixlabel == 'all':
            if unit:
                ax.set_xlabel(f'{s["a_name"]} {unit[0]}')
            else:
                ax.set_xlabel(f'{s["a_name"]}')
        elif ixlabel == a:
            if unit:
                ax.set_xlabel(f'{s["a_name"]} {unit[0]}')
            else:
                ax.set_xlabel(f'{s["a_name"]}')
        if iylabel == 'all':
            if unit:
                ax.set_ylabel(f'{s["b_name"]} {unit[0]}')
            else:
                ax.set_ylabel(f'{s["b_name"]}')
        elif iylabel == a:
            if unit:
                ax.set_ylabel(f'{s["b_name"]} {unit[0]}')
            else:
                ax.set_ylabel(f'{s["b_name"]}')

        if unit:
            if nRound:
                ax.set_title(
                    f'{s["fixed_axis"]} = {np.round(s["fixed_value"],nRound) * scale:.2f} {unit[0]}'
                )
            else:
                ax.set_title(
                    f'{s["fixed_axis"]} = {s["fixed_value"] * scale:.2f} {unit[0]}'
                )
        else:
            if nRound:
                ax.set_title(
                    f'{s["fixed_axis"]} = {np.round(s["fixed_value"],nRound) * scale:.2f}'
                )
            else:
                ax.set_title(
                    f'{s["fixed_axis"]} = {s["fixed_value"] * scale:.2f}'
                )

        if not shared_colorbar:
            label = s["field"] if unit is None else f'{s["field"]} {unit}'
            fig.colorbar(pcm, ax=ax, label=label)

        a += 1

    if shared_colorbar:
        label = S[0]["field"] if unit is None else f'{S[0]["field"]} {unit[1]}'
        fig.colorbar(pcm, ax=axes, label=label)

    if savedir:
        _save_plot(fig, savedir, savename, dpi=dpi, bbox_inches=bbox_inches)
    plt.show()
    return fig, axes

def plot_profiles(
    P,
    unit=None,
    scale=1,
    size=(10, 3.5),
    shared_ylim=True,
    ixlabel='all',
    iylabel='all',
    ilegend=None,
    nRound=None,
    color='Blues', # Blues, Reds, coolwarm
    colorgrad = [0.3, 1],
    pad=None,
    flip=None,
    xlim=None,
    ylim=None,
    savedir=None,
    savename=None,
    dpi=300,
    bbox_inches="tight",
):    
    # P can be
    # P[dim1]
    # P[dim1][dim2]
    # or
    # P[dim1][dim2][dim3]
    # example 
    # P[z][time][y] at fixed x 
    # this creates subplot for each z with all times in each plot
    # or
    # P[z][y] at fixed time and x 
    # this creates 1 plot with all different z 
    
    groups = _normalize_profile_groups(P)

    if flip:
        groups = groups[::-1]
    
    n = len(groups)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(size[0], size[1] * n),
        constrained_layout=True,
    )

    if n == 1:
        axes = [axes]

    if shared_ylim:
        if ylim:
            ymin = ylim[0]
            ymax = ylim[1]
        else:
            ymin = min(
                np.nanmin(p["values"])
                for group in groups
                for p in group
            )
            ymax = max(
                np.nanmax(p["values"])
                for group in groups
                for p in group
            )
        if pad:
            pad = pad * (ymax - ymin) if ymax > ymin else 1.0
            ymin -= pad
            ymax += pad
    else:
        ymin = ymax = None

    a = 0
    # handles, labels = axes[0].get_legend_handles_labels()
    for ax, group in zip(axes, groups):

        cmap = plt.get_cmap(color)
        colors = cmap(np.linspace(colorgrad[0], colorgrad[1], len(group)))

        for p, c in zip(group, colors):
            ax.plot(
                p["coord"] * scale,
                p["values"],
                color = c,
                label=f't = {p["time"]:.1f}',
            )

        along = group[0]["along"]
        field = group[0]["field"]

        if ixlabel == 'all':
            if unit:
                ax.set_xlabel(f"{along} {unit[0]}")
            else:
                ax.set_xlabel(f"{along}")
        elif ixlabel == a:
            if unit:
                ax.set_xlabel(f"{along} {unit[0]}")
            else:
                ax.set_xlabel(f"{along}")

        if iylabel == 'all':
            if unit:
                ax.set_ylabel(f"{field} {unit[1]}")
            else:
                ax.set_ylabel(f"{field}")
        elif iylabel == a:
            if unit:
                ax.set_ylabel(f"{field} {unit[1]}")
            else:
                ax.set_ylabel(f"{field}")

        if unit:
            if nRound:
                fixed_txt = ", ".join(
                    f"{k} = {np.round(v,nRound) * scale:.2f} {unit[0]}"
                    for k, v in group[0]["fixed"].items()
                )
            else:
                fixed_txt = ", ".join(
                    f"{k} = {v * scale:.2f} {unit[0]}"
                    for k, v in group[0]["fixed"].items()
                )
        else:
            if nRound:
                fixed_txt = ", ".join(
                    f"{k} = {np.round(v,nRound) * scale:.2f}"
                    for k, v in group[0]["fixed"].items()
                )
            else:
                fixed_txt = ", ".join(
                    f"{k} = {v * scale:.2f}"
                    for k, v in group[0]["fixed"].items()
                )
        if xlim:
            ax.set_xlim(xlim[0], xlim[1])
        ax.set_title(f"{fixed_txt}")

        if shared_ylim:
            ax.set_ylim(ymin, ymax)

        ax.grid()
        if ilegend == a:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5)
                )

        a += 1
    if savedir:
        _save_plot(fig, savedir, savename, dpi=dpi, bbox_inches=bbox_inches)
    plt.show()
    return fig, axes

