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
    scale=1,
    cmap="coolwarm",
    size=(10, 3.5),
    shared_colorbar=True,
    colorbarname=None,
    ixlabel='all',
    iylabel='all',
    xlabel=None,
    ylabel=None,
    title=None,
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

        if xlabel:
            if ixlabel == 'all':
                ax.set_xlabel(xlabel)
            elif ixlabel == a:
                ax.set_xlabel(xlabel)

        if ylabel:
            if iylabel == 'all':
                ax.set_ylabel(ylabel)
            elif iylabel == a:
                ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title[a])

        if not shared_colorbar:
            if colorbarname:
                fig.colorbar(pcm, ax=ax, label=colorbarname)
            else:
                fig.colorbar(pcm, ax=ax)

        a += 1

    if shared_colorbar:
        if colorbarname:
            fig.colorbar(pcm, ax=axes, label=colorbarname)
        else:
            fig.colorbar(pcm, ax=axes)

    if savedir:
        _save_plot(fig, savedir, savename, dpi=dpi, bbox_inches=bbox_inches)
    plt.show()
    return fig, axes

##############################################################################3

def plot_profiles(
    P,
    scale=1,
    size=(10, 3.5),
    shared_ylim=False,
    ixlabel='all',
    iylabel='all',
    ilegend='all',
    xlabel=None,
    ylabel=None,
    title=None,
    legend=None,
    color=None, # Blues, Reds, coolwarm
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
    for ax, group in zip(axes, groups):
        if color:
            cmap = plt.get_cmap(color)
            colors = cmap(np.linspace(colorgrad[0], colorgrad[1], len(group)))

            for p, c in zip(group, colors):
                ax.plot(
                    p["coord"] * scale,
                    p["values"],
                    color = c,
                    label=f't = {p["time"]:.1f}',
                )
        else:
            for p in group:
                ax.plot(
                    p["coord"] * scale,
                    p["values"],
                    label=f't = {p["time"]:.1f}',
                )

        if xlabel:
            if ixlabel == 'all':
                ax.set_xlabel(xlabel)
            elif ixlabel == a:
                ax.set_xlabel(xlabel)

        if ylabel:
            if iylabel == 'all':
                ax.set_ylabel(ylabel)
            elif iylabel == a:
                ax.set_ylabel(ylabel)

        if title:
            ax.set_title(title[a])

        if xlim:
            ax.set_xlim(xlim[0], xlim[1])

        if shared_ylim:
            ax.set_ylim(ymin, ymax)

        ax.grid()
        if legend:
            if ilegend == 'all':
                ax.legend(legend)
            elif ilegend == a:
                ax.legend(legend)
        a += 1

    if savedir:
        _save_plot(fig, savedir, savename, dpi=dpi, bbox_inches=bbox_inches)
    
    plt.show()
    
    return fig, axes

def plot_profile_fields(
    P,
    unit=None,
    scale=1,
    size=(10, 3.5),
    xlim=None,
    ylim=None,
    savedir=None,
    savename=None,
    dpi=300,
    bbox_inches="tight",
):
    """
    Plot multiple fields along the same profile line.

    P should be a list of profile dicts, for example:
        [profile_c_H2, profile_c_O2, profile_c_Acetate]
    """
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)

    for p in P:
        ax.plot(
            p["coord"] * scale,
            p["values"],
            label=p["field"],
        )

    along = P[0]["along"]

    if unit:
        ax.set_xlabel(f"{along} {unit[0]}")
        ax.set_ylabel(unit[1])
    else:
        ax.set_xlabel(along)
        ax.set_ylabel("value")

    # fixed_txt = ", ".join(
    #     f"{k} = {v * scale:.2f}" for k, v in P[0]["fixed"].items()
    # )

    # ax.set_title(f't = {P[0]["time"]}, {fixed_txt}')

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    ax.grid()
    ax.legend()

    if savedir:
        _save_plot(fig, savedir, savename, dpi=dpi, bbox_inches=bbox_inches)

    plt.show()
    return fig, ax