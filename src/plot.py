import random
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.patches import Patch
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython.display import HTML

from utils import replace_none_nan, BASE_CRS, GEO_CRS, SQM_TO_SQFT, format_sig_figs_compact, sigfig_formatter_compact
from geometry import compute_component_regions
from network import percolate_graph, run_simulation


# 1. Parcel-level summary
def plot_summary(
    gdf_results: gpd.GeoDataFrame,
    bldgs: Optional[gpd.GeoDataFrame] = None,
    destroyed: Optional[gpd.GeoDataFrame] = None,
    spared: Optional[gpd.GeoDataFrame] = None,
    tr_min: float = 0,
    tr_max: Optional[float] = None,
    cmap: str = 'Reds',
    sr_min: float = 0,
    sr_max: Optional[float] = None,
    pr_min: float = 0,
    pr_max: Optional[float] = None,
    or_min: float = 0,
    or_max: Optional[float] = None,
    save: Optional[str] = None,
    histogram_height_ratio: float = 0.5,
    histogram_width_ratio: float = 1.0, 
    FIGURE_WIDTH: float = 15, 
    FIGURE_HEIGHT: float = 5, 
    LABEL_SIZE=15
) -> None:
    
    ALPHA = 0.5
    LINE_WIDTH = 1.5
    TITLE_SIZE = 24
    N_BINS = 20

    _sigfig_format = sigfig_formatter_compact

    # TR (Panel A) sums PR + SUM(SR_owner) + SUM(OR) per row (elemntwise sum)
    _sum_list = lambda x: np.nansum(x) if isinstance(x, (list, tuple, np.ndarray)) else (x if x else 0)
    gdf_results['SR_owner_total'] = gdf_results['SR_owner'].apply(_sum_list)
    gdf_results['OR_total'] = gdf_results['OR'].apply(_sum_list)

    # Plot
    fig, axs = plt.subplots(
        2, 4,
        figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
        gridspec_kw={
            'height_ratios': [1, histogram_height_ratio],
            'width_ratios': [1, 1, 1, 1] # Ensure equal width for all columns
        }
    )

    ax = axs.flatten()

    # Prioritize bldgs gdf for extent, o/w use gdf_results
    if bldgs is not None and not bldgs.empty:
        minx, miny, maxx, maxy = bldgs.total_bounds
    elif not gdf_results.empty:
        minx, miny, maxx, maxy = gdf_results.total_bounds
    else:
        minx, miny, maxx, maxy = 0, 0, 1, 1 # Default

    # Plot responsibility maps
    map_columns = ['TR', 'PR', 'SR_owner_total', 'OR_total']
    map_cmaps = ['bone_r', 'Greens', 'Reds', 'Blues']
    map_titles = ['Total', 'PR', 'SR', 'OR']
    panel_labels = ['A', 'B', 'C', 'D']
    min_vals = [tr_min, pr_min, sr_min, or_min]
    max_vals = [tr_max, pr_max, sr_max, or_max]

    for i, col in enumerate(map_columns):
        # Plot main data
        gdf_results.plot(
            ax=ax[i],
            column=col,
            cmap=map_cmaps[i],
            alpha=ALPHA,
            edgecolor='none',
            vmin=min_vals[i],
            vmax=max_vals[i]
        )
        norm = colors.Normalize(vmin=min_vals[i], vmax=max_vals[i])
        sm = cm.ScalarMappable(cmap=map_cmaps[i], norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax[i], format=_sigfig_format) 
        cbar.ax.tick_params(labelsize=LABEL_SIZE)

        # # Set consistent x and y limits for all map plots
        # ax[i].set_xlim(minx, maxx)
        # ax[i].set_ylim(miny, maxy)

        #TODO Overlays: Plot buildings, destroyed, spared
        # Check if bldgs is not empty before attempting to plot its bounds
        if bldgs is not None and not bldgs.empty:
            bldgs.plot(ax=ax[i], color='none', edgecolor='k', linewidth=0.5)
        if destroyed is not None and not destroyed.empty:
            destroyed.plot(ax=ax[i], color='k', edgecolor='k', linewidth=0.5)
        if spared is not None and not spared.empty:
            spared.plot(ax=ax[i], color='lightblue', edgecolor='k', linewidth=0.5)

        ax[i].set_title(
            f"$\\bf{{{panel_labels[i]}}}$      {map_titles[i]}",
            size=TITLE_SIZE, 
            loc='left'
        )
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    # Plot kdeplots
    histogram_columns = ['TR', 'PR', 'SR_owner_total', 'OR_total']
    histogram_colors = ['k', 'g', 'r', 'lightblue']
    histogram_titles = ['Total', 'PR', 'SR', 'OR']

    for j, col in enumerate(histogram_columns):
        idx = j + 4 # Index for flattened array
        sns.histplot(
            x=gdf_results[col],
            color=histogram_colors[j],
            linewidth=LINE_WIDTH, 
            label=histogram_titles[j],
            bins=N_BINS, 
            ax=ax[idx]
        )
        ax[idx].tick_params(axis='both', labelsize=LABEL_SIZE) 
        ax[idx].xaxis.set_major_formatter(_sigfig_format) 
        ax[idx].set_xlabel(histogram_titles[j], size=LABEL_SIZE) 
        ax[idx].set_ylabel("")

    ax[4].set_ylabel("Count", fontsize=16) 

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    plt.show()



# Whole-neighborhood PR/SR/OR region maps
def plot_component_map(gdf_results, component='PR', site=None, figsize=(14, 10), color=None, save=False):

    color_map = {'PR': 'seagreen', 'SR': 'gold', 'OR': 'royalblue'}
    color = color or color_map[component]

    regions = compute_component_regions(gdf_results, component=component)

    fig, ax = plt.subplots(1, figsize=figsize)

    if site is not None:
        site.to_crs(GEO_CRS).plot(ax=ax, color='none', edgecolor='gray', alpha=0.5)

    gdf_results.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
        ax=ax, color='none', edgecolor='lightgray', linewidth=0.5)

    if len(regions) > 0:
        regions.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
            ax=ax, color=color, edgecolor='none', alpha=0.75)

    ax.set_title(f"{component} regions across the neighborhood ({len(regions)} regions)", fontsize=16)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()

    return regions


def plot_SR_OR_map(gdf_results, sr_regions=None, or_regions=None, site=None, bldgs=None,
                    figsize=(14, 10), save=False):

    if sr_regions is None:
        sr_regions = compute_component_regions(gdf_results, component='SR')
    if or_regions is None:
        or_regions = compute_component_regions(gdf_results, component='OR')

    fig, ax = plt.subplots(1, figsize=figsize)

    if site is not None:
        site.to_crs(GEO_CRS).plot(ax=ax, color='none', edgecolor='gray', alpha=0.5)

    gdf_results.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
        ax=ax, color='none', edgecolor='lightgray', linewidth=0.5)

    if bldgs is not None:
        bldgs.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(ax=ax, color='k', zorder=10)

    if len(sr_regions) > 0:
        sr_regions.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
            ax=ax, color='gold', edgecolor='none', alpha=0.75)

    if len(or_regions) > 0:
        or_regions.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
            ax=ax, color='royalblue', edgecolor='none', alpha=0.75)

    legend_elements = [
        Patch(facecolor='k', label='Building'),
        Patch(facecolor='none', edgecolor='lightgray', linewidth=1, label='Parcel'),
        Patch(facecolor='gold', alpha=0.75, edgecolor='k', label='SR'),
        Patch(facecolor='royalblue', alpha=0.75, edgecolor='k', label='OR'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=14)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()


def plot_PR_SR_OR_map(gdf_results, pr_regions=None, sr_regions=None, or_regions=None,
                       site=None, bldgs=None, figsize=(14, 10), save=False):

    if pr_regions is None:
        pr_regions = compute_component_regions(gdf_results, component='PR')
    if sr_regions is None:
        sr_regions = compute_component_regions(gdf_results, component='SR')
    if or_regions is None:
        or_regions = compute_component_regions(gdf_results, component='OR')

    fig, ax = plt.subplots(1, figsize=figsize)

    if site is not None:
        site.to_crs(GEO_CRS).plot(ax=ax, color='none', edgecolor='gray', alpha=0.5)

    gdf_results.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
        ax=ax, color='none', edgecolor='lightgray', linewidth=0.5)

    if bldgs is not None:
        bldgs.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(ax=ax, color='k', zorder=10)

    if len(pr_regions) > 0:
        pr_regions.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
            ax=ax, color='seagreen', edgecolor='none', alpha=0.75)

    if len(sr_regions) > 0:
        sr_regions.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
            ax=ax, color='gold', edgecolor='none', alpha=0.75)

    if len(or_regions) > 0:
        or_regions.set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
            ax=ax, color='royalblue', edgecolor='none', alpha=0.75)

    legend_elements = [
        Patch(facecolor='k', label='Building'),
        Patch(facecolor='none', edgecolor='lightgray', linewidth=1, label='Parcel'),
        Patch(facecolor='seagreen', alpha=0.75, edgecolor='k', label='PR'),
        Patch(facecolor='gold', alpha=0.75, edgecolor='k', label='SR'),
        Patch(facecolor='royalblue', alpha=0.75, edgecolor='k', label='OR'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=14)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()


def plot_area_and_responsibility_totals(gdf_results, pr_regions, sr_regions, or_regions, comp_rank_or,
                                         label_size=16, figsize=(12, 5), save=False):

    lw = 1
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    area_labels = ['Personal', 'Shared', 'Owed']
    area_colors = ['seagreen', 'gold', 'royalblue']
    area_values = [
        pr_regions.area.sum() * SQM_TO_SQFT,
        sr_regions.area.sum() * SQM_TO_SQFT,
        or_regions.area.sum() * SQM_TO_SQFT,
    ]

    ax[0].bar(area_labels, area_values, color=area_colors, alpha=0.7, edgecolor='k', linewidth=lw)
    ax[0].set_ylabel("Total Area [$ft^2$]", size=label_size)
    ax[0].tick_params(axis='both', labelsize=label_size)
    ax[0].yaxis.set_major_formatter(sigfig_formatter_compact)
    ax[0].legend(handles=[Patch(facecolor=c, alpha=0.7, edgecolor='k', label=l)
                           for l, c in zip(area_labels, area_colors)],
                 loc='best', fontsize=label_size - 4)

    sr_total_true = 0.0
    for row_sr_owner in gdf_results['SR_owner']:
        if isinstance(row_sr_owner, (list, tuple, np.ndarray)):
            sr_total_true += np.nansum(row_sr_owner)
        elif row_sr_owner:
            sr_total_true += row_sr_owner

    resp_labels = ['Personal', 'Shared', 'Owed']
    resp_colors = ['seagreen', 'gold', 'royalblue']
    resp_values = [
        np.nansum(gdf_results['PR']),
        sr_total_true,
        comp_rank_or['total_risk'].sum(),
    ]

    ax[1].bar(resp_labels, resp_values, color=resp_colors, alpha=0.7, edgecolor='k', linewidth=lw)
    ax[1].set_ylabel("Total Responsibility [$ft^3/min$]", size=label_size)
    ax[1].tick_params(axis='both', labelsize=label_size)
    ax[1].yaxis.set_major_formatter(sigfig_formatter_compact)
    ax[1].legend(handles=[Patch(facecolor=c, alpha=0.7, edgecolor='k', label=l)
                           for l, c in zip(resp_labels, resp_colors)],
                 loc='best', fontsize=label_size - 4)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()

    return area_values, resp_values


# Network topology plot
def plot_network(
    G,
    variable='SR',
    node_variable='TR',
    colormap='Reds',
    node_colormap='Blues',
    node_variable_label=None,
    figsize_x=15,
    figsize_y=5,
    node_size=200,
    font_size=10,
    label_size=14,
    cbar_pad=-0.075,
    site=None,
    arrowsize=15,
    ds=None,
    node_vmax=None,
    node_vmin=0,
    edge_vmax=None,
    edge_vmin=0,
    caption=None,
    save=False,
    basemap=False,
    basemap_source=None,
    labeling=True,
    edge_unit_label='ft^3/min',
    node_unit_label='ft^3/min',
    edge_variable_label=None,  
    cbar_tick_style='categorical', 
    cbar_tick_fontsize=None 
):
    import matplotlib.pyplot as plt
    import matplotlib
    import networkx as nx
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    fig, ax = plt.subplots(1, figsize=(figsize_x, figsize_y))
    ax.set_aspect(1)

    tick_fontsize = cbar_tick_fontsize
    if tick_fontsize is None:
        tick_fontsize = max(label_size - 12, 9) if cbar_tick_style == 'categorical' else label_size - 2
    if site is not None:
        site.plot(ax=ax, color='none', edgecolor='gray', alpha=0.5)
    if ds is not None:
        ds.plot(ax=ax, color='aliceblue', edgecolor='royalblue', linestyle='--', alpha=0.15)

    # Plot neighborhood network
    node_positions = nx.get_node_attributes(G, 'pos')

    node_TR = replace_none_nan(nx.get_node_attributes(G, node_variable))
    cmap_TR = plt.cm.get_cmap(node_colormap)

    nx.draw_networkx_nodes(
        G,
        pos=node_positions,
        node_size=node_size,
        cmap=cmap_TR,
        node_color=list(node_TR.values()),
        alpha=0.8,
        edgecolors='black', 
        linewidths=1.5,    
        vmin=node_vmin,
        vmax=node_vmax
    )

    if labeling:
        node_labels = {idx: str(idx) for idx in G.nodes}
        nx.draw_networkx_labels(G, pos=node_positions, labels=node_labels, font_size=font_size, font_color='k')

    # Link attributes
    edge_var = replace_none_nan(nx.get_edge_attributes(G, variable))
    unique_values = list(edge_var.values()) if edge_var else [0]
    cmap_edge = plt.cm.get_cmap(colormap)

    for edge, value in edge_var.items():
        nx.draw_networkx_edges(
            G,
            pos=node_positions,
            edgelist=[edge],
            width=3,
            edge_color=[value],
            edge_cmap=cmap_edge,
            arrowstyle="->",
            connectionstyle='arc3,rad=0.3',
            arrowsize=arrowsize,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax
        )

    # Lat/lon axis labels (for geographic info)
    def lat_lon_formatter(x, pos):
        return f"{x:.4f}"
    ax.xaxis.set_major_formatter(FuncFormatter(lat_lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_lon_formatter))

    # Node colorbar (PR)
    node_vmax_eff = node_vmax if node_vmax is not None else max(node_TR.values())
    sm_TR = matplotlib.cm.ScalarMappable(
        cmap=cmap_TR,
        norm=Normalize(vmin=0, vmax=node_vmax_eff)
    )
    sm_TR.set_array([])
    cbar_ax1 = inset_axes(ax,
                          width="18%",    
                          height="2%",    
                          loc='lower right',
                          bbox_to_anchor=(-0.035, 0.25, 1, 1),  
                          bbox_transform=ax.transAxes,
                          borderpad=0)

    cbar_TR = fig.colorbar(sm_TR, cax=cbar_ax1, orientation='horizontal')
    cbar_TR.set_label(label=f"PR [${node_unit_label}$]", fontsize=label_size, rotation=0, labelpad=5)
    cbar_TR.ax.xaxis.set_ticks_position('bottom')
    
    # Colorbar formatting (Categorical --> Low to high / sigfigs)
    if cbar_tick_style == 'categorical':
        cbar_TR.set_ticks([0, node_vmax_eff])
        cbar_TR.set_ticklabels([format_sig_figs_compact(0), format_sig_figs_compact(node_vmax_eff)])
    cbar_TR.ax.tick_params(labelsize=tick_fontsize)

    # Link colorbar (SR or OR)
    edge_vmax_eff = edge_vmax if edge_vmax is not None else max(unique_values)
    sm = matplotlib.cm.ScalarMappable(
        cmap=cmap_edge,
        norm=Normalize(vmin=0, vmax=edge_vmax_eff)
    )
    sm.set_array([])

    # Colorbar formatting with sigfig
    cbar_ax2 = inset_axes(ax,
                          width="18%",   
                          height="2%",   
                          loc='lower right',
                          bbox_to_anchor=(-0.035, 0.12, 1, 1),  
                          bbox_transform=ax.transAxes,
                          borderpad=0)

    cbar = fig.colorbar(sm, cax=cbar_ax2, orientation='horizontal')
    edge_label_text = edge_variable_label if edge_variable_label is not None else variable
    cbar.set_label(label=f"${edge_label_text}$ [${edge_unit_label}$]", fontsize=label_size, rotation=0, labelpad=5)
    cbar.ax.xaxis.set_ticks_position('bottom')
    if cbar_tick_style == 'categorical':
        cbar.set_ticks([0, edge_vmax_eff])
        cbar.set_ticklabels([format_sig_figs_compact(0), format_sig_figs_compact(edge_vmax_eff)])
    cbar.ax.tick_params(labelsize=tick_fontsize)

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    plt.show()
    return edge_var


class HandlerColormap(HandlerBase):
    """Handler to draw colormap gradient used by `plot_results_final`for network link removal """

    def __init__(self, cmap, num_stripes=40, low=0.0, high=1.0, **kwargs):
        self.cmap = cmap
        self.num_stripes = num_stripes
        self.low = low
        self.high = high
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        stripes = []
        for i in range(self.num_stripes):
            s_width = width / self.num_stripes
            val = self.low + (i / (self.num_stripes - 1)) * (self.high - self.low)
            color = self.cmap(val)
            rect = mpatches.Rectangle(
                [xdescent + i * s_width, ydescent], s_width, height,
                facecolor=color, edgecolor='none', transform=trans,
            )
            stripes.append(rect)
        border = mpatches.Rectangle(
            [xdescent, ydescent], width, height,
            facecolor='none', edgecolor='black', linewidth=0.5, transform=trans,
        )
        stripes.append(border)
        return stripes


def plot_results_final(results,
                        variable_label='SR',
                        theme_colors=('#d62728', '#1f77b4', '#2ca02c'),
                        cmaps=("RdPu", "PuBu", "YlGn"),
                        low=0.2, high=1.0,
                        bar_width=1,
                        overlay_lines=True,
                        num_colors=None):

    modes = list(results.keys())

    if len(modes) == 1:
        fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                 gridspec_kw={'height_ratios': [1, 2.5], 'hspace': 0.1})
        axes = np.array([axes]).T
    else:
        fig, axes = plt.subplots(2, len(modes), figsize=(7 * len(modes), 7), sharex=True,
                                 gridspec_kw={'height_ratios': [1, 2.5], 'hspace': 0.1})

    risk_col = f'total_{variable_label.lower()}'

    # Max subnetwork count across all strategies used as shared color scale for every panel
    max_subnetworks_global = 0
    for mode in modes:
        df = results[mode]['data']
        active_df = df[df[risk_col] > 0]
        if not active_df.empty:
            local_max = active_df.groupby('step')['id'].nunique().max()
            max_subnetworks_global = max(max_subnetworks_global, local_max)

    for idx, mode in enumerate(modes):
        df = results[mode]['data'].copy()

        if len(modes) > 1:
            ax_top, ax_bottom = axes[0, idx], axes[1, idx]
        else:
            ax_top, ax_bottom = axes[0, 0], axes[1, 0]

        theme_color = theme_colors[idx]
        mode_num_colors = num_colors if num_colors is not None else max_subnetworks_global

        # Plot n_subnetwork
        active_df = df[df[risk_col] > 0]
        if not active_df.empty:
            ncc_series = active_df.groupby('step')['id'].nunique()
            ax_top.plot(ncc_series.index, ncc_series.values, color=theme_color, lw=3)

        ax_top.set_title(f"{mode}", fontsize=30, pad=20)
        ax_top.set_ylabel("Number of\nsubnetworks", fontsize=20) if idx == 0 else ax_top.set_ylabel("")
        ax_top.set_ylim(bottom=0, top=max_subnetworks_global * 1.2)
        ax_top.xaxis.set_minor_locator(NullLocator())
        ax_top.grid(True, axis='y', alpha=0.3)
        ax_top.grid(True, axis='x', alpha=0.3)
        ax_top.tick_params('both', labelsize=20, labelbottom=False)

        # plot stacked responsibility share
        steps = sorted(df['step'].unique())
        global_max_risk = df[df['step'] == 0][risk_col].sum()
        cmap = plt.get_cmap(cmaps[idx])
        fixed_colors = [cmap(x) for x in np.linspace(low, high, max_subnetworks_global)]

        plot_data_list = []
        auc_x, auc_y = [], []

        for step in steps:
            step_df = df[df['step'] == step]
            frac = step_df['fraction'].iloc[0]
            current_total_risk = step_df[risk_col].sum()
            rel_risk = (current_total_risk / global_max_risk) if global_max_risk > 0 else 0
            auc_x.append(frac)
            auc_y.append(rel_risk)

            all_rows = step_df.sort_values(by=risk_col, ascending=False)
            row_dict = {'step': step}

            #TODO: FIX: Bucket subnetworks beyond mode_num_colors into "Other" since there may be additional subnetworks that may emerge 
            other_val = 0.0
            for i in range(len(all_rows)):
                val = (all_rows.iloc[i][risk_col] / global_max_risk) * 100
                if i < mode_num_colors:
                    row_dict[f"Rank_{i}"] = val
                else:
                    other_val += val
            if other_val > 0:
                row_dict["Other"] = other_val
            plot_data_list.append(row_dict)

        # Get AUC to quantify mitigation strategy performance
        auc_val = np.trapz(y=auc_y, x=auc_x)

        pivot_df = pd.DataFrame(plot_data_list).set_index('step').fillna(0)
        ordered_cols = [c for c in pivot_df.columns if c.startswith("Rank_")]
        ordered_cols.sort(key=lambda x: int(x.split('_')[1]))
        if "Other" in pivot_df.columns:
            ordered_cols = ordered_cols + ["Other"]
        pivot_df = pivot_df[ordered_cols]

        final_colors = []
        for col in ordered_cols:
            if col == "Other":
                final_colors.append('lightgray')
            else:
                rank_idx = int(col.split('_')[1])
                c_idx = min(rank_idx, len(fixed_colors) - 1)
                final_colors.append(fixed_colors[c_idx])

        pivot_df.plot(kind='bar', stacked=True, ax=ax_bottom, color=final_colors,
                      width=bar_width, edgecolor='k', linewidth=0.2, legend=False)

        if overlay_lines:
            x_indices = np.arange(len(pivot_df))
            x_plot = np.concatenate([x_indices - 0.5, [x_indices[-1] + 0.5]])
            total_vals = pivot_df.sum(axis=1).values
            total_plot = np.concatenate([total_vals, [total_vals[-1]]])
            ax_bottom.plot(x_plot, total_plot, color='black', lw=3, linestyle='-', drawstyle='steps-post')

        fractions_map = df[['step', 'fraction']].drop_duplicates().set_index('step')['fraction']
        desired_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        tick_indices, tick_labels = [], []

        for target in desired_ticks:
            diffs = (fractions_map - target).abs()
            if diffs.min() < 0.05:
                closest_step = diffs.idxmin()
                if closest_step in pivot_df.index:
                    loc = pivot_df.index.get_loc(closest_step)
                    tick_indices.append(loc)
                    if target == 0.0:
                        tick_labels.append("0")
                    elif target == 1.0:
                        tick_labels.append("1")
                    else:
                        tick_labels.append(f"{target:.1f}")

        ax_bottom.xaxis.set_major_locator(FixedLocator(tick_indices))
        ax_bottom.xaxis.set_minor_locator(NullLocator())
        ax_bottom.set_xticklabels(tick_labels, rotation=0)
        ax_bottom.set_xlabel("Fraction of links removed", fontsize=20)
        ax_bottom.tick_params('both', labelsize=20)

        ax_top.xaxis.set_major_locator(FixedLocator(tick_indices))
        ax_top.xaxis.set_minor_locator(NullLocator())

        ax_bottom.set_ylabel(f"{variable_label} [%]", fontsize=20) if idx == 0 else ax_bottom.set_ylabel("")
        ax_bottom.set_ylim(0, 105)

        proxy_rank = mpatches.Rectangle((0, 0), 1, 1)
        handler_map = {proxy_rank: HandlerColormap(cmap, num_stripes=10, low=low, high=high)}
        legend_titles = [f'# subnetworks [1, {max_subnetworks_global}]']
        legend_handles = [proxy_rank]

        if 'Other' in pivot_df.columns:
            other_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='lightgray')
            legend_handles.append(other_patch)
            legend_titles.append(f'Other (rank > {mode_num_colors})')

        if overlay_lines:
            line_total = Line2D([0], [0], color='k', lw=3, linestyle='-')
            legend_handles.append(line_total)
            legend_titles.append(f'Total {variable_label} (AUC: {auc_val:.3f})')

        ax_bottom.legend(legend_handles, legend_titles, handler_map=handler_map,
                         loc='upper right', fontsize=14, handlelength=3, frameon=False)

    plt.tight_layout()
    plt.show()


def plot_subnetwork_heatmaps(results, variable_label='SR', cmap='YlOrRd', figsize=(30, 16)):
    """
    Heatmap grid (one column per strategy): T
        - Top row: Total responsibility (Summed) over the removal fraction
        - Bottom row: Each subnetwork's own responsibility, masked outside its lifespan. Each subnetwork's lineage is traceable across steps
    """
    modes = list(results.keys())
    n_modes = len(modes)
    risk_col = f'total_{variable_label.lower()}'

    fig, axes = plt.subplots(2, n_modes, figsize=figsize,
                             gridspec_kw={'height_ratios': [0.15, 3], 'hspace': 0.05, 'wspace': 0.3})

    if n_modes == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    global_max = 0
    for mode in modes:
        df = results[mode]['data']
        if df.empty:
            continue
        total_series = df.groupby('fraction')[risk_col].sum()
        if not total_series.empty:
            global_max = max(global_max, total_series.max())
    if global_max == 0:
        global_max = 1

    for idx, mode in enumerate(modes):
        result_data = results[mode]
        df = result_data['data']
        lifecycle = result_data['lifecycle']

        ax_total = axes[0, idx]
        ax_heatmap = axes[1, idx]

        if df.empty:
            continue

        total_sr_by_fraction = df.groupby('fraction')[risk_col].sum()
        total_sr_df = pd.DataFrame([total_sr_by_fraction.values],
                                   columns=total_sr_by_fraction.index, index=['Total'])

        divider_top = make_axes_locatable(ax_total)
        cax_top = divider_top.append_axes("right", size="5%", pad=0.1)

        sns.heatmap(total_sr_df, ax=ax_total, cmap=cmap, cbar=True, cbar_ax=cax_top,
                    linewidths=0.5, linecolor='lightgray', square=False,
                    vmin=0, vmax=global_max, yticklabels=['Total'])

        ax_total.set_title(f'{mode}', fontsize=30, fontweight='bold', pad=20)
        ax_total.set_xlabel('')
        ax_total.set_xticklabels([])
        ax_total.set_yticklabels(ax_total.get_yticklabels(), rotation=0, fontsize=16, fontweight='bold')
        ax_total.collections[0].colorbar.ax.tick_params(labelsize=14)

        pivot_data = df.pivot_table(index='id', columns='fraction', values=risk_col, aggfunc='first')
        all_fractions = sorted(pivot_data.columns)

        for subnetwork_id in pivot_data.index:
            if subnetwork_id not in lifecycle:
                continue
            life = lifecycle[subnetwork_id]
            birth_step, death_step = life['birth_step'], life['death_step']

            birth_rows = df[df['step'] == birth_step]
            birth_fraction = birth_rows['fraction'].iloc[0] if not birth_rows.empty and birth_step > 0 else 0.0

            death_fraction = 1.0
            if death_step is not None:
                death_rows = df[df['step'] == death_step]
                if not death_rows.empty:
                    death_fraction = death_rows['fraction'].iloc[0]
                else:
                    death_fraction = df['fraction'].max()

            for frac in all_fractions:
                if frac < birth_fraction:
                    pivot_data.loc[subnetwork_id, frac] = np.nan
                elif frac > death_fraction:
                    pivot_data.loc[subnetwork_id, frac] = np.nan
                elif pd.isna(pivot_data.loc[subnetwork_id, frac]) or pivot_data.loc[subnetwork_id, frac] == 0:
                    pivot_data.loc[subnetwork_id, frac] = np.nan

        pivot_data = pivot_data.loc[~pivot_data.isna().all(axis=1)].sort_index()
        mask = pivot_data.isna()

        divider_bottom = make_axes_locatable(ax_heatmap)
        cax_bottom = divider_bottom.append_axes("right", size="5%", pad=0.1)

        sns.heatmap(pivot_data, ax=ax_heatmap, cmap=cmap, cbar=True, cbar_ax=cax_bottom, mask=mask,
                    linewidths=0.5, linecolor='lightgray', square=False, vmin=0, vmax=global_max)

        ax_heatmap.set_xlabel('Fraction of Links Removed', fontsize=24, labelpad=15)

        x_tick_labels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        x_tick_positions, final_labels = [], []
        current_cols = pivot_data.columns
        for target_frac in x_tick_labels:
            if len(current_cols) == 0:
                break
            distances = [abs(col - target_frac) for col in current_cols]
            closest_idx = distances.index(min(distances))
            x_tick_positions.append(closest_idx)
            if target_frac == 0.0:
                final_labels.append("0")
            elif target_frac == 1.0:
                final_labels.append("1")
            else:
                final_labels.append(f"{target_frac:.1f}")

        ax_heatmap.set_xticks([pos + 0.5 for pos in x_tick_positions])
        ax_heatmap.set_xticklabels(final_labels, rotation=0, fontsize=16)

        y_labels = [str(int(i)) for i in pivot_data.index]
        ax_heatmap.set_yticks([i + 0.5 for i in range(len(y_labels))])
        ax_heatmap.set_yticklabels(y_labels, rotation=0, fontsize=16)
        ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=20)

        ax_total.tick_params('both', labelsize=20)
        ax_heatmap.tick_params('both', labelsize=20)

    axes[1, 0].set_ylabel('Subnetwork ID', fontsize=24, labelpad=15)
    for col in range(1, n_modes):
        axes[1, col].set_ylabel('')

    plt.tight_layout()
    plt.show()


def plot_total_risk_curve_aggregated(long_df, variable_label, figsize=(8, 6), label_size=16):

    colors = {'Random': 'r', 'Localized': 'b', 'Targeted': 'g'}
    markers = {'Random': 'o', 'Localized': '^', 'Targeted': 's'}
    risk_col = f'total_{variable_label.lower()}'

    fig, ax = plt.subplots(1, figsize=figsize)

    for mode_name in long_df['mode'].unique():
        mode_df = long_df[long_df['mode'] == mode_name]
        sns.lineplot(data=mode_df, x='fraction', y=risk_col,
                    color=colors.get(mode_name, 'k'), marker=markers.get(mode_name, 'o'),
                    markersize=8, label=mode_name, ax=ax)

    ax.yaxis.set_tick_params(labelsize=label_size)
    ax.xaxis.set_tick_params(labelsize=label_size)
    ax.yaxis.set_major_formatter(sigfig_formatter_compact)  
    ax.set_ylabel(f'<Total {variable_label}> [$ft^3/min$]', size=label_size)
    ax.set_xlabel('Fraction of links removed', size=label_size)
    ax.legend(fontsize=label_size - 2)
    plt.tight_layout()
    plt.show()



def animate_percolation(G_pre, variable, mode, mode_name, interval=0.01, seed=42, figsize=(7, 6), fps=4):
    """
    Animate link removal for each strategy in specific frames
    """
    rng = random.Random(seed)
    state = {'G': G_pre.copy()}
    node_positions = nx.get_node_attributes(G_pre, 'pos')
    n_edges_total = G_pre.number_of_edges()

    n_total_edges = len(G_pre.edges(data=variable))
    step_size = max(1, int(interval * n_total_edges))
    num_steps = int(1 / interval) + 2

    mode_color = {'Random': 'crimson', 'Localized': 'mediumblue', 'Targeted': 'forestgreen'}.get(mode_name, 'k')
    progress = {'fraction': 0.0}

    fig, ax = plt.subplots(figsize=figsize)

    def update(frame):
        ax.clear()
        if frame > 0:
            state['G'] = percolate_graph(state['G'], variable=variable, mode=mode, num_sample=step_size,
                                          rng=rng, state=state)
            progress['fraction'] = min(1.0, progress['fraction'] + interval)

        nx.draw_networkx_nodes(G_pre, pos=node_positions, node_size=200,
                               node_color='whitesmoke', edgecolors='k', ax=ax)
        nx.draw_networkx_labels(G_pre, pos=node_positions, font_size=7, ax=ax)
        nx.draw_networkx_edges(G_pre, pos=node_positions, edge_color='lightgray',
                               style='dashed', width=1, arrowstyle='-', ax=ax)
        if state['G'].number_of_edges() > 0:
            nx.draw_networkx_edges(state['G'], pos=node_positions, edge_color=mode_color,
                                   width=2, arrowstyle='->', connectionstyle='arc3,rad=0.3', ax=ax)

        ax.set_title(f"{mode_name} removal -- {progress['fraction']*100:.0f}% of links removed "
                    f"({state['G'].number_of_edges()}/{n_edges_total} remaining)", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

    anim = FuncAnimation(fig, update, frames=num_steps, interval=1000 / fps, repeat=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def plot_strategy_comparison(G_pre, variable, checkpoints=(0.0, 0.33, 0.66, 1.0),
                              interval=0.05, seed=42, cell_size=(4, 4), save=False):
    """
    Plot link removal for each strategy in specific snapshots at defined intervals (kinda like "show your work")
    """
    modes = {0: 'Random', 1: 'Localized', 2: 'Targeted'}
    mode_colors = {'Random': 'crimson', 'Localized': 'mediumblue', 'Targeted': 'forestgreen'}
    panel_labels = ['A', 'B', 'C']
    node_positions = nx.get_node_attributes(G_pre, 'pos')

    global_total = np.nansum(list(nx.get_edge_attributes(G_pre, variable).values()))

    fig, axes = plt.subplots(len(modes), len(checkpoints),
                              figsize=(cell_size[0] * len(checkpoints), cell_size[1] * len(modes)))

    for row, (mode_code, mode_name) in enumerate(modes.items()):
        rng = random.Random(seed)
        state = {}
        G_curr = G_pre.copy()

        n_total_edges = len(G_curr.edges(data=variable))
        step_size = max(1, int(interval * n_total_edges))
        num_steps = int(1 / interval) + 1

        snapshots = {0.0: G_curr.copy()}
        current_fraction = 0.0
        for step in range(1, num_steps):
            current_fraction += interval
            if current_fraction > 1.0:
                break
            G_curr = percolate_graph(G_curr, variable=variable, mode=mode_code,
                                     num_sample=step_size, rng=rng, state=state)
            snapshots[round(current_fraction, 4)] = G_curr.copy()

        snap_fractions = sorted(snapshots.keys())

        for col, target_frac in enumerate(checkpoints):
            ax = axes[row, col]
            closest = min(snap_fractions, key=lambda f: abs(f - target_frac))
            G_snap = snapshots[closest]

            nx.draw_networkx_nodes(G_pre, pos=node_positions, node_size=50,
                                   node_color='whitesmoke', edgecolors='k', linewidths=0.4, ax=ax)
            nx.draw_networkx_edges(G_pre, pos=node_positions, edge_color='lightgray',
                                   style='dashed', width=0.6, arrowstyle='-', ax=ax)
            if G_snap.number_of_edges() > 0:
                nx.draw_networkx_edges(G_snap, pos=node_positions, edge_color=mode_colors[mode_name],
                                       width=1.5, arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                       arrowsize=6, ax=ax)

            snap_weights = list(nx.get_edge_attributes(G_snap, variable).values())
            total_responsibility = np.nansum(snap_weights) if snap_weights else 0.0
            pct_of_whole = (total_responsibility / global_total * 100) if global_total > 0 else 0.0

            ax.set_title(f"{closest*100:.0f}% of links removed\nTotal {variable} = {pct_of_whole:.1f}%",
                        fontsize=14, pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('gainsboro')

        axes[row, 0].text(-0.1, 1.15, f"$\\bf{{{panel_labels[row]}}}$", transform=axes[row, 0].transAxes,
                          fontsize=24, ha='left', va='top')
        axes[row, 0].set_ylabel(mode_name, fontsize=18, fontweight='bold',
                                color=mode_colors[mode_name], labelpad=12)

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()



def plot_subnetwork_totals(comp_rank_sr, comp_rank_or, label_size=18, title_size=24, figsize=(14, 9), save=False):
    """
    Plot total and average SR/OR per subnetwork grouped by subnetwork size (# of houses) 
    """

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    panel_labels = ['A', 'B', 'C', 'D']

    num_sr = np.unique(comp_rank_sr['n_nodes'])
    num_or = np.unique(comp_rank_or['n_nodes'])

    sns.barplot(data=comp_rank_sr, x='n_nodes', y='total_risk', palette='Blues_r', edgecolor='k',
                errorbar=("se"), capsize=0.15, order=num_sr[::-1], ax=axs[0, 0])
    axs[0, 0].set_xlabel("# of houses", size=label_size)
    axs[0, 0].set_ylabel("Total SR [$ft^3/min$]", size=label_size)
    axs[0, 0].set_xticklabels([str(n) for n in num_sr[::-1]], fontsize=label_size)
    axs[0, 0].yaxis.set_major_formatter(sigfig_formatter_compact)
    axs[0, 0].tick_params(axis='both', labelsize=label_size)

    sns.barplot(data=comp_rank_or, x='n_nodes', y='total_risk', palette='Wistia_r', edgecolor='k', width=0.5,
                errorbar=("se"), capsize=0.15, order=num_or[::-1], ax=axs[0, 1])
    axs[0, 1].set_xlabel("# of houses", size=label_size)
    axs[0, 1].set_ylabel("Total OR [$ft^3/min$]", size=label_size)
    axs[0, 1].set_xticklabels([str(n) for n in num_or[::-1]], fontsize=label_size)
    axs[0, 1].yaxis.set_major_formatter(sigfig_formatter_compact)
    axs[0, 1].tick_params(axis='both', labelsize=label_size)

    sns.barplot(data=comp_rank_sr, x='n_nodes', y='avg_risk', palette='Reds_r', edgecolor='k',
                order=num_sr[::-1], ax=axs[1, 0])
    axs[1, 0].set_xlabel("# of houses", size=label_size)
    axs[1, 0].set_ylabel("Avg SR [$ft^3/min$]", size=label_size)
    axs[1, 0].set_xticklabels([str(n) for n in num_sr[::-1]], fontsize=label_size)
    axs[1, 0].yaxis.set_major_formatter(sigfig_formatter_compact)
    axs[1, 0].tick_params(axis='both', labelsize=label_size)

    sns.barplot(data=comp_rank_or, x='n_nodes', y='avg_risk', palette='Greens', edgecolor='k',
                order=num_or[::-1], ax=axs[1, 1])
    axs[1, 1].set_xlabel("# of houses", size=label_size)
    axs[1, 1].set_ylabel("Avg OR [$ft^3/min$]", size=label_size)
    axs[1, 1].set_xticklabels([str(n) for n in num_or[::-1]], fontsize=label_size)
    axs[1, 1].yaxis.set_major_formatter(sigfig_formatter_compact)
    axs[1, 1].tick_params(axis='both', labelsize=label_size)

    for ax, label in zip(axs.flat, panel_labels):
        ax.set_title(rf"$\bf{{{label}}}$ ", size=title_size, loc='left')

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()


def plot_SR_example(gdf_results, number=None, one_example=True, save=False):

    for num in gdf_results[gdf_results['SR_avg'].notna()].index:

        if number:
            num = number

        fig, ax = plt.subplots(1)

        DS_30 = gpd.GeoDataFrame([gdf_results.iloc[num]]).set_crs(BASE_CRS).to_crs(GEO_CRS)
        tax_polygon = gpd.GeoDataFrame([gdf_results.iloc[num]], geometry='tax_geometry').set_crs(BASE_CRS).to_crs(GEO_CRS)

        gpd.GeoDataFrame([gdf_results.iloc[num]]).set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
            ax=ax, color='red', alpha=0.5, edgecolor='k', label="Owner's DS30")
        gpd.GeoDataFrame([gdf_results.iloc[num]], geometry='tax_geometry').set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
            ax=ax, alpha=0.5, color='grey', hatch='.', edgecolor='red', label="Owner's Tax Parcel")
        gpd.GeoDataFrame([gdf_results.iloc[num]], geometry='bldg_geometry').set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
            ax=ax, color='k', label="Owner's Building")

        overlaps = []
        overlap_proportions = []
        overlaps_tax_neighbor = []
        overlaps_tax_owner = []

        o_centroid = gpd.GeoDataFrame([gdf_results.iloc[num]], geometry='bldg_geometry').set_crs(BASE_CRS).to_crs(GEO_CRS).centroid
        ax.annotate(f"Owner{num}", xy=(o_centroid.x, o_centroid.y), xytext=(0, 0),
                    textcoords="offset points", ha='center', va='center', fontsize=8, color='white')

        for idx in gdf_results.iloc[num]['SR_ID']:

            gpd.GeoDataFrame([gdf_results.iloc[idx]]).set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
                ax=ax, color='none', edgecolor='k', linestyle='--', label="Neighbor{}'s DS30".format(idx))
            gpd.GeoDataFrame([gdf_results.iloc[idx]], geometry='tax_geometry').set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
                ax=ax, alpha=0.5, edgecolor='k', color='lightgray', label="Neighbor{}'s Tax Parcel".format(idx))
            gpd.GeoDataFrame([gdf_results.iloc[idx]], geometry='bldg_geometry').set_crs(BASE_CRS).to_crs(GEO_CRS).plot(
                ax=ax, alpha=0.5, color='k', label="Neighbor{}'s Building".format(idx))

            centroid = gpd.GeoDataFrame([gdf_results.iloc[idx]], geometry='bldg_geometry').set_crs(BASE_CRS).to_crs(GEO_CRS).centroid
            ax.annotate(f"N{idx}", xy=(centroid.x, centroid.y), xytext=(0, 0),
                        textcoords="offset points", ha='center', va='center', fontsize=8, color='white')

            other_DS_30 = gpd.GeoDataFrame([gdf_results.iloc[idx]]).set_crs(BASE_CRS).to_crs(GEO_CRS)
            other_tax = gpd.GeoDataFrame([gdf_results.iloc[idx]], geometry='tax_geometry').set_crs(BASE_CRS).to_crs(GEO_CRS)

            overlap = DS_30.intersection(other_DS_30, align=False)
            overlap_tax_owner = overlap.intersection(tax_polygon, align=False)
            overlap_tax_neighbor = overlap.intersection(other_tax, align=False)

            if overlap_tax_neighbor.values[0]:
                overlap_tax_neighbor.plot(ax=ax, color='royalblue', hatch='\\\\', alpha=0.5)
            if overlap_tax_owner.values[0]:
                overlap_tax_owner.plot(ax=ax, color='yellow', hatch='////', alpha=0.5)

            overlaps.append(overlap.values[0])
            overlaps_tax_neighbor.append(overlap_tax_neighbor.area.values[0])
            overlaps_tax_owner.append(overlap_tax_owner.area.values[0])
            overlap_proportions.append(overlap.area / DS_30.area)

        legend_elements = [
            Patch(facecolor='grey', alpha=0.5, edgecolor='red', hatch='.', linewidth=1, label="Owner{}'s Tax Parcel".format(str(num))),
            Patch(facecolor='red', alpha=0.5, edgecolor='k', linewidth=1, label="Owner{}'s DS30".format(str(num))),
            Patch(facecolor='k', linewidth=2, label="Owner{}'s Building".format(str(num))),
            Patch(facecolor='lightgray', alpha=0.5, edgecolor='k', linewidth=1, label="Neighbor's Tax Parcel"),
            Patch(facecolor='white', alpha=0.5, edgecolor='k', linewidth=1, linestyle='--', label="Neighbor's DS30"),
            Patch(facecolor='gray', label="Neighbor's Building"),
            Patch(facecolor='yellow', hatch='////', alpha=0.5, edgecolor='k', linewidth=1, label="Owner's SR"),
            Patch(facecolor='royalblue', hatch='\\\\', alpha=0.5, edgecolor='k', linewidth=1, label="Neighbors' SR"),
        ]

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_xticklabels([]);ax.set_yticklabels([])

        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')

        plt.show()

        if one_example:
            break



def plot_OR_example(gdf_results, number=None, one_example=True, save=False):

    for num in gdf_results[gdf_results['OR_avg'].notna()].index:

        if number:
            num = number
        
        fig,ax=plt.subplots(1)

        ds30 = gpd.GeoDataFrame([gdf_results.iloc[num]]).set_crs('epsg:3857').to_crs('epsg:4326')
        ds30_parcel = gpd.GeoDataFrame([gdf_results.iloc[num]], geometry='tax_geometry').set_crs('epsg:3857').to_crs('epsg:4326')
        bldg_owner = gpd.GeoDataFrame([gdf_results.iloc[num]], geometry='bldg_geometry')
        
        ds30.plot(ax=ax, color='red', alpha=0.5, edgecolor='k', label="D_{}".format(str(num)))
        ds30_parcel.plot(ax=ax, alpha=0.5, color='grey', label="Owner's Tax Parcel")
        bldg_owner.set_crs('epsg:3857').to_crs('epsg:4326').plot(ax=ax, color='k', label="Owner's Building")

        # Annotate the centroid of owner
        o_centroid = gpd.GeoDataFrame([gdf_results.iloc[num]], geometry='bldg_geometry').set_crs('epsg:3857').to_crs('epsg:4326').centroid
        ax.annotate(
            f"Owner{num}",
            xy=(o_centroid.x, o_centroid.y),
            xytext=(0, 0),
            textcoords="offset points",
            ha='center',
            va='center',
            fontsize=8,
            color='white'
        )

        # Find actual spatial neighbors
        owner_ds30 = gdf_results.iloc[num].geometry
        owner_parcel = gdf_results.iloc[num]['tax_geometry']
        
        spatial_neighbors = []
        
        for idx in gdf_results.index:
            if idx == num: continue
                
            neighbor_ds30 = gdf_results.iloc[idx].geometry
            neighbor_parcel = gdf_results.iloc[idx]['tax_geometry']
            
            is_neighbor = (
                owner_ds30.intersects(neighbor_ds30) or
                owner_ds30.intersects(neighbor_parcel) or
                owner_parcel.intersects(neighbor_ds30) or
                owner_parcel.intersects(neighbor_parcel)
            )
            
            if is_neighbor:
                spatial_neighbors.append(idx)
        
        if not spatial_neighbors:
            ax.text(0.5, 0.5, f"No spatial neighbors found for Owner {num}", 
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            plt.show()
            if one_example:
                break
            continue
        

        for idx in spatial_neighbors:

            # Neighbor geometry and plot
            ds30_n = gpd.GeoDataFrame([gdf_results.iloc[idx]]).set_crs('epsg:3857').to_crs('epsg:4326')
            ds30_parcel_n = gpd.GeoDataFrame([gdf_results.iloc[idx]], geometry='tax_geometry').set_crs('epsg:3857').to_crs('epsg:4326')
            bldg_n = gpd.GeoDataFrame([gdf_results.iloc[idx]], geometry='bldg_geometry').set_crs('epsg:3857')
            ds30_n.plot(ax=ax, color='none', edgecolor='k', linestyle='--', label="$D_{}$".format(idx))
            ds30_parcel_n.plot(ax=ax, alpha=0.5, edgecolor='k', color='lightgray', label="$t_{}$".format(idx))
            bldg_n.to_crs('epsg:4326').plot(ax=ax, alpha=0.5, color='k', label="$h_{}$".format(idx))
            centroid = gpd.GeoDataFrame([gdf_results.iloc[idx]], geometry='bldg_geometry').set_crs('epsg:3857').to_crs('epsg:4326').centroid
            ax.annotate(
                f"N{idx}",
                xy=(centroid.x, centroid.y),
                xytext=(0, 0), textcoords="offset points",
                ha='center', va='center', fontsize=8, color='white'
            )

            # Compute OR
            owner_ds30_geom = ds30.geometry.iloc[0]       # D_i
            neighbor_ds30_geom = ds30_n.geometry.iloc[0]  # D_j
            
            neighbor_parcel_geom = ds30_parcel_n.geometry.iloc[0] # t_j
            owner_parcel_geom = ds30_parcel.geometry.iloc[0]      # t_i

            # Whoever's propoerty risk lies on owes the mitigation
            intersection_i_tj = owner_ds30_geom.intersection(neighbor_parcel_geom)

            if not intersection_i_tj.is_empty:
                or_shape = intersection_i_tj.difference(neighbor_ds30_geom)

                if not or_shape.is_empty:
                    gpd.GeoDataFrame([1], geometry=[or_shape], crs='epsg:4326').plot(
                        ax=ax, color='yellow', hatch='////', alpha=0.7, label=f"N{idx}'s OR on Owner{num}"
                    )

            # (D_j n t_i) \ D_i: neighbor's DS on owner's own parcel but outside DS (Owner owes neighbor)
            intersection_j_ti = neighbor_ds30_geom.intersection(owner_parcel_geom)

            if not intersection_j_ti.is_empty:
                or_shape_n = intersection_j_ti.difference(owner_ds30_geom)

                if not or_shape_n.is_empty:
                    gpd.GeoDataFrame([1], geometry=[or_shape_n], crs='epsg:4326').plot(
                        ax=ax, color='royalblue', hatch='\\\\', alpha=0.7, label=f"Owner{num}'s OR on N{idx}"
                    )

        # Legend formatting
        legend_elements = [
            Patch(facecolor='gray', label="Building"),
            Patch(facecolor='lightgray', alpha=0.5, edgecolor='k', linewidth=1, label="Parcel"),
            Patch(facecolor='k', linewidth=2, label=rf"$h_{{{num}}}$"),
            Patch(facecolor='red', alpha=0.5, edgecolor='k', linewidth=1, label=rf"$D_{{{num}}}$"),
            Patch(facecolor='yellow', hatch='////', alpha=0.5, edgecolor='k', linewidth=1, label=r"$OR_{Neighbor}$"),    
            Patch(facecolor='white', alpha=0.5, edgecolor='k', linewidth=1, linestyle='--', label=r"$D_{Neighbor}$"),        
            Patch(facecolor='royalblue', hatch='\\\\', alpha=0.5, edgecolor='k', linewidth=1, label=rf"$OR_{{{num}}}$")
        ]    
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.9), loc='upper left', fontsize=16)
        ax.set_xticklabels([]);ax.set_yticklabels([])
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        
        plt.show()

        if one_example:
            break

def plot_network_circular(G, node_attribute='SR', node_color_map='gray_r', edge_alpha=0.25,
                           node_size_factor=1.0, show_labels=True, highlight_node=None,
                           figsize=(12, 12), label_size=24, save=False):
    """
    Node color set by avg of each node's SR or OR links with node size scaled by degree
    """
    connected_nodes = set()
    for edge in G.edges():
        connected_nodes.add(edge[0])
        connected_nodes.add(edge[1])

    nodes = sorted(connected_nodes)
    n_nodes = len(nodes)

    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    positions = {node: (np.cos(angles[i]), np.sin(angles[i])) for i, node in enumerate(nodes)}

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    edge_view = G.in_edges if node_attribute == 'OR' else G.out_edges

    all_edge_attribute_values = [data[node_attribute] for u, v, data in G.edges(data=True) if node_attribute in data]
    min_edge_attr_val = min(all_edge_attribute_values) if all_edge_attribute_values else 0
    max_edge_attr_val = max(all_edge_attribute_values) if all_edge_attribute_values else 0

    node_color_attribute_values = []
    for node in nodes:
        own_edges = [data[node_attribute] for u, v, data in edge_view(node, data=True) if node_attribute in data]
        node_color_attribute_values.append(np.mean(own_edges) if own_edges else min_edge_attr_val)

    if node_color_attribute_values and max_edge_attr_val > min_edge_attr_val:
        norm_node_color_values = [
            (val - min_edge_attr_val) / (max_edge_attr_val - min_edge_attr_val)
            for val in node_color_attribute_values
        ]
    elif node_color_attribute_values:
        norm_node_color_values = [0.5] * len(node_color_attribute_values)
    else:
        norm_node_color_values = [0.5] * len(nodes)

    cmap = plt.get_cmap(node_color_map)
    node_colors = [cmap(val) for val in norm_node_color_values]

    highlight_edges = [edge for edge in G.edges() if edge[0] == highlight_node or edge[1] == highlight_node]

    # Plot network links
    for edge in G.edges():
        start_node, end_node = edge
        start_pos = positions[start_node]
        end_pos = positions[end_node]
        is_highlight_edge = edge in highlight_edges

        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        control_factor = 0.3
        control_x = mid_x * control_factor
        control_y = mid_y * control_factor

        t = np.linspace(0, 1, 100)
        curve_x = (1 - t) ** 2 * start_pos[0] + 2 * (1 - t) * t * control_x + t ** 2 * end_pos[0]
        curve_y = (1 - t) ** 2 * start_pos[1] + 2 * (1 - t) * t * control_y + t ** 2 * end_pos[1]

        edge_weight = G.edges[edge].get('weight', 1)
        if is_highlight_edge:
            line_width = max(2.0, edge_weight * 3)
            alpha = 0.9
        else:
            line_width = max(0.5, edge_weight * 2)
            alpha = edge_alpha

        start_idx = nodes.index(start_node)
        source_node_color = node_colors[start_idx]
        edge_color = 'red' if is_highlight_edge else source_node_color

        ax.plot(curve_x, curve_y, color=edge_color, alpha=alpha, linewidth=line_width,
                solid_capstyle='round', zorder=2 if is_highlight_edge else 1)

        arrow_pos = 0.8
        arrow_idx = int(arrow_pos * (len(curve_x) - 1))
        arrow_x, arrow_y = curve_x[arrow_idx], curve_y[arrow_idx]
        if arrow_idx < len(curve_x) - 1:
            dx = curve_x[arrow_idx + 1] - curve_x[arrow_idx]
            dy = curve_y[arrow_idx + 1] - curve_y[arrow_idx]
            arrow_size = 0.02 * node_size_factor
            if is_highlight_edge:
                arrow_size *= 100
            ax.annotate('', xy=(arrow_x + dx * arrow_size, arrow_y + dy * arrow_size), xytext=(arrow_x, arrow_y),
                        arrowprops=dict(arrowstyle='->', color=edge_color, alpha=alpha, lw=line_width),
                        zorder=2 if is_highlight_edge else 1)

    if show_labels:
        for node in nodes:
            pos = positions[node]
            label_distance = 1.03
            label_x = pos[0] * label_distance
            label_y = pos[1] * label_distance
            ha = 'left' if label_x > 0.1 else ('right' if label_x < -0.1 else 'center')
            va = 'bottom' if label_y > 0.1 else ('top' if label_y < -0.1 else 'center')
            if node == highlight_node:
                ax.text(label_x, label_y, str(node), fontsize=16, ha=ha, va=va, fontweight='bold', color='red')
            else:
                ax.text(label_x, label_y, str(node), fontsize=16, ha=ha, va=va)

    node_positions_x = [positions[node][0] for node in nodes]
    node_positions_y = [positions[node][1] for node in nodes]
    ax.scatter(node_positions_x, node_positions_y, s=300, c=node_colors, edgecolors='black', linewidths=2, zorder=5)

    if highlight_node in nodes:
        highlight_idx = nodes.index(highlight_node)
        ax.scatter(node_positions_x[highlight_idx], node_positions_y[highlight_idx],
                   s=300, c=[node_colors[highlight_idx]], edgecolors='red', linewidths=4, zorder=10)

    if node_attribute and all_edge_attribute_values:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=max_edge_attr_val))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.25, pad=-0.08, format=sigfig_formatter_compact)
        cbar.set_label(f'{node_attribute} [$ft^3/min$]', rotation=90, labelpad=15, size=label_size)
        cbar.ax.tick_params(labelsize=label_size)

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
