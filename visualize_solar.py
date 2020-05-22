# MIT License

# Copyright (c) 2016 Stefan Zapf, Christopher Kraushaar

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Load libraries
import math
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

import matplotlib.pyplot as plt
import numpy as np
import sys
import csv
import warnings
import argparse
import math

from adjustText import adjust_text

warnings.filterwarnings("ignore")

zorder_label = 10
zorder_planet = 9
zorder_moonorbit = 7
zorder_annotations = 8
zorder_orbit = 6

def get_angles(orbit, number_of_datapoints, ax, full_circle = True):
    start = (2 * math.pi) / 11 * (orbit - 1)
    stop = start + (math.pi * (1.8 if full_circle and number_of_datapoints > 2 else 1))
    return np.linspace(start, stop, number_of_datapoints, endpoint=True)


def get_y(angle, radius):
    return math.sin(angle) * radius


def get_x(angle, radius):
    return math.cos(angle) * radius


def label_to_idx(labels, label):
    center_idx_bool = labels == label
    return np.asscalar(np.where(center_idx_bool)[0]), center_idx_bool


def transform_to_correlation_dist(data):
    y_corr = np.corrcoef(data.T)
    # we just need the magnitude of the correlation and don't care whether it's positive or not
    abs_corr = np.abs(y_corr)
    return np.nan_to_num(abs_corr)


def transform_to_positive_corrs(data, sun_idx):
    y_corr = np.corrcoef(data.T)
    positive = y_corr[sun_idx]
    positive = positive >= 0
    return positive

def transform_to_positive(y_corr, sun_idx):
    positive = y_corr[sun_idx]
    positive = positive >= 0
    return positive



def solar_corr(data, labels, center, ax=False, moon_orbit=0.25, base_circle_size=50, calc_corr=True, orbits=10, show_window=True, image_path="solar.png",
               save_png=True, title="Solar Correlation Map", selected=[], show_labels=True):
    labels = np.array(labels)
    center_idx, center_idx_bool = label_to_idx(labels, center)
    plot_idx = 23
    all_idx = np.logical_not(center_idx_bool)
    positive = transform_to_positive_corrs(data, center_idx) if calc_corr else transform_to_positive(data, center_idx)
    corr_dist = transform_to_correlation_dist(data) if calc_corr else data
    sun_corr_dist = corr_dist[center_idx]
    colors = np.linspace(0, 1, num=len(sun_corr_dist))
    cordinate_to_correlation = {}
    step = 1.0 / orbits
    last_orbit = 0.1

    fig = plt.gcf()
    if not ax:
        fig.set_size_inches(20, 20)
    labels_idx = np.array([center_idx])

    # matplotlib.rcParams.update({'font.size': 14})

    all_labels = []
    all_circles = []

    color_map = plt.get_cmap("Paired")
    color_map_circle = plt.get_cmap("Greys", 10)
    color_map_tab10 = plt.get_cmap("tab20")

    if not ax:
        ax = fig.add_subplot(111)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    if not ax:
        ax.set_position([0.01, 0.01, 0.98, 0.98])  # set a new position

    # place sun:
    all_circles.append(ax.scatter(0, 0, color=color_map_circle(7), s=base_circle_size*2, label=labels[center_idx], zorder=zorder_planet))
    # ax.text(0, 0.25, str(labels[center_idx]), verticalalignment="center", horizontalalignment='center', color=color_map_circle(7), zorder=zorder_label)

    last_orbit_had_values = False
    min_orbit = 0
    for orbit in range(1, orbits + 1):
        new_orbit = step * orbit + 0.1
        idx = (sun_corr_dist >= (1 - last_orbit)) & (sun_corr_dist > (1 - new_orbit)) & all_idx
        idx_int = np.where(idx)[0]

        last_orbit = new_orbit
        
        if (sum(idx) == 0 and not last_orbit_had_values):
            min_orbit = orbit
            continue
        else:
            last_orbit_had_values = True

        orbit = orbit - min_orbit

        corr_dists = []
        for index in idx_int:
            corr_dists = np.append(corr_dists, corr_dist[center_idx][index])
        sort_args = np.argsort(corr_dists)
        idx_int = idx_int[sort_args]

        labels_idx = np.append(labels_idx, idx_int)
        planets = sum(idx)
        angles = get_angles(orbit, planets, ax, True)

        print(orbit)
        print(angles)
        print("IDX: " + str(sum(idx)) + " current orbit " + str(new_orbit) + " " + "last orbit: " + str(last_orbit))
        last_idx = -1
        # place planets
        while np.any(idx):
            idx_int = np.where(idx)[0]

            remaining = sum(idx)
            # current_planet = planets - remaining
            current_planet = 0
            print(current_planet)
            # breakpoint()
            current_idx = idx_int[current_planet]
            angle = angles[planets - remaining]
            x = get_x(angle, orbit)
            y = get_y(angle, orbit)
            # current_idx = idx_int[current_planet]
            color = colors[current_idx]
            # plt.scatter(x, y, color=color_map(color), s=100, label=labels[current_idx])
            cordinate_to_correlation[(x, y)] = {"is_planet": True, "corr_sun": corr_dist[center_idx, current_idx] }

            planet_idx = current_idx

            planet_corr = corr_dist[planet_idx]

            # ax.text(x-0.35, y+0.2, "{0:.3f}".format(planet_corr[center_idx]))
            col = "#03C03C" if positive[planet_idx] else "#FF6961"
            if orbit == orbits:
                col = "grey"
            # ax.text(x + 0.15, y + 0.15, str(labels[planet_idx]), verticalalignment="bottom", horizontalalignment='left',
            #        color=col, fontsize='small')
            moon_idx = (planet_corr >= 0.65) & idx
            moon_idx_int = np.where(moon_idx)[0]
            moons = sum(moon_idx)
            moon_angles = get_angles(0, moons, ax, False)

            # add orbit around planet if it has moons
            if moons > 1:
                circle = plt.Circle((x, y), moon_orbit, color='lightgrey', alpha=0.8, fill=True, zorder=zorder_moonorbit)
                ax.add_artist(circle)
                all_circles.append(circle)
            else:
                pointcol = color_map_tab10(2) if labels[planet_idx] in selected else color_map_tab10(0)
                labelcol = color_map_tab10(2) if labels[planet_idx] in selected else color_map_tab10(0)

                idx[planet_idx] = False
                all_idx[planet_idx] = False
                all_circles.append(ax.scatter(x, y, color=pointcol, s=base_circle_size, label=labels[current_idx], zorder=zorder_planet))
                
                # all_labels.append(ax.text(x + 0.15 if x > 0 else x - 0.15, y + 0.15 if y > 0 else y - 0.15, str(labels[planet_idx]), verticalalignment="center", horizontalalignment='right' if x < 0 else 'left',
                #    color=col, fontsize='small'))
                if show_labels:
                    all_labels.append(ax.text(x, y, str(labels[planet_idx]), verticalalignment="center", horizontalalignment='right',
                        color=labelcol, zorder=zorder_label, path_effects=[path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()]))
                continue

            if (current_idx != last_idx):
                print(labels[planet_idx])
                print("Drawing Moons " + str(current_idx) + " moons " + str(sum(moon_idx)))

            last_idx = current_idx
            
            while np.any(moon_idx):
                remaining_moons = sum(moon_idx)
                current_moon = moons - remaining_moons
                current_moon_idx = moon_idx_int[current_moon]
                angle = moon_angles[current_moon]
                m_x = get_x(angle, moon_orbit) + x
                m_y = get_y(angle, moon_orbit) + y

                pointcol = color_map_tab10(2) if labels[current_moon_idx] in selected else color_map_tab10(0)
                labelcol = color_map_tab10(2) if labels[planet_idx] in selected else color_map_tab10(0)

                all_circles.append(ax.scatter(m_x, m_y, color=pointcol, s=base_circle_size, label=labels[current_moon_idx], zorder=zorder_planet))
                cordinate_to_correlation[(m_x, m_y)] = {"is_planet": False, "corr_sun": corr_dist[center_idx][current_moon_idx]}
                col = "#03C03C" if positive[current_moon_idx] else "#FF6961"
                if orbit == orbits:
                    col = "grey"
                #all_labels.append(ax.text(m_x + 0.1 if m_x > x else m_x - 0.1, m_y + 0.1 if m_y > y else m_y - 0.1, str(labels[current_moon_idx]), verticalalignment="center",
                #        horizontalalignment='right' if m_x < x else 'left', fontsize='small', color=col))
                if show_labels:
                    all_labels.append(ax.text(m_x, m_y, str(labels[current_moon_idx]), verticalalignment="center",
                        horizontalalignment='right' if m_x < x else 'left', color=labelcol, zorder=zorder_label, path_effects=[path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()]))
                moon_idx[current_moon_idx] = False
                idx[current_moon_idx] = False
                all_idx[current_moon_idx] = False

        # last_orbit = new_orbit

        print("Drawing circle")
        circlecol = color_map_circle((1 - ((orbit - 1) * step)) / 2)
        circle = plt.Circle((0, 0), orbit, color=circlecol, fill=False, zorder=zorder_orbit)

        ax.add_artist(circle)

        if 1 - float(orbit + min_orbit) / 10 > 0:
            orbit_t = ax.text(0, -orbit + 0.1, "{0:.1f}".format(1 - float(orbit + min_orbit) / 10), verticalalignment="bottom",
                    horizontalalignment="center", color=circlecol, zorder=zorder_orbit)
            all_circles.append(orbit_t)

    labels_pos = np.array(labels)[labels_idx]
    recs = []

    # ax = plt.gca()

    if not ax:
        ax.axis("equal")
    # ax.margins(x=-0.3, y=-0.3)
    # ax.axis([-5, 5, -5, 5])
    # plt.axis([-10, 10, -10, 10])

    legend_elements = [
                    Line2D([0], [0], linestyle='none', marker='o', color=color_map_tab10(0), label='candidate',
                                markersize=10),
                    Line2D([0], [0], linestyle='none', marker='o', color=color_map_circle(7), label='document',
                                markersize=10)
    ]
    if len(selected) > 0:
        legend_elements.append(
            Line2D([0], [0], linestyle='none', marker='o', color=color_map_tab10(2), label='selected candidate',
                    markersize=10)
        )
    ax.legend(handles=legend_elements, loc='upper left').set_zorder(102)

    # plt.subplots_adjust(top=0.95)

    if save_png:
        plt.savefig(image_path)
    if show_window:
        # only require mplcursors if we need an interactive plot
        # import mplcursors
        # cooordinate_to_correlation[(sel.target.x, sel.target.y)]["corr_sun"])
        # cursors = mplcursors.cursor(hover=True)
        # @cursors.connect("add")
        def _(sel):
            sel.annotation.set(position=(15, -15))
            # Note: Needs to be set separately due to matplotlib/matplotlib#8956.
            sel.annotation.get_bbox_patch().set(fc="lightgrey")
            sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white", alpha=0)
            sel.annotation.set_text("Correlation to sun \n{}".format(cordinate_to_correlation[ (sel.target[0],sel.target[1])]["corr_sun"]))

        if show_labels:
            adjust_text(all_labels, add_objects=all_circles, va="center", ha="left", ax=ax, zorder=zorder_annotations, horizontalalignment='left', verticalalignment='center', arrowprops=dict(connectionstyle="angle3", arrowstyle='-', color='darkgrey'))

            # for item in all_labels:
            #    item.set_fontsize(10)
                
        if not ax:
            plt.show()

def main(input_csv, sun, image_path, title):
    # Load data
    data = np.genfromtxt(input_csv, delimiter=",", skip_header=1)
    labels = csv.DictReader(open(input_csv), skipinitialspace=True).fieldnames
    solar_corr(data, labels, sun, image_path=image_path, title=title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a solar correlation map')
    parser.add_argument("csv_file", type=str)
    parser.add_argument("sun_variable", type=str)
    parser.add_argument("image_file_name",  type=str, nargs="?", default="solar.png")
    parser.add_argument("--title", nargs="?", default=None)
    args = parser.parse_args()
    if args.title is None:
        args.title = "Solar Correlation Map for '{}' ".format(args.sun_variable)
    main(args.csv_file, args.sun_variable, args.image_file_name, args.title)