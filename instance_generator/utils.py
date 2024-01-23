import geopandas as gp
import os.path
import matplotlib.pyplot as plt
import numpy as np
from millify import millify
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

from city_type import CityType


BERLIN_REGIONS = [
    ['10318', '10319', '10317', '10315', '10365', '10367', '10369', '10247', '10249', '10245'],
    ['10407', '10405', '10409', '10437', '10439', '10435', '10119', '10115', '10178', '10179', '10117', '10115', '10557'],
    ['10243', '10997', '10999', '10967', '10965', '10961', '10969', '10963', '10829', '10827', '10823', '10781', '10783', '10785', '10787']
]

FRANKFURT_REGIONS = [
    ['60529', '60528', '60598', '60599', '60596', '60327', '60594', '60326', '60486'],
    ['60489', '60488', '60439', '60431', '60487', '60438', '60433', '60437'],
    ['60329', '60311', '60313', '60325', '60323', '60316', '60385', '60314', '60318', '60322']
]

PARIS_REGIONS = [
    ['75012', '75011', '75020', '75004', '75003'],
    ['75016', '75008', '75017', '75007'],
    ['75013', '75014', '75006', '75005', '75015']
]

LYON_REGIONS = [
    ['69800', '69200', '69500', '69008'],
    ['69002', '69005', '69007', '69003'],
    ['69004', '69001', '69009', '69006', '69300']
]


def add_regions(gdf: gp.GeoDataFrame, city_type: CityType) -> None:
    """ Partitions the zip code areas into hard-coded regions.

        It adds two new columns to :param:`gdf`. The first one is "region",
        which will contain the identifier of the region to which the area
        belongs. The second one is "region_pop" which contains the value of
        the total population of the region. (Areas which belong to the same
        region will have the same values for both "region" and "region_pop").

        Parameters:
            - gdf (gp.GeoDataFrame):
                The source dataframe.
            - city_type (.generator.CityType):
                The city the dataframe refers to.
    """

    if city_type == CityType.BERLIN:
        hc_regions = BERLIN_REGIONS
    elif city_type == CityType.FRANKFURT:
        hc_regions = FRANKFURT_REGIONS
    elif city_type == CityType.PARIS:
        hc_regions = PARIS_REGIONS
    else:
        hc_regions = LYON_REGIONS

    def get_region_idx(postal_code: str) -> str:
        try:
            return next(i for i, l in enumerate(hc_regions) if str(postal_code) in l)
        except StopIteration:
            return len(hc_regions)
        
    gdf['region'] = gdf.postal_code.apply(get_region_idx)
    gdf['region_pop'] = gdf.population.groupby(gdf.region).transform('sum')


def plot_city_areas(gdf: gp.GeoDataFrame, city_name: str) -> Path:
    """ Plots a city's areas with their population to a file,
        and returns the corresponding path.
        
        Parameters:
            - gdf (gp.GeoDataFrame):
                Dataframe containing city info.
            - city_name (str):
                City name to use when creating the figure file.

        The figure file will be `output/<city_name>_areas.png`.
    """

    figfile = Path(os.path.join(
        os.path.realpath(os.path.dirname(__file__)),
        '..',
        'output',
        f"{city_name.lower()}_areas.pdf"
    ))

    fig, ax = plt.subplots(figsize=(10,10))
    
    col_min = gdf.population.min()
    col_max = gdf.population.max()
    
    gdf.plot(
        column='population',
        legend=True,
        legend_kwds={
            'orientation': 'horizontal',
            'label': 'Population',
            'ticks': np.linspace(col_min, col_max, num=5),
            'format': lambda x, _: millify(x)
        },
        cmap='OrRd',
        ax=ax)
    
    ax.set_xticks([])
    ax.set_yticks([])

    col_threshold = gdf.population.quantile(0.85)
    
    for _, row in gdf.iterrows():
        color = 'black' if row.population < col_threshold else 'white'
        ax.annotate(text=f"{millify(row.population)}", xy=row.coords, ha='center', color=color)

    fig.savefig(figfile, dpi=96, bbox_inches='tight')

    return figfile


def plot_city_regions(gdf: gp.GeoDataFrame, city_name: str) -> Path:
    """ Plots a city's regions with their population to a file,
        and returns the corresponding path.
        
        Parameters:
            - gdf (gp.GeoDataFrame):
                Dataframe containing city info.
            - city_name (str):
                City name to use when creating the figure file.

        The figure file will be `output/<city_name>_regions.pdf`.
    """

    figfile = Path(os.path.join(
        os.path.realpath(os.path.dirname(__file__)),
        '..',
        'output',
        f"{city_name.lower()}_regions.pdf"
    ))

    colors = ('#003f5c', '#58508d', '#bc5090', '#ff6361')
    cmap = LinearSegmentedColormap.from_list(name='alberto', colors=colors)

    fig, ax = plt.subplots(figsize=(10,10))

    reg_gdf = gdf.dissolve(by='region', aggfunc='sum')
    
    reg_gdf.plot(cmap=cmap, ax=ax)
    gdf.boundary.plot(edgecolor='white', ax=ax)

    for _, region in reg_gdf.iterrows():
        x, y = region.geometry.representative_point().xy
        ax.annotate(millify(region.population), (x[0], y[0]), backgroundcolor='w')
    
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(figfile, dpi=96, bbox_inches='tight')

    return figfile