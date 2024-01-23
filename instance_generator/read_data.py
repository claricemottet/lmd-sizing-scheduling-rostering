from typing import Optional, Tuple
from os.path import join, dirname, realpath
from rasterstats import zonal_stats
from shapely.geometry import Point
import geopandas as gp
import warnings
import numpy as np


FRANCE_ZIP_CODES = join(
    dirname(realpath(__file__)),
    '..', 'data', 'france', 'codes_postaux_region.shp')

GERMANY_ZIP_CODES = join(
    dirname(realpath(__file__)),
    '..', 'data', 'germany', 'plz-5stellig.shp')

EUROPE_POPULATION = join(
    dirname(realpath(__file__)),
    '..', 'data', 'pop-europe', 'GHS_POP_SOURCE_EUROPE_R2016A_3035_100_v1_0.tif'
)


def __add_population(to: gp.GeoDataFrame) -> None:
    """ Adds population data to a GeoDataFrame which doesn't have any.

        The population data is saved in a new column "population".

        Parameters:
            - to (pd.GeoDataFrame):
                The dataframe to which we add the population data.
                It must refer to an area in Europe.
    """
    
    if 'population' in to.columns:
        warnings.warn('GeoDataFrame "to" already has a "population" column: overwriting.', RuntimeWarning)

    stats = zonal_stats(
        vectors=to.to_crs(3035).geometry,
        raster=EUROPE_POPULATION,
        stats='sum')
    population = [s['sum'] for s in stats]
    population = [int(pop) if pop is not None else 0 for pop in population]
    to['population'] = population


def __add_average_dist_to_representative(to: gp.GeoDataFrame) -> None:
    """ Given a GeoDataFrame with one area per row, it adds the average distance
        between the representative point of that area (column "coords") to a point
        sampled uniformly at random in the area's polygon (column "geometry").

        The distance data is saved in a new column "avg_dist_to_coords".

        Parameters:
            - to: (pd.GeoDataFrame):
                The dataframe to which we add the avg distance data.
    """

    def sample_pt(row: gp.GeoSeries) -> Tuple[float, float]:
        minx, miny, maxx, maxy = row.geometry.bounds

        while True:
            x = np.random.uniform(low=minx, high=maxx)
            y = np.random.uniform(low=miny, high=maxy)

            if row.geometry.contains(Point(x, y)):
                break

        return (x, y)

    def get_dist(row: gp.GeoSeries) -> float:
        ctr = Point(row.coords)
        dists = [ctr.distance(Point(sample_pt(row))) for _ in range(500)]

        return np.average(dists)

    to['avg_dist_to_coords'] = to.apply(get_dist, axis=1)


def read_france() -> gp.GeoDataFrame:
    """ Reads population data for France, at the postal code level.
    
        Returns a GeoPandas dataframe with columns:
            - surface: the surface area covered by the postal code
            - population: population living in the postal code area
            - num_households: number of households living in the postal code area
            - postal_code: the postal code itself
            - city: the name of the city in which the postal code area lies
            - department: the numeric code of the department (French admin division)
                in which the zip code area lies
            - geometry: the geojson geometry of the polygon defining the zip code
                area
            - coords: the coordinates of the representative point of the polygon
                defining the zip code area
    """

    france = gp.read_file(FRANCE_ZIP_CODES)
    france = france.rename(columns={
        'SURF': 'surface',
        'POP2010': 'population',
        'MEN2010': 'num_households',
        'ID': 'postal_code',
        'LIB': 'city',
        'DEP': 'department'
    })
    france = france[['surface', 'population', 'num_households', 'postal_code', 'city', 'department', 'geometry']].copy()
    france['coords'] = france.geometry.apply(lambda poly: poly.representative_point().coords[:])
    france['coords'] = [coords[0] for coords in france.coords]

    return france


def read_paris(france_gdf: Optional[gp.GeoDataFrame] = None) -> gp.GeoDataFrame:
    """ Reads population data for Paris, France, at the postal code level.

        Parameters:
            - france_gdf (pd.GeoDataFrame or None):
                If passed, it contains data about the whole of France,
                as returned by :func:`read_france`. Use this parameter
                to avoid re-reading France data multiple time, if calling
                more than one function which returns data about Franch
                cities.
    
        Returns a GeoPandas dataframe. For a description of the columns,
        see the documentation of :func:`read_france`.
    """

    france = read_france() if france_gdf is None else france_gdf
    paris = france[france.department == '75'].copy()
    __add_average_dist_to_representative(to=paris)

    return paris


def read_lyon(france_gdf: Optional[gp.GeoDataFrame] = None) -> gp.GeoDataFrame:
    """ Reads population data for Lyon, France, at the postal code level.
    
        It returns data for a subset of the zip codes lying in Lyon, to
        avoid very sparse suburbs far from the centre.

        Parameters:
            - france_gdf (pd.GeoDataFrame or None):
                If passed, it contains data about the whole of France,
                as returned by :func:`read_france`. Use this parameter
                to avoid re-reading France data multiple time, if calling
                more than one function which returns data about Franch
                cities.
    
        Returns a GeoPandas dataframe. For a description of the columns,
        see the documentation of :func:`read_france`.
    """

    france = read_france() if france_gdf is None else france_gdf
    lyon = france[
        (france.department == '69') &\
        (france.population > 28000) &\
        (~france.city.str.contains('Villefranche')) &\
        (~france.city.str.contains('Givors')) &\
        (~france.city.str.contains('Meyzieu'))].copy()
    __add_average_dist_to_representative(to=lyon)

    return lyon


def read_germany() -> gp.GeoDataFrame:
    """ Reads population data for Germany, at the postal code level.
    
        Returns a GeoPandas dataframe with columns:
            - surface: the surface area covered by the postal code
            - postal_code: the postal code itself
            - city: the name of the city in which the postal code area lies
            - geometry: the geojson geometry of the polygon defining the zip code
                area
            - coords: the coordinates of the representative point of the polygon
                defining the zip code area

        Note: the returned dataframe does NOT contain information about the
        population living in each area. Call city-specific methods such as
        :func:`read_berlin` and :func:`read_frankfurt` to obtain population
        data.
    """

    germany = gp.read_file(GERMANY_ZIP_CODES)
    germany = germany.rename(columns={
        'plz': 'postal_code',
        'note': 'city',
        'qkm': 'surface'
    })
    germany = germany[['postal_code', 'city', 'surface', 'geometry']].copy()
    germany['coords'] = germany.geometry.apply(lambda poly: poly.representative_point().coords[:])
    germany['coords'] = [coords[0] for coords in germany.coords]

    return germany


def read_berlin(germany_gdf: Optional[gp.GeoDataFrame] = None) -> gp.GeoDataFrame:
    """ Reads population data for Berlin, Germany, at the postal code level.
    
        It returns data for a subset of the zip codes lying in Berlin, to
        avoid very sparse suburbs far from the centre.

        Parameters:
            - germany_gdf (pd.GeoDataFrame or None):
                If passed, it contains data about the whole of Germany,
                as returned by :func:`read_germany`. Use this parameter
                to avoid re-reading Germany data multiple time, if calling
                more than one function which returns data about Franch
                cities.
    
        Returns a GeoPandas dataframe. For a description of the columns,
        see the documentation of :func:`read_germany`. In addition to the
        columns of the dataframe returned by :func:`read_germany`, this
        dataframe contains column "population" with the number of people
        living in the postal code area.
    """

    germany = read_germany() if germany_gdf is None else germany_gdf
    berlin = germany[
        (germany.city.str.contains('Berlin')) &
        ((germany.postal_code.str.startswith('10')) |
        (germany.postal_code.str.startswith('11')))
    ].copy()
    __add_population(to=berlin)
    __add_average_dist_to_representative(to=berlin)

    return berlin


def read_frankfurt(germany_gdf: Optional[gp.GeoDataFrame] = None) -> gp.GeoDataFrame:
    """ Reads population data for Frankfurt, Germany, at the postal code level.
    
        It returns data for a subset of the zip codes lying in Frankfurt, to
        avoid very sparse suburbs far from the centre.

        Parameters:
            - germany_gdf (pd.GeoDataFrame or None):
                If passed, it contains data about the whole of Germany,
                as returned by :func:`read_germany`. Use this parameter
                to avoid re-reading Germany data multiple time, if calling
                more than one function which returns data about Franch
                cities.
    
        Returns a GeoPandas dataframe. For a description of the columns,
        see the documentation of :func:`read_germany`. In addition to the
        columns of the dataframe returned by :func:`read_germany`, this
        dataframe contains column "population" with the number of people
        living in the postal code area.
    """

    germany = read_germany() if germany_gdf is None else germany_gdf
    frankfurt = germany[
        (germany.city.str.contains('Frankfurt')) &
        (germany.postal_code.str.startswith('60'))
    ].copy()
    __add_population(to=frankfurt)
    __add_average_dist_to_representative(to=frankfurt)

    return frankfurt[frankfurt.population >= 1000].copy()