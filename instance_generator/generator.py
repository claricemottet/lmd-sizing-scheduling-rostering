from typing import Tuple, Union, Dict, List
from pathlib import Path
from shapely.geometry import Point
from scipy.stats import truncnorm
from math import ceil
import geopandas as gpd
import os.path
import numpy as np
import warnings
import json

from city_type import CityType
from demand_type import DemandType
from read_data import read_paris, read_lyon, read_berlin, read_frankfurt
from utils import add_regions, plot_city_areas, plot_city_regions


class NpEncoder(json.JSONEncoder):
    """ Helper class that allows to output numpy data types to json natively. """

    def default(self, obj):
        """ If :param:`obj` is a numpy type, convert it to the corresponding
            Python native type, so it can be serialised by the `json` library.
        """

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        raise TypeError(f"Unserialisable object {obj} of type {type(obj)}")


class InstanceGenerator:
    """ Instance generator for the tactical LMD staffing problem. """

    num_time_intervals: int
    demand_baseline: float
    demand_type: DemandType
    num_scenarios: int

    def __init__(self, city_type: CityType, **kwargs):
        """ Initialises the instance generator object.
        
            Parameters:
                - city_type (CityType):
                    The city we want to base our instance on.
                - num_time_intervals (int):
                    Number of time intervals to consider.
                - demand_baseline (float):
                    Baseline to use to generate demand, in parcels/1'000 people.
                    At each time interval, each area should receive on average
                    :param:`demand_baseline` * area_population / 1'000 parcels.
                - demand_type (DemandType):
                    Demand pattern type.
                - num_scenarios (int):
                    Number of demand scenarios to generate.
        """

        self.city_type = city_type
        self.set_params(**kwargs)
        self.gdf = self.__read_gdf()
        add_regions(self.gdf, self.city_type)

        self.city_centre = self.__get_city_centre()
        self.geography = self.__get_geography()

    def set_params(self, **kwargs) -> None:
        """ Changes the parameters of the instance generator object.
        
            Parameters:
                - num_time_intervals (int):
                    Number of time intervals to consider.
                - demand_baseline (float):
                    Baseline to use to generate demand, in parcels/1'000 people.
                    At each time interval, each area should receive on average
                    :param:`demand_baseline` * area_population / 1'000 parcels.
                - demand_type (DemandType):
                    Demand pattern type.
                - num_scenarios (int):
                    Number of demand scenarios to generate.
        """

        self.num_time_intervals = kwargs.get('num_time_intervals')
        self.demand_baseline = kwargs.get('demand_baseline')
        self.demand_type = kwargs.get('demand_type')
        self.num_scenarios = kwargs.get('num_scenarios')

    def generate(self) -> Path:
        """ Generates a city/instance file, writes it to disk and returns the
            path to the file.

            It writes the instance in folder `output`.
        """

        basename = self.__get_basename()

        obj = dict(
            name=basename,
            num_time_intervals=self.num_time_intervals,
            num_scenarios=self.num_scenarios,
            demand_baseline=self.demand_baseline,
            demand_type=self.demand_type.name.lower(),
            geography=self.geography,
            scenarios=self.__get_lmd_data()
        )

        filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '..', 
            'output',
            basename + '.json'
        )

        with open(filename, 'w') as f:
            json.dump(obj, fp=f, indent=4, cls=NpEncoder)

        return Path(filename)

    def plot_city_areas(self) -> Path:
        """ Plots a city's areas with their population to a file,
            and returns the corresponding path.

            For more information see :func:`.utils.plot_city_areas`.
        """

        return plot_city_areas(gdf=self.gdf, city_name=self.city_type.name.capitalize())

    def plot_city_regions(self) -> Path:
        """ Plots a city's regions with their population to a file,
            and returns the corresponding path.

            For more information see :func:`.utils.plot_city_regions`.
        """

        return plot_city_regions(gdf=self.gdf, city_name=self.city_type.name.capitalize())
    
    def __correct_area(self, area: float) -> float:
        """ Applies unit-of-measure corrections to areas.
        """

        if self.city_type in (CityType.PARIS, CityType.LYON):
            return area / 1e6
        elif self.city_type in (CityType.BERLIN, CityType.FRANKFURT):
            return area * 1e4
        
    def __correct_distance(self, area: float) -> float:
        """ Applies unit-of-measure corrections to areas.
        """

        if self.city_type in (CityType.PARIS, CityType.LYON):
            return area / 1e3
        elif self.city_type in (CityType.BERLIN, CityType.FRANKFURT):
            return area * 1e2

    def __read_gdf(self) -> gpd.GeoDataFrame:
        """ Reads the geopandas dataframe corresponding to the real-life city
            we are generating the instance for.
        """

        if self.city_type == CityType.PARIS:
            return read_paris()
        elif self.city_type == CityType.LYON:
            return read_lyon()
        elif self.city_type == CityType.BERLIN:
            return read_berlin()
        elif self.city_type == CityType.FRANKFURT:
            return read_frankfurt()
        else:
            raise ValueError(f"{self.city_type} is not a valid city type.")

    def __get_basename(self) -> str:
        """ Gets an instance basename, which combines:
                - The real-life city name.
                - The number of time intervals.
                - The demand baseline.
                - The number of scenarios.
                - The demand type.
            
            It returns it as a string in which each component is separated by an underscore.
        """

        return f"{self.city_type.name.lower()}_" +\
               f"db={self.demand_baseline:.2f}_" +\
               f"dt={self.demand_type.name.lower().replace('_', '')}"

    def __get_city_centre(self) -> Point:
        """ Returns the centroid of the city shape, as a shapely Point. """
        
        return self.gdf.geometry.unary_union.exterior.centroid

    def __get_area_population(self, area_id: str) -> int:
        """ Returns the population leaving in an area.
        
            Parameters:
                - area_id (str):
                    The identifier of the area (usually its postal code).
        """

        area_df = self.gdf[self.gdf.postal_code == area_id]

        if len(area_df) == 0:
            valid_ids = ', '.join(sorted(self.gdf.postal_code.unique()))
            raise ValueError(f"There is no area with {area_id=}. Possible values are {valid_ids}.")
        elif len(area_df) > 1:
            warnings.warn(f"City dataframe for {self.city_type=} contains {len(area_df)} entries with {area_id=}.")

        return int(area_df.population.mean())

    def __get_geography(self) -> Dict:
        """ Returns a dictionary (to be serialised into json) with geometry information
            about the city, its regions and their areas.
            
            For each of the three levels (city, region, area), it provides:
                - The population living there.
                - The surface area.
            Furthemore, it identifies each region and area with an ID.
            Finally, for each area it provides:
                - The average distance between the depot and a point sampled
                  uniformly at random in the area polygon.
        """

        obj = dict()
        city_geom = self.gdf.geometry.unary_union
        
        obj['population'] = round(self.gdf.population.sum(), ndigits=0)
        obj['surface_area'] = round(self.__correct_area(city_geom.area), ndigits=2)
        
        regions = list()
        for region_id in self.gdf.region.unique():
            rdf = self.gdf[self.gdf.region == region_id]
            reg_geom = rdf.geometry.unary_union
            reg = dict()

            reg['id'] = region_id
            reg['population'] = round(rdf.population.sum(), ndigits=0)
            reg['surface_area'] = round(self.__correct_area(reg_geom.area), ndigits=2)

            areas = list()
            for area_id in rdf.postal_code.unique():
                adf = rdf[rdf.postal_code == area_id]

                assert len(adf) == 1
                adf = adf.iloc[0]

                area = dict()

                area['id'] = area_id
                area['population'] = round(adf.population, ndigits=0)
                area['surface_area'] = round(self.__correct_area(adf.geometry.area), ndigits=2)
                area['avg_distance_to_depot'] = round(self.__correct_distance(adf.avg_dist_to_coords), ndigits=3)

                areas.append(area)

            reg['areas'] = areas
            regions.append(reg)

        obj['regions'] = regions

        return dict(city=obj)

    def __get_lmd_data(self) -> List:
        """ Returns a list of scenarios with demand data.
            See :func:`__generate_scenario` for more information.
        """

        return [
            self.__generate_scenario(num=num)
            for num in range(self.num_scenarios)
        ]

    def __generate_scenario(self, num: int) -> Dict:
        """ Returns a dictionary with demand data, for a single scenario.
        
            The dictionary has two keys: "scenario_num" with a progressive
            scenario id provided by :param:`num`, and "data" which is a list
            of dictionaries. Each dictionary refers to an area and contains
            the number of parcels to deliver there at each period, and the
            number of couriers that one would need to perform all deliveries,
            according to the modified Figliozzi model.

            Parameters:
                - num (int):
                    Scenario id.
        """

        area_ids = sorted(self.gdf.postal_code.unique())
        data = list()

        for area_id in area_ids:
            demand = self.__generate_demand(area_id=area_id)
            required_couriers = self.__required_couriers(area_id=area_id, demand=demand)

            data.append({
                'area_id': area_id,
                'demand': demand,
                'required_couriers': required_couriers
            })

        return dict(
            scenario_num=num,
            data=data)
    
    def __required_couriers(self, area_id: str, demand: List[int]) -> List[int]:
        """ Determines how many couriers are required in a given area at each
            period, according to the Figliozzi model.

            Parameters:
                - area_id (str):
                    Id of the area.
                - demand (List[str]):
                    Number of parcels to deliver at each period.
        """

        # Constants for the Figliozzi model
        k = 0.77
        capacity = 5
        speed_kmh = 21
        service_time_min = 5
        period_min = 120

        def get_area(region):
            for area in region['areas']:
                if area['id'] == area_id:
                    return area
            return None

        for region in self.geography['city']['regions']:
            if (area := get_area(region)) is not None:
                area_sqkm = area['surface_area']
                avg_dist_km = area['avg_distance_to_depot']
                break
        else:
            raise ValueError(f"Cannot find area {area_id}!")

        m = list()

        for n in demand:
            if n == 0:
                m.append(0)
                continue

            cap_bound = int(ceil(n / capacity))
            time_bound_numerator = (k / speed_kmh) * np.sqrt(area_sqkm * n) + n * service_time_min
            time_bound_denominator = period_min + (k / (speed_kmh * n)) * np.sqrt(area_sqkm * n) - 2 * avg_dist_km / speed_kmh
            time_bound = int(ceil(time_bound_numerator / time_bound_denominator))

            m.append(max(cap_bound, time_bound))

        return m

    def __generate_demand(self, area_id: str) -> List[int]:
        """ Calls the appropriate demand generating function, depending on the
            desired demand type for the instance.

            Parameters:
                - area_id (str):
                    Id of the area we generate demand for.
        """

        if self.demand_type == DemandType.UNIFORM:
            return self.__generate_uniform_demand(area_id=area_id)
        elif self.demand_type == DemandType.PEAK:
            return self.__generate_peak_demand(area_id=area_id)
        elif self.demand_type == DemandType.DOUBLE_PEAK:
            return self.__generate_double_peak_demand(area_id=area_id)
        elif self.demand_type == DemandType.AT_END:
            return self.__generate_at_end_demand(area_id=area_id)
        else:
            raise ValueError(f"Demand generation not yet implemented for demand type {self.demand_type.name}")

    def __get_noisy_demand(self, area_id: str) -> int:
        """ Generates a noisy estimate for the daily demand of a given area.
            
            The base demand is given by the formula:
                demand_baseline * population / 1'000
            where demand_baseline is an instance generation parameter, and
            population is the number of people living in the area. The noisy
            estimate is generated sampling uniformly at random in the interval
            [75%, 125%] of the base demand.

            Parameters:
                - area_id (str):
                    Id of the area we generate demand for.
        """

        pop = self.__get_area_population(area_id=area_id)
        demand = self.demand_baseline * pop / 1_000
        noisy_demand = demand * np.random.uniform(low=0.75, high=1.25)

        return int(noisy_demand)

    def __generate_uniform_demand(self, area_id: str) -> List[int]:
        """ Generate demand for a given area (for all time intervals) following
            the uniform demand type.

            We first calculate the total demand of that area over the entire day.
            This is given by the noisy demand function :func:`__get_noisy_demand`.
            Next, we distribute this demand uniformly over the time horizon.

            Parameters:
                - area_id (str):
                    Id of the area we generate demand for.

            Returns a list with as many entries as time intervals in the time
            horizon. Each entry contains the number of parcels to deliver in the
            area during the given time interval.
        """

        demand = self.__get_noisy_demand(area_id=area_id)
        time_intervals = np.random.uniform(low=0, high=self.num_time_intervals, size=demand)
        
        return list(np.histogram(time_intervals, bins=range(self.num_time_intervals + 1))[0])

    def __generate_peak_demand(self, area_id: str) -> List[int]:
        """ Generate demand for a given area (for all time intervals) following
            the peak demand type.

            We first calculate the total demand of that area over the entire time
            horizon. This is given by the noisy demand function :func:`__get_noisy_demand`.
            Next, we distribute this demand according to a truncated normal distribution
            with:
                * Mean at the middle of the time horizon.
                * Standard deviation equal to 1/6 of the time horizon length.
                * Truncation at the extremes of the time horizon.

            Parameters:
                - area_id (str):
                    Id of the area we generate demand for.

            Returns a list with as many entries as time intervals in the time
            horizon. Each entry contains the number of parcels to deliver in the
            area during the given time interval.
        """

        demand = self.__get_noisy_demand(area_id=area_id)
        
        tn_low, tn_high = 0, self.num_time_intervals
        loc, scale = self.num_time_intervals / 2, self.num_time_intervals / 6
        a, b = (tn_low - loc) / scale, (tn_high - loc) / scale

        time_intervals = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=demand)

        return list(np.histogram(time_intervals, bins=range(self.num_time_intervals + 1))[0])

    def __generate_double_peak_demand(self, area_id: str) -> List[int]:
        """ Generate demand for a given area (for all time intervals) following
            the double-peak demand type.

            We first calculate the total demand of that area over the entire time
            horizon. This is given by the noisy demand function :func:`__get_noisy_demand`.
            Next, we distribute this demand according to the sum of two truncated normal
            distributions with:
                * Means at 1/3rd and 2/3rds of the time horizon.
                * Standard deviation equal to 1/10 of the time horizon length.
                * Truncation at the extremes of the time horizon.

            Parameters:
                - area_id (str):
                    Id of the area we generate demand for.

            Returns a list with as many entries as time intervals in the time
            horizon. Each entry contains the number of parcels to deliver in the
            area during the given time interval.
        """

        demand = self.__get_noisy_demand(area_id=area_id)
        half_demand = int(demand / 2)
        
        tn_low, tn_high = 0, self.num_time_intervals
        loc1 = self.num_time_intervals / 3
        loc2 = 2 * self.num_time_intervals / 3
        scale = self.num_time_intervals / 10
        a1, b1 = (tn_low - loc1) / scale, (tn_high - loc1) / scale
        a2, b2 = (tn_low - loc2) / scale, (tn_high - loc2) / scale

        time_intervals = np.append(
            truncnorm.rvs(a=a1, b=b1, loc=loc1, scale=scale, size=half_demand),
            truncnorm.rvs(a=a2, b=b2, loc=loc2, scale=scale, size=half_demand))

        return list(np.histogram(time_intervals, bins=range(self.num_time_intervals + 1))[0])

    def __generate_at_end_demand(self, area_id: str) -> List[int]:
        """ Generate demand for a given area (for all time intervals) following
            the at-end demand type.

            We first calculate the total demand of that area over the entire time
            horizon. This is given by the noisy demand function :func:`__get_noisy_demand`.
            Next, we distribute this demand according to a truncated normal distribution
            with:
                * Mean at 2/3rds of the time horizon.
                * Standard deviation equal to 1/5 of the time horizon length.
                * Truncation at the extremes of the time horizon.

            Parameters:
                - area_id (str):
                    Id of the area we generate demand for.

            Returns a list with as many entries as time intervals in the time
            horizon. Each entry contains the number of parcels to deliver in the
            area during the given time interval.
        """

        demand = self.__get_noisy_demand(area_id=area_id)
        
        tn_low, tn_high = 0, self.num_time_intervals
        loc, scale = 2 * self.num_time_intervals / 3, self.num_time_intervals / 5
        a, b = (tn_low - loc) / scale, (tn_high - loc) / scale

        time_intervals = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, size=demand)

        return list(np.histogram(time_intervals, bins=range(self.num_time_intervals + 1))[0])