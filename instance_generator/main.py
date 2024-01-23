from city_type import CityType
from demand_type import DemandType
from generator import InstanceGenerator
from itertools import product
from tqdm import tqdm

NUM_TIME_INTERVALS = [8]
DEMAND_BASELINE = [0.5, 1.0, 2.0, 4.0]
DEMAND_TYPE = list(DemandType)
NUM_SCENARIOS = [30]

if __name__ == '__main__':
    all_params = product(NUM_TIME_INTERVALS, DEMAND_BASELINE, DEMAND_TYPE, NUM_SCENARIOS)

    city = CityType.FRANKFURT
    generator = InstanceGenerator(city_type=city)
    generator.plot_city_areas()
    generator.plot_city_regions()

    print(f"Generating instances for {city.name.capitalize()}")

    for nt, db, dt, ns in tqdm(all_params):
        generator.set_params(num_time_intervals=nt, demand_baseline=db, demand_type=dt, num_scenarios=ns)
        generator.generate()