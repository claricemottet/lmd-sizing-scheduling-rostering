from enum import Enum

class DemandType(Enum):
    """ Enum to choose demand type:
            - Uniform: there is the same expected demand at each time interval.
            - Peak: plotting demand on the y-axis and time on the x-axis, one should
                see a truncated normal distribution with a peak in the middle of the
                day.
            - Double peak: one should see the sum of two truncated normal
                distributions with the same variance but different means, at 1/3 and
                2/3 of the day time horizon.
            - At end: one should see a truncated normal distribution with mean at
                2/3 of the day time horizon. I.e., everyone wants parcels delivered
                in the afternoon/evening, when they come back home from work.
    """

    UNIFORM = 1
    PEAK = 2
    DOUBLE_PEAK = 3
    AT_END = 4