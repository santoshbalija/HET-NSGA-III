import datetime
import numpy as np
from pymoo.util.display import MultiObjectiveDisplay


class HeterogeneousDisplay(MultiObjectiveDisplay):

    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)

        time_in_s = algorithm.evaluator.scheduler.time()
        time_as_str = str(datetime.timedelta(seconds=int(time_in_s)))
        self.output.append("time", time_as_str, width=9)
        self.output.append("sec", time_in_s, width=12)
        self.output.append("time_no_units", time_in_s, width=12)
        self.output.append("feasibility_rate", np.mean(np.all(algorithm.pop.get('CV') <= 0, axis=1)), width=9)



