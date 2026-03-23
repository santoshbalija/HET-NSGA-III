import time
import timeit
from collections import deque
from threading import Thread

from pyheterogeneous.core.problem import find_target_group


class Scheduler:

    def is_alive(self):
        return True

    def submit(self, job):
        pass

    def time(self):
        pass


class SynchronousScheduler(Scheduler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ts = 0.0

    def submit(self, job, **kwargs):
        exec_job(job)

        eval_time = time_of_job(job)
        job["eval_time"] = eval_time

        self.ts += eval_time

        return job

    def time(self):
        return self.ts


def time_as_int():
    return round(time.time() * 1000)


def time_of_job(job):
    problem, pop, targets, callback = job["problem"], job["pop"], job["targets"], job.get("callback")
    exec_job(job)

    N = len(pop)
    eval_times = problem.eval_times
    I = set([find_target_group(eval_times, target) for target in targets])

    eval_time = 0.0
    for i in I:
        _, t = eval_times[i]
        eval_time += N * t

    return eval_time


def exec_job(job):
    problem, pop, targets, callback = job["problem"], job["pop"], job["targets"], job.get("callback")

    start = timeit.timeit()
    out = problem.evaluate(pop.get("X"), targets=targets)
    exec_time = timeit.timeit() - start

    job["exec_time"] = exec_time
    job["out"] = out

    if callback is not None:
        callback(job)

    return job


async def async_exec_job(job):
    exec_job(job)


class AsynchronousScheduler(Scheduler):

    def __init__(self, n_workers=2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.start_time = None

        self.queue = deque()

        workers = []
        for k in range(n_workers):
            worker = Worker(self.queue)
            worker.start()
            workers.append(worker)

    def submit(self, job, **kwargs):

        # set the start time when the first job is submitted
        if self.start_time is None:
            self.start_time = time_as_int()

        self.queue.append(job)

    def time(self):
        if self.start_time is None:
            return 0
        else:
            return time_as_int() - self.start_time


def pop_left(q):
    try:
        return q.popleft()
    except:
        return None


class Worker(Thread):

    def __init__(self, queue, **kwargs):
        super().__init__(**kwargs)
        self.queue = queue
        self.stop = False

    def run(self):

        while not self.stop:

            job = pop_left(self.queue)

            if job is not None:
                print(id(self), job)
                exec_job(job)
            else:
                time.sleep(0.01)
