import copy
import warnings
from multiprocessing.pool import ThreadPool

import numpy as np

from ezmodel.util.metrics import calc_metric
from ezmodel.util.misc import at_least2d
from ezmodel.util.partitioning.crossvalidation import CrossvalidationPartitioning


class Benchmark:

    def __init__(self,
                 models,
                 metrics=["mse", "mae", "r2", "spear"],
                 show_warnings=False,
                 raise_exception=False,
                 n_threads=None,
                 verbose=False):

        """

        This is a benchmark class which evaluates the performance of surrogates on a data set.

        Parameters
        ----------
        models : list
            A list of dictionaries where the keys `label` and `obj` need to be set.

        metrics : list
            What metrics shall be calculated during the benchmark.

        show_warnings : bool
            Whether warnings should be displayed or not. This can cause to show all warnings from
            all surrogates.

        raise_exception : bool
            Whether an exception should be raised if a surrogate fails.

        n_threads : int
            If multi-threading should be used this can be enabled, then each process of fitting is
            parallelized.

        verbose : bool
            Whether the benchmark should provide printouts or not.

        """

        super().__init__()
        self.models = dict(models)

        # the models can either be provided in a dict or directly -> then this reformatting is necessary
        for k, v in self.models.items():
            if not isinstance(v, dict):
                self.models[k] = dict(model=v)
            self.models[k]["label"] = k

        # the metrics to be used to evaluate the performance of each model
        self.metrics = metrics

        # whether warnings shall be displayed
        self.show_warnings = show_warnings

        # whether an exception shall be raised if a model fails
        self.raise_exception = raise_exception

        # the number of threads, None simply disables multi-threading
        self.n_threads = n_threads

        # if true, the benchmark does some printouts
        self.verbose = verbose

        # the records when the benchmark has been executed
        self.records = None

        # the overall results grouped by each surrogate
        self.records_by_model = None

        # storage for data for each experiment
        self.data = None

    def do(self, X, y, partitions=None):
        X, y = at_least2d(X, expand="r"), at_least2d(y, expand="c")[:, 0]
        assert len(X) == len(y)

        # when no partitions are provided by default a k-fold cross-validation is done
        if partitions is None:
            partitions = CrossvalidationPartitioning(X, 3, seed=1).do()

        # store the current setup of the benchmark
        self.data = dict(X=X, y=y, partitions=partitions)

        # an empty list where the records are added to
        self.records = []

        # create all the entries each of them representing a job for fitting a model - for each partition
        for k in range(len(partitions)):

            # for each model to be tested
            for _, entry in self.models.items():
                model = entry["model"]
                val = dict(
                    benchmark="benchmark@%s" % id(self),
                    proto=model,
                    partition=k,
                )
                self.records.append({**entry, **val})

        # fit all the models for each partition
        self._do()

        # evaluate the performance for each of the methods
        self._performance()

        # do some more post-processing to prepare the results
        self._postprocessing()

        return self

    def _do(self):
        if self.n_threads is None:
            ret = [fit(self.data, entry, self.show_warnings, self.raise_exception) for entry in self.records]
        else:
            with ThreadPool(self.n_threads) as pool:
                ret = pool.starmap(fit,
                                   [(self.data, entry, self.show_warnings, self.raise_exception) for entry in
                                    self.records])
        return ret

    def _performance(self):

        # the true data for predictions
        partitions, y_true = self.data["partitions"], self.data["y"]

        # now set the results based on the predictions
        for record in self.records:

            # set all metrics to none which ensures they are set if an error occurs
            record["trn_error"] = np.nan
            for metric in self.metrics:
                record[metric] = np.nan

            # if the model was fitted successfully
            if record["success"]:

                try:

                    trn, tst = partitions[record["partition"]]
                    y_hat, trn_y_hat = record["y_hat"], record["trn_y_hat"]

                    record["trn_error"] = calc_metric("mae", y_true[trn], trn_y_hat)

                    for metric in self.metrics:
                        warnings.simplefilter("ignore")
                        record[metric] = calc_metric(metric, y_true[tst], y_hat)

                    if self.verbose:
                        print(record["label"], record["obj"].time)

                except:
                    pass

    def _postprocessing(self):
        records = self.records

        # now let us create the results of the benchmark
        results = dict(self.models)
        for key in results.keys():
            results[key] = dict(runs=[], performance={}, **results[key])

        for entry in records:
            hash = entry["label"]
            results[hash]["runs"].append(entry)

        collect = lambda k, x: np.array([run[x] for run in results[k]["runs"]])

        # collect the data from all the runs
        for metric in self.metrics:
            for k, v in results.items():
                vals = collect(k, metric)
                v["performance"][metric] = dict(mean=vals.mean(), std=vals.std(), min=vals.min(), max=vals.max(),
                                                values=vals)

        for k, v in results.items():
            v["n_runs"] = len(v["runs"])
            v["success"] = np.all(collect(k, "success"))
            v["trn_error"] = np.max(collect(k, "trn_error"))

        self.records_by_model = results

    def results(self,
                only_successful=True,
                as_list=True,
                sorted_by=None,
                max_trn_error=None,
                include_metadata=False,
                include_records=False):

        ret = {}
        for k, v in self.records_by_model.items():
            if not only_successful or v["success"]:
                ret[k] = v
                if sorted_by is not None:
                    metric, value, _ = sorted_by
                    v["metric"] = v["performance"][metric][value]

        # if not a list should be returned that's it
        if as_list:

            # otherwise get a list and do some additionally work
            ret = list(ret.values())

            if sorted_by is not None:
                metric, value, ascending = sorted_by

                # sort the indices by their performance
                sign = 1 if ascending else -1
                ret = sorted(ret, key=lambda x: sign * x["metric"])
            else:
                ret = sorted(ret, key=lambda x: x["performance"]["mae"]["mean"])

            # filter out solutions with too much error on training set (for approximation models)
            if max_trn_error is not None:

                is_valid = [e["trn_error"] < max_trn_error for e in ret]

                if not any(is_valid):
                    print(f"WARNING: No model with maximum training error of {max_trn_error} was found.")

                else:
                    ret = [ret[i] for i in range(len(ret)) if is_valid[i]]

        if include_metadata:
            ret = {**dict(self.data), "results": ret}

            if include_records:
                ret["records"] = self.records

        return ret

    def statistics(self, metric="mae", vals=['mean', 'std', 'min', 'max', 'median'], sort_by="mean", ascending=True):
        try:
            import pandas as pd
        except:
            raise Exception("Please install the pandas toolbox for using statistics: pip install pandas")

        df = pd.DataFrame(self.records)
        tbl = df.groupby(["label"]).agg({metric: vals})
        df.groupby(["label"]).agg({metric: vals})

        if sort_by is not None:
            tbl = tbl.sort_values((metric, sort_by), ascending=ascending)

        return tbl

    def correlation(self, models=None):

        if models is None:
            models = [model for model, vals in self.data.items() if vals["success"]]

        preds = {}
        for model in models:
            preds[model] = np.concatenate([run["y_hat"] for run in self.data[model]["runs"]])

        import pandas as pd
        return pd.DataFrame(preds).corr()


# ----------------------------------------------------------------
# Util
# ----------------------------------------------------------------

def fit(data, record, show_warning, raise_exception):
    try:

        X, y, partitions = data["X"], data["y"], data["partitions"]

        # get the data to be used for evaluation
        trn, tst = partitions[record["partition"]]

        model = copy.deepcopy(record["proto"])
        model.fit(X[trn], y[trn])

        trn_y_hat = model.predict(X[trn])
        tst_y_hat = model.predict(X[tst])

        # set the required entries including the prediction from the model
        record["trn_y_hat"] = trn_y_hat[:, 0]
        record["y_hat"] = tst_y_hat[:, 0]
        record["model"] = model
        record["success"] = True

        return model

    except Exception as e:
        record["success"] = False

        if show_warning:
            print("WARNING: %s has failed: %s" % (model, str(e)))

        if raise_exception:
            raise e
