from __future__ import annotations

from typing import Optional
from warnings import warn

import numpy as np
import xgboost as xgb

from .benchmarking import detectron_test_statistics
from .defaults import XGB_PARAMS
from .utils.data import XGBDetectronRecord


def ecdf(x):
    """
    Compute the empirical cumulative distribution function
    :param x: array of 1-D numerical data
    :return: a function that takes a value and returns the probability that
        a random sample from x is less than or equal to that value
    """
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size

    return result


class DetectronResult:
    """
    A class to store the results of a Detectron test
    """

    def __init__(self, cal_record: XGBDetectronRecord, test_record: XGBDetectronRecord):
        """
        :param cal_record: Result of running benchmarking.detectron_test_statistics using IID test data
        :param test_record: Result of running benchmarking.detectron_test_statistics using the unknown test data
        """
        self.cal_record = cal_record
        self.test_record = test_record

        self.cal_counts = cal_record.counts()
        self.test_count = test_record.counts()[0]

        self.cdf = ecdf(self.cal_counts)
        self.p_value = self.cdf(self.test_count)

    def calibration_trajectories(self):
        rec = self.cal_record.get_record()
        return rec[['seed', 'ensemble_idx', 'rejection_rate']]

    def test_trajectory(self):
        rec = self.test_record.get_record()
        return rec[['ensemble_idx', 'rejection_rate']]

    def __repr__(self):
        return f"DetectronResult(p_value={self.p_value}, test_statistic={self.test_count}, " \
               f"baseline={self.cal_counts.mean():.2f}±{self.cal_counts.std():.2f})"


def detectron(train: tuple[np.ndarray, np.ndarray],
              val: tuple[np.ndarray, np.ndarray],
              observation_data: np.ndarray,
              base_model: xgb.Booster,
              iid_test: Optional[tuple[np.ndarray, np.ndarray]] = None,
              calibration_record: Optional[XGBDetectronRecord | DetectronResult] = None,
              xgb_params=XGB_PARAMS,
              ensemble_size=10,
              num_calibration_runs=100,
              num_boost_round=10,
              patience=3,
              balance_train_classes=True
              ):
    """
    Perform a Detectron test on a model, using the given data
    :param train: the original split used to train the model
    :param val: the original split used to validate the model
    :param observation_data:
    :param base_model: the base model to use for the pseudo-labeling, this should be a trained XGBoost model
    :param iid_test: An unseen test set that is assumed to be i.i.d. with the training data, used for calibration.
    Note this set can be none, in which case the calibration_record must be provided
    :param calibration_record: The result of running detectron or benchmarking.detectron_test_statistics on iid_test.
    Note this set can be none, in which case the iid_test must be provided.
    The calibration data can be collected from a previous detectron run. See the example #1 in
    the README for more details.
    :param xgb_params: (defaults.XGB_PARAMS) Parameters to pass to xgboost.train, see
    https://xgboost.readthedocs.io/en/latest/parameter.html
    :param ensemble_size: (10) number of models trained to disagree with each-other.
    Typically, a value of 5-10 is sufficient, but larger values may be required for very large datasets
    :param num_calibration_runs: (100) the number of different random runs to perform,
        each run operates on a random sample from test
    :param num_boost_round: (10) xgb parameter for the number of boosting rounds
    :param patience: (3) number of ensemble rounds to wait without improvement in the rejection rate
    :param balance_train_classes: (True) If True, the training data will be automatically balanced using weights.
    Disable only if your data is already class balanced.
    :return: DetectronResult object containing the results of the test.
    """
    obs_size = len(observation_data)
    obs_labels = np.zeros(obs_size)

    if iid_test is not None:
        test_size = len(iid_test[0])

        assert test_size > obs_size, \
            "The test set must be larger than the observation set to perform statistical bootstrapping"
        if test_size < 2 * obs_size:
            warn("The test set is smaller than twice the observation set, this may lead to poor calibration")

        cal_record = detectron_test_statistics(train=train, val=val, test=iid_test, base_model=base_model,
                                               sample_size=obs_size, xgb_params=xgb_params, ensemble_size=ensemble_size,
                                               num_runs=num_calibration_runs, num_boost_round=num_boost_round,
                                               patience=patience, balance_train_classes=balance_train_classes)

    else:
        if calibration_record is None:
            raise ValueError("If iid_test is not provided, calibration_record must be provided")
        if isinstance(calibration_record, DetectronResult):
            calibration_record = calibration_record.cal_record
        assert calibration_record.sample_size == obs_size, \
            "The calibration record must have been generated with the same sample size as the observation set"
        cal_record = calibration_record

    test_record = detectron_test_statistics(train=train, val=val, test=(observation_data, obs_labels),
                                            base_model=base_model, sample_size=obs_size, xgb_params=xgb_params,
                                            ensemble_size=ensemble_size, num_runs=1,
                                            num_boost_round=num_boost_round, patience=patience,
                                            balance_train_classes=balance_train_classes)

    return DetectronResult(cal_record, test_record)
