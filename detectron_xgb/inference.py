from __future__ import annotations

from typing import Optional
from warnings import warn

import numpy as np
import xgboost as xgb

from .benchmarking import detectron_test_statistics
from .defaults import XGB_PARAMS
from .utils.data import XGBDetectronRecord


def ecdf(x):
    x = np.sort(x)

    def result(v):
        return np.searchsorted(x, v, side='right') / x.size

    return result


class DetectronResult:
    def __init__(self, cal_record: XGBDetectronRecord, test_record: XGBDetectronRecord):
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
              calibration_record: Optional[XGBDetectronRecord] = None,
              xgb_params=XGB_PARAMS,
              ensemble_size=10,
              num_calibration_runs=100,
              num_boost_round=10,
              patience=3,
              balance_train_classes=True
              ):
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
        assert calibration_record.sample_size == obs_size, \
            "The calibration record must have been generated with the same sample size as the observation set"
        cal_record = calibration_record

    test_record = detectron_test_statistics(train=train, val=val, test=(observation_data, obs_labels),
                                            base_model=base_model, sample_size=obs_size, xgb_params=xgb_params,
                                            ensemble_size=ensemble_size, num_runs=1,
                                            num_boost_round=num_boost_round, patience=patience,
                                            balance_train_classes=balance_train_classes)

    return DetectronResult(cal_record, test_record)
