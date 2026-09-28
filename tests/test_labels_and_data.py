import numpy as np
import pandas as pd

from sepsis.data import HourlySplit, patient_ptr_from, split_patients
from sepsis.features import last_window
from sepsis.labels import make_labels


def test_labels_are_the_official_sepsis_label():
    df = pd.DataFrame({'patient_id': 'a', 'ICULOS': range(1, 6),
                       'SepsisLabel': [0, 0, 1, 1, 1]})
    y = make_labels(df)
    assert y.tolist() == [0, 0, 1, 1, 1]
    assert y.dtype == np.int8


def test_split_is_patient_level_disjoint_and_stratified(raw_df):
    df = pd.concat([raw_df.assign(patient_id=raw_df['patient_id'] + f'_{i}')
                    for i in range(10)])
    train, val = split_patients(df, val_frac=0.2, seed=1)
    assert not train & val
    assert train | val == set(df['patient_id'])
    septic = set(df.loc[df['SepsisLabel'] == 1, 'patient_id'])
    assert len(septic & val) == 2   # 20% of the 10 septic patients


def make_split():
    ids = pd.Series(['a'] * 5 + ['b'] * 2)
    ptr, pids = patient_ptr_from(ids)
    X = np.arange(7, dtype=np.float32)[:, None]       # feature = row number
    y = np.array([0, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    return HourlySplit(X, y, ptr, pids)


def test_every_hour_gets_a_window_padded_with_the_first_hour():
    Xw, yw, patient = make_split().windows(window=3)
    assert len(Xw) == 7                                  # one per hour, from hour 1
    assert Xw[0, :, 0].tolist() == [0, 0, 0]             # a, hour 1: padded
    assert Xw[1, :, 0].tolist() == [0, 0, 1]
    assert Xw[4, :, 0].tolist() == [2, 3, 4]             # full history
    assert Xw[5, :, 0].tolist() == [5, 5, 5]             # b never sees a's rows
    assert Xw[6, :, 0].tolist() == [5, 5, 6]
    assert yw.tolist() == [0, 0, 0, 1, 1, 0, 0]
    assert patient.tolist() == [0] * 5 + [1] * 2


def test_serving_window_matches_training_window():
    split = make_split()
    Xw, _, _ = split.windows(window=3)
    for hours in range(1, 6):                  # patient a after 1..5 hours
        np.testing.assert_array_equal(last_window(split.X[:hours], 3),
                                      Xw[hours - 1])


def test_hourly_split_round_trip(tmp_path):
    split = make_split()
    split.save(tmp_path / 's.npz')
    loaded = HourlySplit.load(tmp_path / 's.npz')
    np.testing.assert_array_equal(loaded.X, split.X)
    assert loaded.patient_ids.tolist() == ['a', 'b']
