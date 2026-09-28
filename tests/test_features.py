import numpy as np
import pandas as pd

from sepsis.features import CLIP, FEATURE_COLS, RAW_COLS, Preprocessor, fill_features

MEDIANS = {'HR': 80.0, 'Lactate': 1.5}


def test_obs_flag_marks_measured_hours_only():
    raw = pd.DataFrame({'Lactate': [np.nan, 2.0, np.nan]})
    out = fill_features(raw, MEDIANS)
    assert out['Lactate_obs'].tolist() == [0, 1, 0]


def test_imputation_is_causal():
    # Hour 0 must NOT see hour 1's value (no back-fill from the future)
    raw = pd.DataFrame({'Lactate': [np.nan, 4.0, np.nan]})
    out = fill_features(raw, MEDIANS)
    assert out['Lactate'].tolist() == [1.5, 4.0, 4.0]


def test_forward_fill_does_not_cross_patients():
    raw = pd.DataFrame({'patient_id': ['a', 'a', 'b'],
                        'HR': [100.0, np.nan, np.nan]})
    out = fill_features(raw, MEDIANS, group_col='patient_id')
    assert out['HR'].tolist() == [100.0, 100.0, 80.0]


def test_output_has_all_features_in_order():
    medians = {c: 1.0 for c in RAW_COLS}
    out = fill_features(pd.DataFrame({'HR': [90.0]}), medians)
    assert list(out.columns) == FEATURE_COLS
    assert not out.isna().any().any()


def test_preprocessor_scales_and_leaves_binary_columns(raw_df):
    pre = Preprocessor.fit(raw_df, group_col='patient_id')
    X = pre.transform(raw_df, group_col='patient_id')
    assert X.shape == (len(raw_df), len(FEATURE_COLS))
    assert X.dtype == np.float32
    assert np.abs(X).max() <= CLIP
    hr = FEATURE_COLS.index('HR')
    assert abs(X[:, hr].mean()) < 1e-4          # standardised
    gender = FEATURE_COLS.index('Gender')
    assert set(np.unique(X[:, gender])) <= {0.0, 1.0}   # untouched


def test_preprocessor_json_round_trip(raw_df, tmp_path):
    pre = Preprocessor.fit(raw_df, group_col='patient_id')
    pre.to_json(tmp_path / 'p.json')
    loaded = Preprocessor.from_json(tmp_path / 'p.json')
    np.testing.assert_array_equal(pre.transform(raw_df, 'patient_id'),
                                  loaded.transform(raw_df, 'patient_id'))


def test_single_patient_matches_grouped_transform(raw_df):
    # Serving transforms one patient at a time; training transforms all
    # patients grouped. Both must give identical features.
    pre = Preprocessor.fit(raw_df, group_col='patient_id')
    grouped = pre.transform(raw_df, group_col='patient_id')
    mask = (raw_df['patient_id'] == 'p001').to_numpy()
    single = pre.transform(raw_df[mask].reset_index(drop=True))
    np.testing.assert_array_equal(grouped[mask], single)
