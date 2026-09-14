import warnings

import pandas as pd

from deepuplift.data.preprocessing import TabularPreprocessor


def test_high_dimensional_numeric_features_do_not_fragment_preprocessed_frame():
    frame = pd.DataFrame({f"x{i}": [float(i), float(i + 1)] for i in range(300)})
    frame.loc[0, "x299"] = float("nan")
    processor = TabularPreprocessor(list(frame.columns))

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        encoded = processor.fit_transform(frame)

    assert encoded.shape == (2, 300)
    assert encoded.loc[0, "x0"] == 0.0
    assert encoded.loc[1, "x299"] == 300.0
    assert not any(issubclass(item.category, pd.errors.PerformanceWarning) for item in observed)
