"""Property tests for dagnam CSV loader column roles round-trip.

# Feature: dataset-column-roles, Property 18: Dagnam loader column roles round-trip
"""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from dagnam.data.loaders.csv_loader import _FEATURE_ROLES, _TARGET_ROLES, _split_by_roles

# ------------------------------------------------------------------
# Strategies
# ------------------------------------------------------------------

# Pool of safe column names to draw from (avoids slow regex generation)
_COL_NAME_POOL = [f"col_{i}" for i in range(20)]
_col_name_st = st.sampled_from(_COL_NAME_POOL)

# All roles that _split_by_roles recognises
_feature_role_st = st.sampled_from(sorted(_FEATURE_ROLES))
_target_role_st = st.sampled_from(sorted(_TARGET_ROLES))
_any_role_st = st.sampled_from(
    sorted(_FEATURE_ROLES) + sorted(_TARGET_ROLES) + ["ignore"]
)


@st.composite
def column_roles_strategy(draw):
    """Generate a dict of unique column names → roles with ≥1 target."""
    names = draw(
        st.lists(_col_name_st, min_size=2, max_size=8, unique=True)
    )

    # Pick at least one column to be a target
    target_idx = draw(st.integers(min_value=0, max_value=len(names) - 1))

    roles: dict[str, str] = {}
    for i, name in enumerate(names):
        if i == target_idx:
            roles[name] = draw(_target_role_st)
        else:
            roles[name] = draw(_any_role_st)
    return roles


def _make_df(column_names: list[str], n_rows: int = 5) -> pd.DataFrame:
    """Build a trivial DataFrame with the given columns."""
    return pd.DataFrame(
        {col: range(n_rows) for col in column_names}
    )


# ------------------------------------------------------------------
# Property tests
# ------------------------------------------------------------------


class TestSplitByRolesRoundTrip:
    """Validates: Requirements 15.1, 15.2, 15.4"""

    @given(data=column_roles_strategy())
    @settings(max_examples=100)
    def test_feature_and_target_match_roles(self, data: dict[str, str]):
        """For any valid column_roles with ≥1 target, _split_by_roles returns
        feature columns matching feature roles and label column matching a
        target role.
        """
        df = _make_df(list(data.keys()))
        label_col, feature_cols = _split_by_roles(df, data)

        # The label column must have a target role
        assert data[label_col] in _TARGET_ROLES

        # Every returned feature column must have a feature role
        for col in feature_cols:
            assert data[col] in _FEATURE_ROLES

        # Every column with a feature role must appear in feature_cols
        expected_features = {c for c, r in data.items() if r in _FEATURE_ROLES}
        assert set(feature_cols) == expected_features

        # Every column with a target role must be either the label_col
        # (first target) or a subsequent target (not in features/ignore)
        expected_targets = [c for c in df.columns if data.get(c) in _TARGET_ROLES]
        assert label_col == expected_targets[0]

    @given(data=column_roles_strategy())
    @settings(max_examples=100)
    def test_ignore_columns_excluded(self, data: dict[str, str]):
        """Columns with role 'ignore' do not appear in features or target."""
        df = _make_df(list(data.keys()))

        # Ensure at least one target exists (guaranteed by strategy)
        has_target = any(r in _TARGET_ROLES for r in data.values())
        if not has_target:
            return

        label_col, feature_cols = _split_by_roles(df, data)

        ignore_cols = {c for c, r in data.items() if r == "ignore"}
        assert label_col not in ignore_cols
        assert ignore_cols.isdisjoint(set(feature_cols))

    @given(data=column_roles_strategy())
    @settings(max_examples=100)
    def test_feature_columns_preserve_order(self, data: dict[str, str]):
        """Feature columns preserve original DataFrame column order."""
        col_names = list(data.keys())
        df = _make_df(col_names)

        has_target = any(r in _TARGET_ROLES for r in data.values())
        if not has_target:
            return

        _, feature_cols = _split_by_roles(df, data)

        # feature_cols should be in the same relative order as df.columns
        expected_ordered = [c for c in df.columns if data.get(c) in _FEATURE_ROLES]
        assert feature_cols == expected_ordered
