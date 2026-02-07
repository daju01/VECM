from __future__ import annotations

from hypothesis import given, strategies as st

from vecm_project.scripts.cache_keys import hash_config


@given(
    st.dictionaries(
        keys=st.text(min_size=1, max_size=8),
        values=st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(max_size=12)),
        max_size=8,
    )
)
def test_hash_config_is_order_independent(sample: dict) -> None:
    items = list(sample.items())
    reversed_payload = dict(reversed(items))
    assert hash_config(sample) == hash_config(reversed_payload)


@given(st.text(min_size=1, max_size=12), st.integers())
def test_hash_config_changes_on_value_update(key: str, value: int) -> None:
    base = {key: value}
    mutated = {key: value + 1}
    assert hash_config(base) != hash_config(mutated)
