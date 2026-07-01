"""SDK-local parameter interpreter — verdict + message parity with the backend.

These cases mirror ``mvp-backend/tests/validation/contracts/test_interpret.py``
so both runtimes assert the SAME verdicts and messages (the parity oracle for
Phase 6), plus the extra cases needed to drive full branch coverage of the port.
"""

from __future__ import annotations

from dagnam._contracts import validate_params
from dagnam._contracts._interpret import _check_number

# --- Padding (the motivating bug) -------------------------------------------


def test_bare_int_padding_emits_structured_payload() -> None:
    [e] = validate_params("convolution-layer", {"filters": 8, "kernelSize": 3, "padding": 2}, "n1")
    assert e.node_id == "n1"
    assert e.type == "parameter_error"
    assert e.severity == "error"
    assert e.code == "PARAM_PADDING_NOT_TYPED"
    assert e.field == "padding"
    assert e.expected == "{mode:'same'|'valid'|'explicit', value?}"
    assert e.got == "2"
    assert e.fix_hint == "wrap an explicit pad as {mode:'explicit', value:N}"
    assert e.message == (
        "convolution-layer: padding must be a typed object "
        "{mode:'same'|'valid'|'explicit', value?}, got 2"
    )


def test_typed_padding_passes() -> None:
    errs = validate_params(
        "convolution-layer",
        {"filters": 8, "kernelSize": 3, "padding": {"mode": "explicit", "value": 2}},
        "n1",
    )
    assert errs == []


def test_same_mode_dict_padding_passes() -> None:
    errs = validate_params(
        "convolution-layer",
        {"filters": 8, "kernelSize": 3, "padding": {"mode": "same"}},
        "n1",
    )
    assert errs == []


def test_legacy_string_padding_is_tolerated() -> None:
    # Parity with interpret.py: bare 'valid'/'same' are accepted (normalizer is Plan 2).
    assert (
        validate_params(
            "convolution-layer", {"filters": 8, "kernelSize": 3, "padding": "same"}, "n1"
        )
        == []
    )


def test_invalid_string_padding_is_rejected() -> None:
    errs = validate_params(
        "convolution-layer", {"filters": 8, "kernelSize": 3, "padding": "foo"}, "n1"
    )
    assert len(errs) == 1
    assert "must be 'valid', 'same'" in errs[0].message


def test_padding_dict_with_unknown_mode_is_rejected() -> None:
    errs = validate_params(
        "convolution-layer", {"filters": 8, "kernelSize": 3, "padding": {"mode": "bogus"}}, "n1"
    )
    assert len(errs) == 1
    assert ".mode must be" in errs[0].message


def test_explicit_padding_needs_non_negative_int() -> None:
    errs = validate_params(
        "convolution-layer",
        {"filters": 8, "kernelSize": 3, "padding": {"mode": "explicit", "value": -1}},
        "n1",
    )
    assert len(errs) == 1
    assert "non-negative integer" in errs[0].message


# --- Numeric constraints ----------------------------------------------------


def test_filters_below_min_rejected() -> None:
    errs = validate_params("convolution-layer", {"filters": 0, "kernelSize": 3}, "n2")
    assert any("filters" in e.message and "at least 1" in e.message for e in errs)


def test_filters_above_max_rejected() -> None:
    errs = validate_params("convolution-layer", {"filters": 5000, "kernelSize": 3}, "n2")
    assert any("filters" in e.message and "at most 2048" in e.message for e in errs)


def test_non_integer_value_rejected() -> None:
    errs = validate_params("convolution-layer", {"filters": 8.5, "kernelSize": 3}, "n2")
    assert any("filters" in e.message and "whole number" in e.message for e in errs)


def test_non_numeric_value_rejected() -> None:
    errs = validate_params("convolution-layer", {"filters": "lots", "kernelSize": 3}, "n2")
    assert any("filters" in e.message and "must be a number" in e.message for e in errs)


def test_number_below_recommended_emits_soft_warning() -> None:
    # Hard bounds pass (>= 1e-8) but the value is below the recommended floor
    # (warn_min 1e-5): a non-blocking advisory, not an error. Covers the SOFT
    # below-recommended branch of _check_number.
    [w] = validate_params("output-layer", {"learningRate": 1e-6}, "n2")
    assert w.code == "PARAM_NUMBER_BELOW_RECOMMENDED"
    assert w.severity == "warning"
    assert w.field == "learningRate"
    assert "unusually low" in w.message
    assert "recommended >= 1e-05" in w.message


def test_check_number_without_numeric_block_is_a_noop() -> None:
    # Defensive guard: a number-kind param always carries a numeric block in the
    # canonical schema, so validate_params never reaches this with nc is None;
    # exercise the branch directly for parity with interpret.py.
    assert _check_number(5, {"key": "x"}, "cid", "n2") == []


def test_missing_required_param_rejected() -> None:
    errs = validate_params("convolution-layer", {"kernelSize": 3}, "n2")
    assert any("filters" in e.message and "missing required" in e.message.lower() for e in errs)


def test_camelcase_kernel_size_is_resolved() -> None:
    errs = validate_params("convolution-layer", {"filters": 8, "kernelSize": 3}, "n2")
    assert errs == []


# --- Enums ------------------------------------------------------------------


def test_enum_rejects_unknown_value() -> None:
    errs = validate_params(
        "convolution-layer", {"filters": 8, "kernelSize": 3, "activation": "banana"}, "n3"
    )
    assert any("activation" in e.message and "one of" in e.message for e in errs)


def test_valid_enum_value_passes() -> None:
    errs = validate_params(
        "convolution-layer", {"filters": 8, "kernelSize": 3, "activation": "relu"}, "n3"
    )
    assert errs == []


def test_numeric_enum_option_accepts_int_value() -> None:
    assert validate_params("quantization-aware", {"bits": 8}, "n3") == []


# --- Other kinds (bool/string) fall through with no per-param check ----------


def test_bool_kind_param_is_not_validated() -> None:
    assert validate_params("batch-normalization", {"center": True, "scale": False}, "n3") == []


# --- Conditional applicability (weight decay) -------------------------------


def test_weight_decay_inactive_when_optimizer_is_adam() -> None:
    errs = validate_params("output-layer", {"optimizer": "adam", "weight_decay": -5}, "n4")
    assert errs == []


def test_weight_decay_inactive_when_optimizer_absent() -> None:
    # No controlling optimizer present -> the gated param is inactive (skipped).
    errs = validate_params("output-layer", {"weight_decay": -5}, "n4")
    assert errs == []


def test_weight_decay_validated_when_adamw() -> None:
    errs = validate_params("output-layer", {"optimizer": "adamw", "weight_decay": -5}, "n4")
    assert any("weight_decay" in e.message for e in errs)


def test_applies_when_is_case_insensitive() -> None:
    errs = validate_params("output-layer", {"optimizer": "AdamW", "weight_decay": -5}, "n4")
    assert any("weight_decay" in e.message for e in errs)


# --- Unknown components are out of scope ------------------------------------


def test_unknown_component_yields_no_errors() -> None:
    assert validate_params("does-not-exist", {"whatever": 1}, "n5") == []
