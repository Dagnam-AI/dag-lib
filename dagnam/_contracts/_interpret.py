"""SDK-local mirror of the backend's declarative parameter interpreter.

Verdicts and messages match mvp-backend/src/validation/contracts/interpret.py
exactly (the test cases are shared). Scope is the declarative param tier only:
numeric bounds, enums, typed padding, required-ness, conditional applicability.
Cross-parameter / shape rules are out of scope (a later plan).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dagnam._contracts._schema import COMPONENT_REGISTRY, DIAGNOSTICS, render

_PADDING_MODES = ("valid", "same", "explicit")
_PADDING_SHAPE = "{mode:'same'|'valid'|'explicit', value?}"

# Severity -> diagnostic type. Error-severity keeps the historical
# ``parameter_error`` type; advisory severities get distinct types that are NOT
# spec rule ids, so they stay out of the count-stable corpus and never block.
_SEVERITY_TYPE = {
    "error": "parameter_error",
    "warning": "parameter_warning",
    "info": "parameter_info",
}


@dataclass(frozen=True)
class ParamError:
    """A single declarative-parameter validation failure.

    Carries the structured diagnostic payload (spec §6) rendered from the same
    shipped catalog the backend and frontend use, so all three runtimes emit
    byte-identical messages.
    """

    type: str
    message: str
    node_id: str | None
    severity: str
    code: str | None = None
    field: str | None = None
    expected: str | None = None
    got: str | None = None
    fix_hint: str | None = None


def _diag(
    code: str,
    node_id: str,
    *,
    component_id: str,
    field: str = "",
    expected: str = "",
    got: str = "",
) -> ParamError:
    message, fix_hint = render(
        code, component_id=component_id, field=field, expected=expected, got=got
    )
    severity = DIAGNOSTICS[code].get("severity", "error")
    return ParamError(
        type=_SEVERITY_TYPE[severity],
        message=message,
        node_id=node_id,
        severity=severity,
        code=code,
        field=field or None,
        expected=expected or None,
        got=got or None,
        fix_hint=fix_hint or None,
    )


def _repr(value: Any) -> str:
    return f"{value!r}"


def _snake_to_camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _camel_to_snake(key: str) -> str:
    out: list[str] = []
    for ch in key:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _variants(name: str) -> tuple[str, str, str]:
    return name, _snake_to_camel(name), _camel_to_snake(name)


def _candidate_keys(param: Mapping[str, Any]) -> list[str]:
    names = [param["key"], *(param.get("aliases") or [])]
    candidates: list[str] = []
    for name in names:
        for variant in _variants(name):
            if variant not in candidates:
                candidates.append(variant)
    return candidates


def _resolve(config: Mapping[str, Any], param: Mapping[str, Any]) -> tuple[bool, Any]:
    """Return ``(present, value)``. A key whose value is ``None`` counts as absent."""
    for candidate in _candidate_keys(param):
        if candidate in config and config[candidate] is not None:
            return True, config[candidate]
    return False, None


def _control_value(config: Mapping[str, Any], control_key: str) -> Any:
    for candidate in _variants(control_key):
        if candidate in config and config[candidate] is not None:
            return config[candidate]
    return None


def _param_active(param: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    applies_when = param.get("applies_when")
    if not applies_when:
        return True
    for control_key, allowed in applies_when.items():
        allowed_lower = {str(a).lower() for a in allowed}
        if str(_control_value(config, control_key)).lower() not in allowed_lower:
            return False
    return True


def _check_padding(value: Any, key: str, cid: str, node_id: str) -> list[ParamError]:
    if isinstance(value, str):
        if value in ("valid", "same"):
            return []
        return [
            _diag(
                "PARAM_PADDING_BAD_STRING",
                node_id,
                component_id=cid,
                field=key,
                expected=_PADDING_SHAPE,
                got=_repr(value),
            )
        ]
    if isinstance(value, dict):
        mode = value.get("mode")
        if mode not in _PADDING_MODES:
            return [
                _diag(
                    "PARAM_PADDING_BAD_MODE", node_id, component_id=cid, field=key, got=_repr(mode)
                )
            ]
        if mode == "explicit":
            v = value.get("value")
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                return [
                    _diag(
                        "PARAM_PADDING_BAD_EXPLICIT_VALUE",
                        node_id,
                        component_id=cid,
                        field=key,
                        got=_repr(v),
                    )
                ]
        return []
    return [
        _diag(
            "PARAM_PADDING_NOT_TYPED",
            node_id,
            component_id=cid,
            field=key,
            expected=_PADDING_SHAPE,
            got=_repr(value),
        )
    ]


def _fmt(x: float) -> str:
    return f"{x:g}"


def _check_number(value: Any, param: Mapping[str, Any], cid: str, node_id: str) -> list[ParamError]:
    nc = param.get("numeric")
    if nc is None:
        return []
    # bool is an int subclass in Python; a checkbox value is never a valid number.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [
            _diag(
                "PARAM_NUMBER_NOT_A_NUMBER",
                node_id,
                component_id=cid,
                field=param["key"],
                got=_repr(value),
            )
        ]
    lo, hi, integer = nc.get("min"), nc.get("max"), nc.get("integer", False)
    if integer and not float(value).is_integer():
        return [
            _diag(
                "PARAM_NUMBER_NOT_INTEGER",
                node_id,
                component_id=cid,
                field=param["key"],
                got=_repr(value),
            )
        ]
    if lo is not None and value < lo:
        return [
            _diag(
                "PARAM_NUMBER_BELOW_MIN",
                node_id,
                component_id=cid,
                field=param["key"],
                expected=_fmt(lo),
                got=_repr(value),
            )
        ]
    if hi is not None and value > hi:
        return [
            _diag(
                "PARAM_NUMBER_ABOVE_MAX",
                node_id,
                component_id=cid,
                field=param["key"],
                expected=_fmt(hi),
                got=_repr(value),
            )
        ]
    # Hard bounds passed: advisory SOFT bounds (non-blocking warnings).
    warn_lo, warn_hi = nc.get("warn_min"), nc.get("warn_max")
    if warn_lo is not None and value < warn_lo:
        return [
            _diag(
                "PARAM_NUMBER_BELOW_RECOMMENDED",
                node_id,
                component_id=cid,
                field=param["key"],
                expected=_fmt(warn_lo),
                got=_repr(value),
            )
        ]
    if warn_hi is not None and value > warn_hi:
        return [
            _diag(
                "PARAM_NUMBER_ABOVE_RECOMMENDED",
                node_id,
                component_id=cid,
                field=param["key"],
                expected=_fmt(warn_hi),
                got=_repr(value),
            )
        ]
    return []


def validate_params(component_id: str, config: Mapping[str, Any], node_id: str) -> list[ParamError]:
    """Validate *config* against the canonical schema for *component_id*.

    Unknown components are out of contract scope, so they yield no errors here.
    """
    spec = COMPONENT_REGISTRY.get(component_id)
    if spec is None:
        return []
    errors: list[ParamError] = []
    for param in spec["params"]:
        if not _param_active(param, config):
            continue
        present, value = _resolve(config, param)
        if not present:
            if param.get("required"):
                errors.append(
                    _diag(
                        "PARAM_REQUIRED_MISSING",
                        node_id,
                        component_id=component_id,
                        field=param["key"],
                    )
                )
            continue
        kind = param["kind"]
        if kind == "padding":
            errors.extend(_check_padding(value, param["key"], component_id, node_id))
        elif kind == "number":
            errors.extend(_check_number(value, param, component_id, node_id))
        elif kind == "enum":
            enum_values = param.get("enum_values")
            if enum_values is not None and str(value) not in enum_values:
                errors.append(
                    _diag(
                        "PARAM_ENUM_NOT_ALLOWED",
                        node_id,
                        component_id=component_id,
                        field=param["key"],
                        expected=str(enum_values),
                        got=_repr(value),
                    )
                )
        # Categorical advisories (non-blocking info/warning), kind-independent.
        present_value = str(value).lower()
        for adv in param.get("advisories") or []:
            if str(adv["when_value"]).lower() == present_value:
                errors.append(
                    _diag(
                        adv["code"],
                        node_id,
                        component_id=component_id,
                        field=param["key"],
                        got=_repr(value),
                    )
                )
    return errors
