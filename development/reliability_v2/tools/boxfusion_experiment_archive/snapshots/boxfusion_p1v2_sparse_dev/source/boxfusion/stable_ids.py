"""Deterministic stable identifiers for overlapping BoxFusion groups.

BoxFusion normally gives every global box a fusion group whose minimum source
proposal id is unique.  Rare association paths can leave two live groups with
the same source members, however.  Using the minimum member directly then
aliases two different boxes to one online-memory identity.

The resolver below is deliberately a no-op for the normal unique-minimum
case.  On a collision it keeps the minimum id for one canonical group and
prefers a source id that occurs only in each remaining group.  A deterministic
high-range synthetic id is used only when the groups have no distinct source
member at all.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Iterable, Sequence, Tuple

import numpy as np


_SYNTHETIC_ID_START = 1 << 62
_SYNTHETIC_ID_STOP = (1 << 63) - 1


def _normalize_group(group: Iterable[int], index: int) -> Tuple[int, ...]:
    if isinstance(group, (str, bytes)):
        raise TypeError(f"fusion group {index} must be an integer sequence")
    try:
        values = tuple(group)
    except TypeError as exc:
        raise TypeError(
            f"fusion group {index} must be an integer sequence"
        ) from exc
    if not values:
        raise ValueError(f"fusion group {index} must not be empty")

    normalized = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(
                f"fusion group {index} must contain only integers"
            )
        value = int(value)
        if value < 0:
            raise ValueError(
                f"fusion group {index} ids must be non-negative"
            )
        if value >= _SYNTHETIC_ID_START:
            raise ValueError(
                f"fusion group {index} id exceeds the source-id range"
            )
        normalized.append(value)
    return tuple(sorted(set(normalized)))


def _synthetic_id(
    *,
    base_id: int,
    group: Tuple[int, ...],
    ordinal: int,
    forbidden: set[int],
) -> int:
    payload = (
        f"{base_id}:{ordinal}:" + ",".join(str(value) for value in group)
    ).encode("ascii")
    digest = int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )
    span = _SYNTHETIC_ID_STOP - _SYNTHETIC_ID_START + 1
    candidate = _SYNTHETIC_ID_START + digest % span
    while candidate in forbidden:
        candidate += 1
        if candidate > _SYNTHETIC_ID_STOP:
            candidate = _SYNTHETIC_ID_START
    return candidate


def resolve_fusion_stable_ids(
    fusion_groups: Sequence[Iterable[int]],
) -> np.ndarray:
    """Return one deterministic, unique int64 id per fusion group.

    If the group minima are already unique, the returned values are exactly
    those minima.  This guarantees that enabling the collision repair cannot
    change existing successful scenes.
    """

    if isinstance(fusion_groups, (str, bytes)) or not isinstance(
        fusion_groups, Sequence
    ):
        raise TypeError("fusion_groups must be a sequence")
    groups = tuple(
        _normalize_group(group, index)
        for index, group in enumerate(fusion_groups)
    )
    if not groups:
        return np.empty(0, dtype=np.int64)

    raw_ids = np.asarray([group[0] for group in groups], dtype=np.int64)
    base_counts = Counter(int(value) for value in raw_ids)
    if all(count == 1 for count in base_counts.values()):
        return raw_ids

    member_counts = Counter(
        member for group in groups for member in group
    )
    reserved_base_ids = set(base_counts)
    resolved = raw_ids.copy()
    used_ids = {
        base_id
        for base_id, count in base_counts.items()
        if count == 1
    }

    for base_id in sorted(
        value for value, count in base_counts.items() if count > 1
    ):
        indices = [
            index
            for index, value in enumerate(raw_ids)
            if int(value) == base_id
        ]
        indices.sort(key=lambda index: (groups[index], index))

        keeper = indices[0]
        resolved[keeper] = base_id
        used_ids.add(base_id)
        for ordinal, index in enumerate(indices[1:], start=1):
            group = groups[index]
            candidates = [
                member
                for member in group
                if member_counts[member] == 1
                and member not in used_ids
                and member not in reserved_base_ids
            ]
            if not candidates:
                candidates = [
                    member
                    for member in group
                    if member not in used_ids
                    and member not in reserved_base_ids
                ]
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda member: (member_counts[member], member),
                )
            else:
                chosen = _synthetic_id(
                    base_id=base_id,
                    group=group,
                    ordinal=ordinal,
                    forbidden=used_ids | reserved_base_ids,
                )
            resolved[index] = chosen
            used_ids.add(chosen)

    if len(set(int(value) for value in resolved)) != len(resolved):
        raise RuntimeError("failed to resolve unique fusion stable ids")
    return resolved


__all__ = ["resolve_fusion_stable_ids"]
