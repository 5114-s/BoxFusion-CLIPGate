#!/usr/bin/env python3
"""CPU-only unified history, raw-score provenance, and reachable-state audit."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_scannet_mechanism_controls as ev

core = ev.audit.core
from boxfusion.m1_anchor_online import OnlineAnchorRecovery, deterministic_tail_score


def reachable_bytes(value, seen=None):
    """CPython reachable size, counting aliased objects/buffers once per root."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    count = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        # getsizeof includes an owned allocation, but not a view's base.
        if value.base is not None:
            count += reachable_bytes(value.base, seen)
        return count
    if isinstance(value, dict):
        return count+sum(reachable_bytes(k, seen)+reachable_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list, set, frozenset)):
        return count+sum(reachable_bytes(x, seen) for x in value)
    if hasattr(value, "__dict__"):
        return count+reachable_bytes(vars(value), seen)
    return count


@dataclass
class Exemplar:
    box: np.ndarray
    raw: float
    frame: int
    source_id: int
    source: str

    @property
    def rank(self):
        return (-self.raw, self.frame, self.source_id)


@dataclass
class Cell:
    support: int = 0
    last_frame: int = -1
    last_ordinal: int = -1
    proposal: Exemplar | None = None
    anchor: Exemplar | None = None

    def add(self, ordinal, observation):
        if self.last_frame != observation.frame:
            self.support = min(3, self.support+1)
            self.last_frame = observation.frame
        self.last_ordinal = ordinal
        previous = getattr(self, observation.source)
        if previous is None or observation.rank < previous.rank:
            setattr(self, observation.source, observation)


def canonical_observations(evidence, frame):
    """Deduplicate exact geometry per source; union carries both source tags."""
    groups = {"proposal": {}, "anchor": {}}
    dropped = {"proposal": 0, "anchor": 0, "cross_source_identical": 0}
    for source in groups:
        ids = getattr(evidence, f"{source}_ids")
        boxes = getattr(evidence, f"{source}_corners")
        scores = getattr(evidence, f"{source}_scores")
        for index, (source_id, box, score) in enumerate(zip(ids, boxes, scores)):
            signature = ev.geometry_signature(box)
            item = (Exemplar(np.array(box, copy=True), float(score), int(frame), int(source_id), source), index)
            old = groups[source].get(signature)
            if old is None or item[0].rank < old[0].rank:
                groups[source][signature] = item
        dropped[source] = len(ids)-len(groups[source])
    dropped["cross_source_identical"] = len(groups["proposal"].keys() & groups["anchor"].keys())
    observations = sorted([item[0] for group in groups.values() for item in group.values()],
                          key=lambda row: (-row.raw, row.source, row.source_id))
    pidx = sorted(item[1] for item in groups["proposal"].values())
    aidx = sorted(item[1] for item in groups["anchor"].values())
    canonical = core.EvidenceFrame(
        proposal_ids=evidence.proposal_ids[pidx], proposal_boxes_2d=evidence.proposal_boxes_2d[pidx],
        proposal_corners=evidence.proposal_corners[pidx], proposal_scores=evidence.proposal_scores[pidx],
        anchor_ids=evidence.anchor_ids[aidx], anchor_corners=evidence.anchor_corners[aidx],
        anchor_scores=evidence.anchor_scores[aidx])
    return canonical, observations, dropped


class UnifiedVoxelHistory:
    """One support/eviction pool; source tags only retain score-compatible reps."""
    def __init__(self, scene, active_cap=5120, confirmed_cap=1664):
        self.scene = scene
        self.active_cap, self.confirmed_cap = active_cap, confirmed_cap
        self.active, self.confirmed = {}, {}
        self.frames_seen = 0
        self.dropped_active = self.dropped_confirmed = 0

    def update(self, ordinal, observations):
        assert ordinal == self.frames_seen
        for observation in observations:
            key = core.voxel_key(observation.box)
            current = self.confirmed.get(key)
            if current is not None:
                current.add(ordinal, observation)
                continue
            current = self.active.setdefault(key, Cell())
            current.add(ordinal, observation)
            if current.support >= 3:
                if len(self.confirmed) < self.confirmed_cap:
                    self.confirmed[key] = current
                else:
                    self.dropped_confirmed += 1
                del self.active[key]
        if len(self.active) > self.active_cap:
            def priority(item):
                key, value = item
                raw = max(r.raw for r in (value.proposal, value.anchor) if r is not None)
                return (value.support, value.last_ordinal, raw, tuple(-x for x in key))
            remove = sorted(self.active.items(), key=priority)[:len(self.active)-self.active_cap]
            for key, _ in remove:
                del self.active[key]
            self.dropped_active += len(remove)
        self.frames_seen += 1
        assert len(self.active) <= self.active_cap and len(self.confirmed) <= self.confirmed_cap

    def rows(self, native):
        proposals = [c.proposal for c in self.confirmed.values() if c.proposal is not None]
        proposals.sort(key=lambda r: (r.rank, core.voxel_key(r.box)))
        if proposals and native:
            overlap = core._aabb_iou_matrix(np.stack([p.box for p in proposals]), np.stack([n[1] for n in native]))
            proposals = [p for p, drop in zip(proposals, np.any(overlap >= .25, axis=1)) if not drop]
        chosen = []
        for row in proposals:
            if chosen and np.any(core._aabb_iou_matrix(row.box[None], np.stack([r.box for r in chosen])) >= .5):
                continue
            chosen.append(row)
            if len(chosen) >= 12:
                break
        p = [(0, r.box.copy(), float(core._size_price(r.box))) for r in chosen]
        a = [(r, deterministic_tail_score(r.raw, self.scene, r.frame, r.source_id))
             for c in self.confirmed.values() if (r := c.anchor) is not None]
        a.sort(key=lambda item: (-item[1], core.voxel_key(item[0].box)))
        return p, [(0, r.box.copy(), float(score)) for r, score in a[:640]]


def exact_readout_dedup(value):
    seen, keep = set(), []
    for index, row in enumerate(value["rows"]):
        key = ev.geometry_signature(row[1])
        if key not in seen:
            seen.add(key)
            keep.append(index)
    reduced = ev.case([value["rows"][i] for i in keep], [value["sources"][i] for i in keep])
    return reduced, len(value["rows"])-len(keep)


def rescore(value, raw_scores, variant):
    assert len(value["rows"]) == len(raw_scores)
    rows = []
    for row, source, raw in zip(value["rows"], value["sources"], raw_scores):
        score = float(row[2])
        if source != "native":
            score = float(raw)
            if variant == "temperature2":
                clipped = np.clip(score, 1e-6, 1-1e-6)
                score = float(1/(1+np.exp(-.5*np.log(clipped/(1-clipped)))))
        rows.append((row[0], row[1], score))
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(rows, value["rows"]))
    return ev.case(rows, value["sources"])


def update_plr_provenance(state, metadata):
    for identity, birth in state.births.items():
        previous = metadata.get(identity)
        if previous is not None and previous[0] is birth:
            continue
        track = state.tracks[identity]
        observation = state._medoid(track)
        assert np.array_equal(observation.corners, birth.corners)
        metadata[identity] = (birth, float(observation.score), observation.frame_id, observation.source_id)


def same_rows(a, b):
    return len(a) == len(b) and all(x[0] == y[0] and x[2] == y[2] and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_unified():
    box = np.asarray([[i, j, k] for i in (0., .1) for j in (0., .1) for k in (0., .1)])
    history = UnifiedVoxelHistory("test")
    for ordinal in range(3):
        obs = [Exemplar(box, .7, ordinal*25, 1, "proposal"), Exemplar(box, .8, ordinal*25, 2, "anchor")]*3
        history.update(ordinal, obs)
        assert len(history.confirmed) == int(ordinal == 2)
    item = next(iter(history.confirmed.values()))
    assert item.support == 3
    history.update(3, [Exemplar(box, .9, 75, 3, "anchor")])
    assert item.support == 3 and item.anchor.raw == .9
    p, a = history.rows([])
    assert len(p) == len(a) == 1
    reduced, n = exact_readout_dedup(ev.case(p+a, ["proposal", "anchor"]))
    assert n == 1 and len(reduced["rows"]) == 1
    array = np.zeros(1000)
    assert reachable_bytes([array, array]) < reachable_bytes([array, array.copy()])


def replay(out, protocol, scenes):
    config = yaml.safe_load((ROOT/protocol["state_config"]).read_text())["online_candidate_map"]["state"]
    config = {**config, "mvsr_only": False, "selected_calr_v1": True, "audit_controls": False,
              "audit_direct_plr": False, "audit_raw_iou_max": False, "audit_mvsr_factors": False,
              "audit_plr_controls": False}
    config["m2"] = {**config["m2"], "strength_variant": "no_visibility", "support_mode": "max"}
    parent = ROOT/protocol["parent_report"]
    protocol_hash = core.sha256(out/"protocol.json")
    output = out/"replay"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol_sha256": protocol_hash, "scenes": {}, "new_model_inference": False}
    for index, scene in enumerate(scenes, 1):
        path = output/f"{scene}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                existing = pickle.load(handle)
            assert existing["protocol_sha256"] == protocol_hash
            manifest["scenes"][scene] = existing["summary"]
            continue
        started = time.perf_counter()
        natives = core.load_native(ROOT/protocol["native_prefix_root"]/f"{scene}.npz")
        evidences = core.load_evidence(ROOT/protocol["evidence_root"]/f"{scene}.npz")
        assert [f for f, _ in natives] == [f for f, _ in evidences]
        full = core.build_reliability_candidate_map(scene, config)
        canonical_plr = core.build_reliability_candidate_map(scene, config).m1p
        canonical_anchor = OnlineAnchorRecovery(scene)
        unified = UnifiedVoxelHistory(scene)
        proposal = core.ProposalScoreMemory(active_capacity=1024, archive_capacity=1024,
                                            ttl_keyframes=10, output_capacity=12)
        anchor = core.AnchorScoreMemory(capacity=4096, output_capacity=640)
        plr_metadata, prefixes, peaks, trace = {}, {}, {}, []
        exact_input_duplicates = {"proposal": 0, "anchor": 0, "cross_source_identical": 0}
        checkpoints = {int(np.ceil(len(natives)*f))-1: str(f) for f in (.25, .5, .75, 1.)}
        for ordinal, ((frame, native), (_, evidence)) in enumerate(zip(natives, evidences)):
            full.update(frame, native, evidence)
            update_plr_provenance(full.m1p, plr_metadata)
            proposal.update(ordinal, frame, evidence.proposal_ids, evidence.proposal_corners, evidence.proposal_scores)
            anchor.update(ordinal, frame, evidence.anchor_ids, evidence.anchor_corners, evidence.anchor_scores)
            canonical, observations, dropped = canonical_observations(evidence, frame)
            for key in dropped:
                exact_input_duplicates[key] += dropped[key]
            canonical_plr.update(ordinal, frame, proposal_ids=canonical.proposal_ids,
                                 proposal_corners=canonical.proposal_corners, proposal_scores=canonical.proposal_scores,
                                 native_corners=native.corners)
            canonical_anchor.update(ordinal, frame, canonical.anchor_ids, canonical.anchor_corners, canonical.anchor_scores)
            unified.update(ordinal, observations)
            keep = np.all(np.ptp(native.corners, axis=1) >= .3, axis=1)
            native_rows = [(0, np.array(box, copy=True), float(score)) for box, score in zip(native.corners[keep], native.scores[keep])]
            snapshot = full.materialize_terminal(frame, native.ids[keep], native.corners[keep], native.scores[keep])
            full_rows = ev.audit.rows_from_snapshot(snapshot)
            full_case = ev.case(full_rows, ["native" if s == "native" else "proposal" if s == "m1p" else "anchor" for s in snapshot.sources])
            n = len(native_rows)
            native_hcsr = full_rows[:n]
            pp, aa = proposal.rows(native_rows), anchor.rows(scene)[:256]
            memory = ev.case(native_hcsr+pp+aa, ["native"]*n+["proposal"]*len(pp)+["anchor"]*len(aa))
            cp = [(0, r["box"], r["score"]) for r in canonical_plr.terminal_rows(native.corners[keep])]
            ca = [(0, r["box"], r["score"]) for r in canonical_anchor.rows()]
            common_layered, common_removed = exact_readout_dedup(ev.case(native_hcsr+cp+ca, ["native"]*n+["proposal"]*len(cp)+["anchor"]*len(ca)))
            up, ua = unified.rows(native_rows)
            common_unified, unified_removed = exact_readout_dedup(ev.case(native_hcsr+up+ua, ["native"]*n+["proposal"]*len(up)+["anchor"]*len(ua)))
            assert sum(s == "native" for s in common_layered["sources"]) == sum(s == "native" for s in common_unified["sources"])
            roots = {
                "full": (full.m1p.tracks, full.m1p.births, full.m1p.retired_births, full.m1a_control.active, full.m1a_control.births),
                "memory": (proposal.active, proposal.archive, anchor.state),
                "layered_common": (canonical_plr.tracks, canonical_plr.births, canonical_plr.retired_births, canonical_anchor.active, canonical_anchor.births),
                "unified_common": (unified.active, unified.confirmed),
            }
            byte_counts = {key: reachable_bytes(value) for key, value in roots.items()}
            serialized = {key: len(pickle.dumps(value, protocol=5)) for key, value in roots.items()}
            entries = {"full": len(full.m1p.tracks)+len(full.m1p.births)+len(full.m1a_control.active)+len(full.m1a_control.births),
                       "memory": len(proposal.active)+len(proposal.archive)+len(anchor.state),
                       "layered_common": len(canonical_plr.tracks)+len(canonical_plr.births)+len(canonical_anchor.active)+len(canonical_anchor.births),
                       "unified_common": len(unified.active)+len(unified.confirmed)}
            for name in roots:
                for kind, values in (("reachable_bytes", byte_counts), ("serialized_bytes", serialized), ("entries", entries)):
                    peak_key = f"{name}_{kind}"
                    peaks[peak_key] = max(peaks.get(peak_key, 0), values[name])
            trace.append({"frame": frame, "ordinal": ordinal, "reachable_bytes": byte_counts,
                          "serialized_bytes": serialized, "entries": entries,
                          "unified_output": len(common_unified["rows"]), "layered_output": len(common_layered["rows"])})
            if ordinal not in checkpoints:
                continue
            f = checkpoints[ordinal]; tag = f"p{int(float(f)*100)}"
            expected = ev.audit.read_rows(parent/"predictions"/f"full_{tag}"/f"{scene}_boxes.pkl")
            assert same_rows(full_rows, expected), (scene, f, "full cache reproduction mismatch")
            expected_memory = ev.audit.read_rows(parent/"predictions"/f"memory_a256_{tag}"/f"{scene}_boxes.pkl")
            assert same_rows(memory["rows"], expected_memory), (scene, f, "memory cache reproduction mismatch")
            full_plr = full.m1p.terminal_rows(native.corners[keep])
            full_anchor = full.m1a_control.rows()
            raw_full = [r[2] for r in native_hcsr]+[plr_metadata[r["track_id"]][1] for r in full_plr]+[r["raw_score"] for r in full_anchor]
            lookup_p = {ev.geometry_signature(r.box): r.raw_score for r in proposal.archive.values()}
            lookup_a = {ev.geometry_signature(r.box): r.raw_score for r in anchor.state.values()}
            raw_memory = [r[2] for r in native_hcsr]+[lookup_p[ev.geometry_signature(r[1])] for r in pp]+[lookup_a[ev.geometry_signature(r[1])] for r in aa]
            prefixes[f] = {"full_current": full_case, "full_raw": rescore(full_case, raw_full, "raw"),
                           "full_temperature2": rescore(full_case, raw_full, "temperature2"),
                           "memory_current": memory, "memory_raw": rescore(memory, raw_memory, "raw"),
                           "memory_temperature2": rescore(memory, raw_memory, "temperature2"),
                           "layered_common": common_layered, "unified_common": common_unified,
                           "exact_readout_removed": {"layered_common": common_removed, "unified_common": unified_removed}}
        summary = {"keyframes": len(natives), "peaks": peaks, "exact_input_duplicates": exact_input_duplicates,
                   "diagnostic_calr_log_serialized_bytes": len(pickle.dumps(full.m1a_control.events, protocol=5)),
                   "seconds": time.perf_counter()-started, "trace": trace}
        payload = {"protocol_sha256": protocol_hash, "summary": summary, "prefixes": prefixes}
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=5)
        temporary.replace(path)
        manifest["scenes"][scene] = summary
        ev.save_json(out/"replay_manifest.partial.json", {**manifest, "completed_scenes": index})
        if index % 5 == 0:
            print(f"unified/provenance/state replay {index}/{len(scenes)}", flush=True)
    ev.save_json(out/"replay_manifest.json", manifest)
    print("CPU replay complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ev.DEFAULT)
    args = parser.parse_args()
    test_unified()
    protocol = json.loads((args.output/"protocol.json").read_text())
    scenes = [s.strip() for s in (ROOT/protocol["scene_list"]).read_text().splitlines() if s.strip() and not s.startswith("#")]
    assert len(scenes) == 100
    replay(args.output, protocol, scenes)


if __name__ == "__main__":
    main()
