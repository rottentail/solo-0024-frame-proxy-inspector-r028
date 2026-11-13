"""Proxy-triage probes supporting the two-stage inspection design.

These probes measure the PRODUCTION fingerprint and detector code paths
(`probe_avi`, `decode_frame_at`, `fingerprint_frame`, `fingerprint_distance`,
`analyze_frames`). No pipeline code is added here; the probes exist to ground
every quantitative claim in docs/proxy-two-stage-design.md so an independent
reviewer can recompute them.

Run:  PYTHONPATH=src:. python3 tools/check_proxy_probes.py
"""

from __future__ import annotations

import struct

from frame_proxy.media.avi import probe_avi
from frame_proxy.media.avi_writer import build_sample_avi
from frame_proxy.media.detector import analyze_frames
from frame_proxy.media.fingerprint import (
    fingerprint_distance,
    fingerprint_frame,
)
from frame_proxy.media.frames import Frame, decode_frame_at
from tools.harness import require, success

THRESHOLD_AT_055 = 0.50 - 0.55 * 0.30  # 0.335, production default
THRESHOLD_AT_095 = 0.50 - 0.95 * 0.30  # 0.215, most sensitive allowed


# ---------------------------------------------------------------------------
# Deterministic adversarial-fixture AVI builder (mirrors media/avi_writer.py)
# ---------------------------------------------------------------------------


def _chunk(tag: bytes, payload: bytes) -> bytes:
    padding = b"\x00" if len(payload) & 1 else b""
    return tag + struct.pack("<I", len(payload)) + payload + padding


def _list_chunk(kind: bytes, children: tuple[bytes, ...]) -> bytes:
    return _chunk(b"LIST", kind + b"".join(children))


def build_avi_from_frames(
    width: int,
    height: int,
    fps: int,
    frames_rgb: list[bytes],
) -> bytes:
    """Build a bottom-up 24-bit BGR AVI from top-down RGB frame buffers."""
    row_stride = ((width * 3 + 3) // 4) * 4
    frame_size = row_stride * height
    frame_count = len(frames_rgb)
    avih = struct.pack(
        "<14I",
        1_000_000 // fps,
        frame_size * fps,
        0,
        0x10,
        frame_count,
        0,
        1,
        frame_size,
        width,
        height,
        0,
        0,
        0,
        0,
    )
    strh = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        b"vids",
        b"DIB ",
        0,
        0,
        0,
        0,
        1,
        fps,
        0,
        frame_count,
        frame_size,
        0,
        frame_size,
        0,
        0,
        width,
        height,
    )
    strf = struct.pack(
        "<IiiHHIIiiII", 40, width, height, 1, 24, 0, frame_size, 2835, 2835, 0, 0
    )
    header_list = _list_chunk(
        b"hdrl",
        (
            _chunk(b"avih", avih),
            _list_chunk(b"strl", (_chunk(b"strh", strh), _chunk(b"strf", strf))),
        ),
    )
    media_chunks: list[bytes] = []
    for rgb in frames_rgb:
        payload = bytearray()
        for y in range(height - 1, -1, -1):
            row = rgb[y * width * 3 : (y + 1) * width * 3]
            bgr = bytearray(len(row))
            bgr[0::3] = row[2::3]
            bgr[1::3] = row[1::3]
            bgr[2::3] = row[0::3]
            payload.extend(bgr)
            payload.extend(b"\x00" * (row_stride - len(row)))
        media_chunks.append(_chunk(b"00db", bytes(payload)))
    body = b"AVI " + header_list + _list_chunk(b"movi", media_chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


# ---------------------------------------------------------------------------
# Frame fixtures and proxy constructions
# ---------------------------------------------------------------------------


def solid_frame(width: int, height: int, gray: int) -> bytes:
    return bytes((gray, gray, gray)) * (width * height)


def checker_frame(width: int, height: int, even: int, odd: int) -> bytes:
    """Gray checkerboard; parity (x+y)%2==0 takes `even`."""
    out = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            gray = even if (x + y) % 2 == 0 else odd
            offset = (y * width + x) * 3
            out[offset : offset + 3] = bytes((gray, gray, gray))
    return bytes(out)


def halves_frame(
    width: int,
    height: int,
    left_rgb: tuple[int, int, int],
    right_rgb: tuple[int, int, int],
) -> bytes:
    out = bytearray(width * height * 3)
    left = bytes(left_rgb)
    right = bytes(right_rgb)
    for y in range(height):
        for x in range(width):
            offset = (y * width + x) * 3
            out[offset : offset + 3] = left if x < width // 2 else right
    return bytes(out)


def subdecode(frame: Frame, stride: int) -> Frame:
    """Nearest-neighbour strided sub-decode: keeps every stride-th row/column.

    This is the pixel set a strided sub-decoder would read directly from the
    DIB payload, touching 1/stride**2 of the pixel bytes.
    """
    out = bytearray()
    for y in range(0, frame.height, stride):
        row = frame.rgb[y * frame.width * 3 : (y + 1) * frame.width * 3]
        for x in range(0, frame.width, stride):
            out += row[x * 3 : x * 3 + 3]
    return Frame(
        index=frame.index,
        width=len(range(0, frame.width, stride)),
        height=len(range(0, frame.height, stride)),
        rgb=bytes(out),
    )


def box_proxy(frame: Frame, factor: int) -> Frame:
    """Box-averaged downscale by an integer factor (reads every source pixel)."""
    proxy_width = frame.width // factor
    proxy_height = frame.height // factor
    out = bytearray(proxy_width * proxy_height * 3)
    for py in range(proxy_height):
        for px in range(proxy_width):
            totals = [0, 0, 0]
            for dy in range(factor):
                for dx in range(factor):
                    offset = ((py * factor + dy) * frame.width + (px * factor + dx)) * 3
                    for channel in range(3):
                        totals[channel] += frame.rgb[offset + channel]
            count = factor * factor
            destination = (py * proxy_width + px) * 3
            out[destination : destination + 3] = bytes(
                value // count for value in totals
            )
    return Frame(
        index=frame.index, width=proxy_width, height=proxy_height, rgb=bytes(out)
    )


def decode_all(media: bytes, width_stride: int = 1) -> list[Frame]:
    descriptor = probe_avi(media)
    return [
        decode_frame_at(media, descriptor, index)
        for index in range(0, descriptor.frame_count, width_stride)
    ]


def adjacent_distances(frames: list[Frame]) -> list[float]:
    prints = [fingerprint_frame(frame) for frame in frames]
    return [
        fingerprint_distance(left, right)
        for left, right in zip(prints, prints[1:])
    ]


def detect_cuts(
    frames: list[Frame],
    *,
    sensitivity: float,
    temporal_stride: int = 1,
    min_scene_frames: int = 2,
) -> tuple[int, ...]:
    sampled = tuple(range(0, len(frames), temporal_stride))
    prints = tuple(fingerprint_frame(frames[index]) for index in sampled)
    result = analyze_frames(
        len(frames),
        sampled,
        prints,
        sensitivity=sensitivity,
        min_scene_frames=min_scene_frames,
    )
    return result.cut_frames


# ---------------------------------------------------------------------------
# P1: friendly material — proxy and full resolution agree
# ---------------------------------------------------------------------------


def probe_p1() -> None:
    media = build_sample_avi()  # 128x72, scenes (5,4,5,3), cuts at 5, 9, 14
    frames = decode_all(media)
    require(len(frames) == 17, "P1: expected 17 demo frames")

    full_cuts = detect_cuts(frames, sensitivity=0.55)
    require(full_cuts == (5, 9, 14), f"P1: full-res cuts {full_cuts}")

    proxy2 = [subdecode(frame, 2) for frame in frames]
    proxy4 = [subdecode(frame, 4) for frame in frames]
    cuts2 = detect_cuts(proxy2, sensitivity=0.55)
    cuts4 = detect_cuts(proxy4, sensitivity=0.55)
    require(cuts2 == (5, 9, 14), f"P1: stride-2 proxy cuts {cuts2}")
    require(cuts4 == (5, 9, 14), f"P1: stride-4 proxy cuts {cuts4}")

    for label, resolution in (("full", frames), ("proxy-2", proxy2), ("proxy-4", proxy4)):
        distances = adjacent_distances(resolution)
        within = [d for i, d in enumerate(distances) if (i + 1) not in (5, 9, 14)]
        at_cuts = {i + 1: round(distances[i], 4) for i in (4, 8, 13)}
        print(
            f"  P1 {label:8s} max-within-scene={max(within):.4f} "
            f"cut-distances={at_cuts} threshold={THRESHOLD_AT_055:.3f}"
        )

    # Read-cost accounting in full-frame equivalents (pixel bytes touched).
    candidates = (5, 9, 14)
    confirm_window = 1
    confirm_frames = sorted(
        {
            index
            for cut in candidates
            for index in range(cut - confirm_window, cut + confirm_window + 1)
            if 0 <= index < len(frames)
        }
    )
    cost_full = float(len(frames))
    cost_two_stage = len(frames) / 16.0 + len(confirm_frames)
    print(
        f"  P1 cost: full-scan={cost_full:.1f}  two-stage="
        f"{len(frames)}/16 + {len(confirm_frames)} confirm = {cost_two_stage:.4f} "
        f"({100.0 * cost_two_stage / cost_full:.1f}% of full)"
    )
    success(
        "P1-friendly-material",
        {
            "cuts_full": list(full_cuts),
            "cuts_proxy4": list(cuts4),
            "two_stage_cost_pct": round(100.0 * cost_two_stage / cost_full, 1),
        },
    )


# ---------------------------------------------------------------------------
# P2: palette collision — a blind spot the proxy does not cause
# ---------------------------------------------------------------------------


def probe_p2() -> None:
    width, height = 64, 36
    red, blue = (200, 40, 40), (40, 40, 200)
    scene_a = halves_frame(width, height, red, blue)
    scene_b = halves_frame(width, height, blue, red)
    media = build_avi_from_frames(
        width, height, 12, [scene_a] * 8 + [scene_b] * 8
    )
    frames = decode_all(media)
    proxy2 = [subdecode(frame, 2) for frame in frames]

    boundary = adjacent_distances(frames)[7]
    boundary_proxy = adjacent_distances(proxy2)[7]
    mean_abs = sum(
        abs(a - b) for a, b in zip(frames[7].rgb, frames[8].rgb)
    ) / len(frames[7].rgb)
    cuts_full = detect_cuts(frames, sensitivity=0.95)
    cuts_proxy = detect_cuts(proxy2, sensitivity=0.95)
    require(cuts_full == (), f"P2: full-res cuts {cuts_full}")
    require(cuts_proxy == (), f"P2: proxy cuts {cuts_proxy}")
    print(
        f"  P2 boundary distance full={boundary:.4f} proxy={boundary_proxy:.4f} "
        f"while mean |pixel delta|={mean_abs:.1f}/255"
    )
    success(
        "P2-palette-collision",
        {"full_distance": round(boundary, 4), "proxy_distance": round(boundary_proxy, 4)},
    )


# ---------------------------------------------------------------------------
# P3: aliasing — proxy misses and proxy hallucinations on periodic texture
# ---------------------------------------------------------------------------


def probe_p3() -> None:
    width, height = 64, 36

    # (a) Miss: full-res catches a texture-vs-flat cut the stride-2 proxy loses.
    scene_a = checker_frame(width, height, 136, 120)  # subdecode samples 136
    scene_b = solid_frame(width, height, 128)
    media = build_avi_from_frames(
        width, height, 12, [scene_a] * 6 + [scene_b] * 6
    )
    frames = decode_all(media)
    proxy2 = [subdecode(frame, 2) for frame in frames]
    distance_full = adjacent_distances(frames)[5]
    distance_proxy = adjacent_distances(proxy2)[5]
    cuts_full = detect_cuts(frames, sensitivity=0.95)
    cuts_proxy = detect_cuts(proxy2, sensitivity=0.95)
    require(cuts_full == (6,), f"P3a: full-res cuts {cuts_full}")
    require(cuts_proxy == (), f"P3a: proxy cuts {cuts_proxy}")
    needed = (0.50 - distance_proxy) / 0.30
    require(
        needed > 0.95,
        f"P3a: sensitivity {needed:.2f} should be outside the allowed range",
    )
    miss_full = distance_full
    miss_proxy = distance_proxy
    print(
        f"  P3a miss: full={distance_full:.4f} (>={THRESHOLD_AT_095:.3f}) "
        f"proxy={distance_proxy:.4f}; sensitivity needed for proxy={needed:.2f} "
        f"(allowed max 0.95)"
    )

    # (b) Hallucination: phase-inverted checkerboards are identical to the
    # full-res fingerprint but look like a cut to the stride-2 proxy.
    scene_p0 = checker_frame(width, height, 120, 136)
    scene_p1 = checker_frame(width, height, 136, 120)
    media = build_avi_from_frames(
        width, height, 12, [scene_p0] * 6 + [scene_p1] * 6
    )
    frames = decode_all(media)
    proxy2 = [subdecode(frame, 2) for frame in frames]
    distance_full = adjacent_distances(frames)[5]
    distance_proxy = adjacent_distances(proxy2)[5]
    cuts_full = detect_cuts(frames, sensitivity=0.95)
    cuts_proxy = detect_cuts(proxy2, sensitivity=0.95)
    require(distance_full == 0.0, f"P3b: full-res distance {distance_full}")
    require(cuts_full == (), f"P3b: full-res cuts {cuts_full}")
    require(cuts_proxy == (6,), f"P3b: proxy cuts {cuts_proxy}")
    print(
        f"  P3b hallucination: full={distance_full:.4f} (no cut at any "
        f"sensitivity) proxy={distance_proxy:.4f} (candidate; confirmation "
        f"rejects it)"
    )

    # (c) Box-averaged proxy distorts the statistics themselves.
    box = box_proxy(frames[0], 2)
    print(
        f"  P3c box-proxy on checker(120,136): contrast "
        f"{fingerprint_frame(frames[0]).contrast:.2f} -> "
        f"{fingerprint_frame(box).contrast:.2f}, histogram "
        f"{[round(v, 3) for v in fingerprint_frame(frames[0]).luma_histogram]} -> "
        f"{[round(v, 3) for v in fingerprint_frame(box).luma_histogram]}"
    )
    success(
        "P3-aliasing",
        {
            "miss_full_distance": round(miss_full, 4),
            "miss_proxy_distance": round(miss_proxy, 4),
            "hallucination_proxy_distance": round(distance_proxy, 4),
            "note": "see printed P3a/P3b/P3c lines for measured values",
        },
    )


# ---------------------------------------------------------------------------
# P4: temporal stride — short scenes fall between sampled frames
# ---------------------------------------------------------------------------


def probe_p4() -> None:
    width, height = 64, 36
    scene_a = solid_frame(width, height, 60)
    scene_b = solid_frame(width, height, 200)
    scene_c = solid_frame(width, height, 120)
    media = build_avi_from_frames(
        width, height, 12, [scene_a] * 6 + [scene_b] * 2 + [scene_c] * 6
    )
    frames = decode_all(media)
    cuts_stride1 = detect_cuts(frames, sensitivity=0.55, temporal_stride=1)
    cuts_stride4 = detect_cuts(frames, sensitivity=0.55, temporal_stride=4)
    require(cuts_stride1 == (6, 8), f"P4: stride-1 cuts {cuts_stride1}")
    require(cuts_stride4 == (8,), f"P4: stride-4 cuts {cuts_stride4}")
    print(
        f"  P4 2-frame scene at [6,8): stride-1 cuts={list(cuts_stride1)} "
        f"stride-4 cuts={list(cuts_stride4)} (scene invisible, cut 6 lost, "
        f"boundary reported at 8)"
    )
    success(
        "P4-temporal-stride",
        {"stride1": list(cuts_stride1), "stride4": list(cuts_stride4)},
    )


# ---------------------------------------------------------------------------
# P5: equal-budget comparison — local re-check vs from-start precise scan
# ---------------------------------------------------------------------------


def simulate_budget(
    *,
    n_frames: int,
    cut_positions: tuple[int, ...],
    budget: float,
    proxy_cost: float,
    confirm_window: int,
    false_positions: tuple[int, ...],
    miss_positions: tuple[int, ...],
) -> dict[str, object]:
    # Strategy A: dense precise scan from frame 0 until the budget is spent.
    a_covered = min(int(budget), n_frames)
    a_proven = [c for c in cut_positions if c < a_covered]
    a_unexamined = [c for c in cut_positions if c >= a_covered]

    # Strategy B: proxy-scan everything, then confirm candidates in file order.
    remaining = budget - n_frames * proxy_cost
    candidates = sorted(
        (set(cut_positions) - set(miss_positions)) | set(false_positions)
    )
    confirmable = max(0, int(remaining // confirm_window))
    confirmed = candidates[:confirmable]
    b_proven = [c for c in confirmed if c in cut_positions]
    b_false_filtered = [c for c in confirmed if c in false_positions]
    b_unresolved = candidates[confirmable:]
    return {
        "budget": budget,
        "a_proven": len(a_proven),
        "a_unexamined": len(a_unexamined),
        "b_proven": len(b_proven),
        "b_false_filtered": len(b_false_filtered),
        "b_unresolved": len(b_unresolved),
        "b_missed_by_proxy": len(miss_positions),
        "b_cost": n_frames * proxy_cost + len(confirmed) * confirm_window,
    }


def probe_p5() -> None:
    n_frames = 3000  # service frame-count cap
    cuts = tuple(75 * i for i in range(1, 40))  # 39 cuts, evenly spread
    false_candidates = tuple(100 * j + 37 for j in range(30))  # aliasing-prone clip
    misses = (750, 1500, 2250)  # proxy-invisible cuts (adversarial scenario)
    proxy_cost = 1.0 / 16.0  # stride-4 spatial sub-decode
    confirm_window = 5  # candidate +/- 2 original frames

    print(
        "  P5 budget | A: proven | A: unexamined || "
        "B: proven | B: false-filtered | B: unresolved | B: proxy-missed | B: cost"
    )
    rows = []
    for budget in (300, 600, 1200, 3000):
        row = simulate_budget(
            n_frames=n_frames,
            cut_positions=cuts,
            budget=budget,
            proxy_cost=proxy_cost,
            confirm_window=confirm_window,
            false_positions=false_candidates,
            miss_positions=(),
        )
        rows.append(row)
        print(
            f"  P5 {budget:6d} | {row['a_proven']:9d} | {row['a_unexamined']:13d} ||"
            f" {row['b_proven']:8d} | {row['b_false_filtered']:15d} |"
            f" {row['b_unresolved']:13d} | {row['b_missed_by_proxy']:15d} |"
            f" {row['b_cost']:7.1f}"
        )
    friendly_600 = rows[1]
    require(friendly_600["b_proven"] == 39, "P5: B should prove all cuts at 600")
    require(friendly_600["a_proven"] == 7, "P5: A proves only prefix cuts at 600")

    adversarial = simulate_budget(
        n_frames=n_frames,
        cut_positions=cuts,
        budget=600,
        proxy_cost=proxy_cost,
        confirm_window=confirm_window,
        false_positions=false_candidates,
        miss_positions=misses,
    )
    require(adversarial["b_proven"] == 36, "P5: adversarial B proves 36 cuts")
    print(
        f"  P5 adversarial@600: B proves {adversarial['b_proven']} cuts, "
        f"{adversarial['b_missed_by_proxy']} cuts proxy-invisible "
        f"(positions {list(misses)}) -> must be reported unreviewed"
    )
    success(
        "P5-equal-budget",
        {
            "friendly": rows,
            "adversarial_600": adversarial,
        },
    )


def main() -> int:
    print("proxy-triage probes (production fingerprint/detector code paths)")
    probe_p1()
    probe_p2()
    probe_p3()
    probe_p4()
    probe_p5()
    print("ALL PROXY PROBES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
