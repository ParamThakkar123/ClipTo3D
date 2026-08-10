"""The guided-capture client's JavaScript (MPO-247).

Capture is where reconstruction quality is decided, and the coaching logic is
the whole value of this client — so it is tested against synthetic frames and
scripted sensor readings rather than trusted because it looks reasonable.

The blur metric is deliberately the same one `keyframes.py` uses server-side,
and that correspondence is tested directly: coaching against a different
metric than the one which later judges the footage would be worse than no
coaching at all.

Skipped when Node is unavailable; nothing here needs a camera or a browser.
"""

import shutil

import numpy as np
import pytest

from tests.test_viewer import REPO, module_from, run_node  # noqa: F401

CAPTURE = REPO / "viewer" / "capture.html"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="needs Node to run the capture client's JS"
)

W, H = 96, 54


def capture_source() -> str:
    html = CAPTURE.read_text(encoding="utf-8")
    start = html.index('<script type="module">') + len('<script type="module">')
    return html[start:html.rindex("</script>")]


def metrics_module() -> str:
    """The pure metric and coaching functions, sliced above the DOM wiring."""
    src = capture_source()
    body = src[src.index("// --- metrics"):src.index("// --- wiring")]
    return module_from(
        body, "varianceOfLaplacian", "frameMotion", "isRotationDominated",
        "coverageFraction", "headingBucket", "captureAdvice", "captureVerdict",
        "uploadResumable", "HEADING_BUCKETS",
    )


def js_array(a) -> str:
    return "[" + ",".join(f"{v:.6g}" for v in np.asarray(a).ravel()) + "]"


@pytest.fixture(scope="module")
def frame_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("frames")


# --- sharpness ------------------------------------------------------------

def checkerboard(w=W, h=H, cell=4, amplitude=255.0):
    y, x = np.mgrid[0:h, 0:w]
    return (((x // cell + y // cell) % 2) * amplitude).astype(np.float64)


def blurred(img, passes=6):
    """Repeated box blur — what a fast pan does to a frame."""
    out = img.copy()
    for _ in range(passes):
        p = np.pad(out, 1, mode="edge")
        out = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + p[1:-1, 1:-1]) / 5.0
    return out


def sharpness_js(img, frame_dir, name="frame") -> float:
    """Run the shipped metric over an image, passed as a file.

    Not inlined into the snippet: a 96x54 frame is 5,184 numbers and Windows
    refuses the resulting command line.
    """
    path = frame_dir / f"{name}.f32"
    path.write_bytes(np.asarray(img, dtype=np.float32).tobytes())
    mod = metrics_module()
    return run_node(f"""
        import {{ readFileSync }} from 'node:fs';
        const {{ varianceOfLaplacian }} = await import('{mod}');
        const raw = readFileSync(process.argv[1]);
        const g = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
        console.log(JSON.stringify({{ v: varianceOfLaplacian(g, {W}, {H}) }}));
    """, str(path))["v"]


def test_sharp_frame_scores_far_above_a_blurred_one(frame_dir):
    sharp = sharpness_js(checkerboard(), frame_dir, "sharp")
    soft = sharpness_js(blurred(checkerboard()), frame_dir, "soft")
    assert sharp > soft * 20, (sharp, soft)


def test_flat_frame_has_no_sharpness(frame_dir):
    assert sharpness_js(np.full((H, W), 128.0), frame_dir, "flat") == pytest.approx(0, abs=1e-9)


def test_javascript_sharpness_matches_the_python_keyframe_metric(frame_dir):
    """The client must coach against the metric that later judges the frames.

    `keyframes.py` uses variance of the Laplacian via OpenCV; this is the same
    kernel written out in JS. They will not agree bit-for-bit (OpenCV's border
    handling differs) but they must rank frames identically and agree closely
    on the interior.
    """
    cv2 = pytest.importorskip("cv2")

    rng = np.random.default_rng(0)
    images = [checkerboard(), blurred(checkerboard(), 2), blurred(checkerboard(), 8),
              rng.random((H, W)) * 255.0, np.full((H, W), 40.0)]

    js = [sharpness_js(im, frame_dir, f"cmp{i}") for i, im in enumerate(images)]
    py = []
    for im in images:
        lap = cv2.Laplacian(im.astype(np.float64), cv2.CV_64F)
        # Compare on the interior only: the JS kernel skips the border, which
        # is where OpenCV's reflected padding invents edges.
        py.append(float(lap[1:-1, 1:-1].var()))

    assert np.argsort(js).tolist() == np.argsort(py).tolist(), (js, py)
    for a, b in zip(js, py):
        assert a == pytest.approx(b, rel=0.25, abs=1.0), (js, py)


# --- movement -------------------------------------------------------------

def test_frame_motion_is_zero_for_an_identical_frame_and_grows_with_change():
    mod = metrics_module()
    result = run_node(f"""
        const {{ frameMotion }} = await import('{mod}');
        const a = Float32Array.from({{length: 100}}, (_, i) => i);
        const same = Float32Array.from(a);
        const shifted = Float32Array.from(a, v => v + 10);
        const far = Float32Array.from(a, v => v + 80);
        console.log(JSON.stringify({{
          identical: frameMotion(a, same),
          small: frameMotion(a, shifted),
          large: frameMotion(a, far),
          noPrev: frameMotion(null, a),
        }}));
    """)
    assert result["identical"] == 0
    assert 0 < result["small"] < result["large"]
    assert result["noPrev"] == 0


# --- the failure mode that actually matters -------------------------------

@pytest.mark.parametrize("rate,motion,dominated,why", [
    # Spinning on the spot: lots of gyro, little translation-induced motion.
    (90.0, 0.02, True, "pure rotation must be caught"),
    (40.0, 0.01, True, "slower spin still has no parallax"),
    # Walking around the subject: motion is large for the rotation present.
    (90.0, 0.20, False, "orbiting produces motion beyond the rotation"),
    (10.0, 0.05, False, "barely rotating"),
    # Still: not rotation-dominated, caught by the movement check instead.
    (2.0, 0.0005, False, "holding still"),
    # No gyroscope at all — must not guess.
    (None, 0.02, False, "no sensor means no verdict"),
])
def test_rotation_without_translation_is_detected(rate, motion, dominated, why):
    """Pure rotation yields no depth at all, and is the common user instinct.

    A camera alone cannot separate it from orbiting — both change the frame.
    The gyroscope can, which is why this is the one piece that depends on a
    sensor rather than on pixels.
    """
    mod = metrics_module()
    js_rate = "null" if rate is None else repr(rate)
    result = run_node(f"""
        const {{ isRotationDominated }} = await import('{mod}');
        console.log(JSON.stringify({{ d: isRotationDominated({js_rate}, {motion}) }}));
    """)
    assert result["d"] is dominated, why


# --- viewpoint coverage ---------------------------------------------------

def test_heading_buckets_wrap_and_cover_the_circle():
    mod = metrics_module()
    result = run_node(f"""
        const {{ headingBucket, coverageFraction, HEADING_BUCKETS }} = await import('{mod}');
        const orbit = new Uint8Array(HEADING_BUCKETS);
        for (let d = 0; d < 360; d += 5) orbit[headingBucket(d)] = 1;
        const wobble = new Uint8Array(HEADING_BUCKETS);
        for (let d = 80; d < 110; d += 2) wobble[headingBucket(d)] = 1;
        console.log(JSON.stringify({{
          orbit: coverageFraction(orbit),
          wobble: coverageFraction(wobble),
          wraps: [headingBucket(-10), headingBucket(350), headingBucket(370)],
          missing: headingBucket(null),
          empty: coverageFraction(new Uint8Array(HEADING_BUCKETS)),
        }}));
    """)
    assert result["orbit"] == 1.0, "a full circle must read as full coverage"
    assert result["wobble"] < 0.15, "a 30-degree wobble is not an orbit"
    # -10 degrees is the same heading as 350.
    assert result["wraps"][0] == result["wraps"][1]
    assert result["wraps"][2] == 1, "370 degrees wraps to 10"
    assert result["missing"] is None
    assert result["empty"] == 0.0


# --- coaching -------------------------------------------------------------

def advise(**state) -> dict:
    base = dict(recording=True, sharpness=200, motion=0.03, coverage=0.5,
                seconds=30, rotationDominated=False)
    base.update(state)
    mod = metrics_module()
    fields = ",".join(f"{k}: {str(v).lower() if isinstance(v, bool) else v}"
                      for k, v in base.items())
    return run_node(f"""
        const {{ captureAdvice }} = await import('{mod}');
        console.log(JSON.stringify(captureAdvice({{ {fields} }})));
    """)


def test_coaching_surfaces_the_worst_problem_first():
    """One message at a time — a HUD listing four complaints gets ignored."""
    # Blur outranks everything: an unusable frame is unusable regardless.
    both = advise(sharpness=5, rotationDominated=True, motion=0.0001)
    assert both["level"] == "bad" and "blur" in both["text"]

    spin = advise(rotationDominated=True)
    assert spin["level"] == "bad" and "on the spot" in spin["text"]

    fast = advise(motion=0.5)
    assert fast["level"] == "bad" and "too fast" in fast["text"]

    still = advise(motion=0.0)
    assert still["level"] == "warn" and "keep moving" in still["text"]


def test_coaching_counts_down_then_asks_for_coverage_then_approves():
    short = advise(seconds=4)
    assert "4s more" in short["text"] or "8s more" in short["text"]

    thin = advise(seconds=30, coverage=0.1)
    assert "circling" in thin["text"]

    good = advise(seconds=30, coverage=0.9)
    assert good["level"] == "good" and "close the loop" in good["text"]


def test_idle_preview_coaches_without_nagging():
    idle = advise(recording=False, sharpness=500)
    assert idle["level"] == "info"
    blurry_idle = advise(recording=False, sharpness=5)
    assert blurry_idle["level"] == "warn"


# --- post-capture verdict -------------------------------------------------

def verdict(**state) -> dict:
    base = dict(seconds=30, blurFraction=0.05, rotationFraction=0.05, coverage=0.8)
    base.update(state)
    mod = metrics_module()
    fields = ",".join(
        f"{k}: {'null' if v is None else v}" for k, v in base.items())
    return run_node(f"""
        const {{ captureVerdict }} = await import('{mod}');
        console.log(JSON.stringify(captureVerdict({{ {fields} }})));
    """)


def test_a_good_capture_is_accepted():
    assert verdict()["usable"] is True


@pytest.mark.parametrize("state,fragment", [
    (dict(seconds=4), "aim for"),
    (dict(blurFraction=0.8), "motion-blurred"),
    (dict(rotationFraction=0.9), "no parallax"),
    (dict(coverage=0.05), "too few distinct viewpoints"),
])
def test_unreconstructable_captures_are_named_not_silently_uploaded(state, fragment):
    v = verdict(**state)
    assert v["usable"] is False
    assert any(fragment in p for p in v["problems"]), v["problems"]


def test_missing_compass_does_not_fail_a_capture():
    """Plenty of devices have no magnetometer; that is not the user's fault."""
    v = verdict(coverage=None)
    assert v["usable"] is True
    assert not any("viewpoint" in p for p in v["problems"])


def test_multiple_problems_are_all_reported():
    v = verdict(seconds=3, blurFraction=0.9, rotationFraction=0.9)
    assert len(v["problems"]) == 3


# --- resumable upload client ----------------------------------------------

def test_upload_resumes_after_a_dropped_connection():
    """The client must continue from the server's offset, not restart."""
    mod = metrics_module()
    result = run_node(f"""
        const {{ uploadResumable }} = await import('{mod}');
        const size = 10_000, chunk = 2_000;
        let stored = 0, calls = [], failed = false;

        const api = async (path, init = {{}}) => {{
          calls.push(path.split('?')[0] + ' ' + (init.method || 'GET'));
          if (path.startsWith('/uploads/resumable'))
            return {{ upload_id: 'u1', offset: 0, chunk_size: chunk }};
          if (init.method === 'PATCH') {{
            // The network dies once, halfway through.
            if (stored === 4000 && !failed) {{ failed = true; throw new Error('network'); }}
            stored += init.body.size;
            return {{ offset: stored, total: size, complete: stored >= size }};
          }}
          return {{ offset: stored, total: size }};      // status probe
        }};

        const blob = {{ size, slice: (a, b) => ({{ size: b - a }}) }};
        const progress = [];
        const id = await uploadResumable(api, blob, 'clip.webm', {{
          onProgress: f => progress.push(f),
          sleep: async () => {{}},
        }});
        console.log(JSON.stringify({{
          id, stored, progress,
          statusProbes: calls.filter(c => c.endsWith('GET')).length,
          patches: calls.filter(c => c.includes('PATCH')).length,
        }}));
    """)
    assert result["id"] == "u1"
    assert result["stored"] == 10_000, "the assembled upload is the wrong size"
    assert result["statusProbes"] == 1, "should re-sync exactly once, after the failure"
    # 5 chunks plus the one that failed.
    assert result["patches"] == 6
    assert result["progress"] == sorted(result["progress"]), "progress went backwards"


def test_upload_resyncs_on_a_conflict_rather_than_looping():
    """A 409 means a chunk landed whose ack we lost — retrying it never ends."""
    mod = metrics_module()
    result = run_node(f"""
        const {{ uploadResumable }} = await import('{mod}');
        const size = 6_000, chunk = 2_000;
        let stored = 2_000, patches = 0, conflicted = false;
        const api = async (path, init = {{}}) => {{
          if (path.startsWith('/uploads/resumable'))
            return {{ upload_id: 'u2', offset: 0, chunk_size: chunk }};
          if (init.method === 'PATCH') {{
            patches++;
            // The first chunk is refused: the server already has those bytes.
            if (!conflicted) {{
              conflicted = true;
              const e = new Error('conflict'); e.status = 409; e.offset = stored;
              throw e;
            }}
            stored += init.body.size;
            return {{ offset: stored, total: size }};
          }}
          return {{ offset: stored, total: size }};
        }};
        const blob = {{ size, slice: (a, b) => ({{ size: b - a }}) }};
        await uploadResumable(api, blob, 'clip.webm', {{ sleep: async () => {{}} }});
        console.log(JSON.stringify({{ stored, patches }}));
    """)
    assert result["stored"] == 6_000
    # One refused, then two real chunks from the corrected offset.
    assert result["patches"] == 3


def test_upload_gives_up_after_repeated_failures():
    mod = metrics_module()
    result = run_node(f"""
        const {{ uploadResumable }} = await import('{mod}');
        let attempts = 0;
        const api = async (path, init = {{}}) => {{
          if (path.startsWith('/uploads/resumable'))
            return {{ upload_id: 'u3', offset: 0, chunk_size: 1000 }};
          if (init.method === 'PATCH') {{ attempts++; throw new Error('down'); }}
          return {{ offset: 0, total: 5000 }};
        }};
        const blob = {{ size: 5000, slice: (a, b) => ({{ size: b - a }}) }};
        let threw = null;
        try {{
          await uploadResumable(api, blob, 'c.webm', {{ retries: 3, sleep: async () => {{}} }});
        }} catch (e) {{ threw = e.message; }}
        console.log(JSON.stringify({{ threw, attempts }}));
    """)
    assert result["threw"] == "down"
    assert result["attempts"] == 4, "one attempt plus three retries"
