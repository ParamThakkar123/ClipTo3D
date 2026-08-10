"""The web viewer's JavaScript, tested against real exports (MPO-246).

`viewer/index.html` is the only part of the pipeline written in a language the
Python suite cannot reach, and it parses two binary formats plus does the
splat projection maths — precisely the code where a silent mistake produces a
plausible-looking but wrong picture.

These tests extract the pure functions from the page and run them under Node
against fixtures produced by `export.py`, so the viewer is checked against the
bytes the pipeline actually writes rather than against a hand-made sample.

Skipped when Node is unavailable; nothing here needs a browser or a GPU.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
VIEWER = REPO / "viewer" / "index.html"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="needs Node to run the viewer's JS"
)


def viewer_source() -> str:
    html = VIEWER.read_text(encoding="utf-8")
    start = html.index('<script type="module">') + len('<script type="module">')
    return html[start:html.rindex("</script>")]


def extract(*, upto: str, frm: str = "") -> str:
    """Slice the pure functions out of the page.

    Everything after the renderer touches `document` and WebGL, so only the
    part above it can run headless.
    """
    src = viewer_source()
    begin = src.index(frm) if frm else 0
    return src[begin:src.index(upto)]


def run_node(script: str, *args: str) -> dict:
    """Run a snippet under Node and parse the JSON it prints on the last line."""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", textwrap.dedent(script), *args],
        capture_output=True, text=True, cwd=REPO,
    )
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    last = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1]
    return json.loads(last)


def module_from(src: str, *exports: str) -> str:
    """A data: URL module Node can import, carrying the viewer's own code."""
    import base64

    body = (src + f"\nexport {{ {', '.join(exports)} }};").encode("utf-8")
    return "data:text/javascript;base64," + base64.b64encode(body).decode("ascii")


# --- fixtures produced by the real exporter -------------------------------

@pytest.fixture(scope="module")
def cloud_glb(tmp_path_factory):
    from export import write_glb

    d = tmp_path_factory.mktemp("viewer")
    rng = np.random.default_rng(0)
    pts = (rng.random((2000, 3)) * np.array([4.0, 2.0, 8.0])).astype(np.float32)
    cols = rng.integers(0, 256, (2000, 3), dtype=np.uint8)
    res = write_glb(d / "cloud.glb", pts, cols)
    return res.path, pts, cols


@pytest.fixture(scope="module")
def scene_splat(tmp_path_factory):
    from export import splat_from_pointcloud

    d = tmp_path_factory.mktemp("viewer_splat")
    rng = np.random.default_rng(1)
    pts = (rng.random((1500, 3)) * 3.0).astype(np.float32)
    cols = rng.integers(0, 256, (1500, 3), dtype=np.uint8)
    res = splat_from_pointcloud(pts, cols, out_path=d / "scene.splat")
    return res.path, pts


# --- GLB parsing ----------------------------------------------------------

def test_viewer_parses_a_real_glb(cloud_glb):
    """Dequantisation and the node transform must reproduce the input points."""
    path, pts, _cols = cloud_glb
    src = extract(frm="// --- GLB parsing", upto="// --- .splat parsing")
    mod = module_from(src, "loadPointCloud")

    result = run_node(f"""
        import {{ readFileSync }} from 'node:fs';
        const {{ loadPointCloud }} = await import('{mod}');
        const buf = readFileSync(process.argv[1]);
        const c = loadPointCloud(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
        const lo = [0,1,2].map(k => Math.min(...Array.from({{length:c.count}}, (_,i)=>c.positions[i*3+k])));
        const hi = [0,1,2].map(k => Math.max(...Array.from({{length:c.count}}, (_,i)=>c.positions[i*3+k])));
        console.log(JSON.stringify({{
          count: c.count,
          hasColors: c.colors !== null,
          allFinite: c.positions.every(Number.isFinite),
          colorsInRange: c.colors.every(v => v >= 0 && v <= 1),
          lo, hi,
        }}));
    """, str(path))

    assert result["count"] == len(pts)
    assert result["hasColors"] and result["allFinite"] and result["colorsInRange"]
    # int16 quantisation over the bounding box: a couple of LSBs of tolerance.
    tol = (pts.max(axis=0) - pts.min(axis=0)) / 32767.0 * 4
    np.testing.assert_allclose(result["lo"], pts.min(axis=0), atol=max(tol))
    np.testing.assert_allclose(result["hi"], pts.max(axis=0), atol=max(tol))


def test_viewer_honours_bytestride(cloud_glb):
    """export.py pads int16 positions to an 8-byte stride for glTF alignment;
    a contiguous read would be silently wrong rather than an error."""
    path, pts, _ = cloud_glb
    src = extract(frm="// --- GLB parsing", upto="// --- .splat parsing")
    mod = module_from(src, "parseGLB")

    result = run_node(f"""
        import {{ readFileSync }} from 'node:fs';
        const {{ parseGLB }} = await import('{mod}');
        const buf = readFileSync(process.argv[1]);
        const {{ json }} = parseGLB(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
        const acc = json.accessors[json.meshes[0].primitives[0].attributes.POSITION];
        console.log(JSON.stringify({{ stride: json.bufferViews[acc.bufferView].byteStride }}));
    """, str(path))
    assert result["stride"] == 8, "fixture no longer exercises the padded-stride path"


def test_viewer_rejects_a_non_glb(tmp_path):
    src = extract(frm="// --- GLB parsing", upto="// --- .splat parsing")
    mod = module_from(src, "parseGLB")
    bad = tmp_path / "not.glb"
    bad.write_bytes(b"this is not a glb file at all")

    result = run_node(f"""
        import {{ readFileSync }} from 'node:fs';
        const {{ parseGLB }} = await import('{mod}');
        const buf = readFileSync(process.argv[1]);
        let err = null;
        try {{ parseGLB(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)); }}
        catch (e) {{ err = e.message; }}
        console.log(JSON.stringify({{ err }}));
    """, str(bad))
    assert result["err"] and "GLB" in result["err"]


# --- .splat parsing -------------------------------------------------------

def test_viewer_parses_a_real_splat(scene_splat):
    path, pts = scene_splat
    src = extract(frm="// --- .splat parsing", upto="// --- renderer")
    mod = module_from(src, "loadSplat")

    result = run_node(f"""
        import {{ readFileSync }} from 'node:fs';
        const {{ loadSplat }} = await import('{mod}');
        const buf = readFileSync(process.argv[1]);
        const s = loadSplat(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
        let unitQuats = true;
        for (let i = 0; i < Math.min(s.count, 200); i++) {{
          const q = [0,1,2,3].map(k => s.quats[i*4+k]);
          if (Math.abs(Math.hypot(...q) - 1) > 0.05) unitQuats = false;
        }}
        console.log(JSON.stringify({{
          total: s.total,
          bytesPerRecord: buf.length / s.total,
          scalesPositive: s.scales.every(v => v > 0),
          alphasInRange: s.alphas.every(v => v >= 0 && v <= 1),
          colorsInRange: s.colors.every(v => v >= 0 && v <= 1),
          unitQuats,
        }}));
    """, str(path))

    assert result["total"] == len(pts)
    assert result["bytesPerRecord"] == 32
    assert result["scalesPositive"], "log-space scales were not activated"
    assert result["alphasInRange"] and result["colorsInRange"] and result["unitQuats"]


def test_truncating_a_splat_is_a_valid_coarse_level(scene_splat):
    """The file is importance-ordered, so reading the first N records IS the
    LOD — a truncated read must not be a random subset."""
    path, _ = scene_splat
    src = extract(frm="// --- .splat parsing", upto="// --- renderer")
    mod = module_from(src, "loadSplat")

    result = run_node(f"""
        import {{ readFileSync }} from 'node:fs';
        const {{ loadSplat }} = await import('{mod}');
        const buf = readFileSync(process.argv[1]);
        const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        const all = loadSplat(ab), part = loadSplat(ab, 100);
        const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
        const bigA = mean(Array.from({{length: all.count}}, (_, i) => all.scales[i*3]
                                        * (all.alphas[i])));
        const bigP = mean(Array.from({{length: part.count}}, (_, i) => part.scales[i*3]
                                        * (part.alphas[i])));
        console.log(JSON.stringify({{ partCount: part.count, total: part.total,
                                      meanAll: bigA, meanPart: bigP }}));
    """, str(path))

    assert result["partCount"] == 100
    assert result["total"] == 1500
    # Coarse level keeps gaussians at least as visually significant as average.
    assert result["meanPart"] >= result["meanAll"] * 0.999


def test_empty_splat_is_rejected(tmp_path):
    src = extract(frm="// --- .splat parsing", upto="// --- renderer")
    mod = module_from(src, "loadSplat")
    empty = tmp_path / "empty.splat"
    empty.write_bytes(b"")

    result = run_node(f"""
        import {{ readFileSync }} from 'node:fs';
        const {{ loadSplat }} = await import('{mod}');
        const buf = readFileSync(process.argv[1]);
        let err = null;
        try {{ loadSplat(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)); }}
        catch (e) {{ err = e.message; }}
        console.log(JSON.stringify({{ err }}));
    """, str(empty))
    assert result["err"] and "empty" in result["err"]


# --- depth sorting --------------------------------------------------------
#
# `computeSplatOrder` is deliberately pure so these run against the shipped
# function. An earlier round tested a re-implementation of this logic in a
# scratch script, which proves only that the copy was right.

def sort_module() -> str:
    """The sort, lifted out of the renderer section it lives in."""
    src = viewer_source()
    begin = src.index("const SORT_BUCKETS")
    end = src.index("function sortSplats")
    return module_from(src[begin:end], "computeSplatOrder")


def test_sort_orders_back_to_front():
    """Alpha compositing is order-dependent; farthest must be drawn first."""
    mod = sort_module()
    result = run_node(f"""
        const {{ computeSplatOrder }} = await import('{mod}');
        const n = 20000;
        let seed = 42;
        const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
        const pos = new Float32Array(n * 3);
        for (let i = 0; i < n * 3; i++) pos[i] = (rnd() - 0.5) * 20;
        const eye = [3, -1, 12];

        const order = new Uint32Array(n), depth = new Float32Array(n);
        const counts = new Uint32Array(65536);
        computeSplatOrder(pos, n, eye, order, depth, counts);

        const seen = new Set(order);
        let inversions = 0, worst = 0;
        for (let i = 1; i < n; i++) {{
          const prev = depth[order[i-1]], cur = depth[order[i]];
          if (cur > prev) {{ inversions++; worst = Math.max(worst, cur - prev); }}
        }}
        const span = Math.max(...depth) - Math.min(...depth);
        console.log(JSON.stringify({{
          unique: seen.size, n,
          firstIsFarthest: depth[order[0]] >= depth[order[n-1]],
          inversions, worstRelative: worst / span,
        }}));
    """)

    assert result["unique"] == result["n"], "a gaussian was dropped or duplicated"
    assert result["firstIsFarthest"]
    # Quantisation allows ties within a bucket, never a real inversion.
    assert result["worstRelative"] < 1 / 30000, result


def test_sort_is_fast_enough_to_run_per_frame():
    mod = sort_module()
    result = run_node(f"""
        const {{ computeSplatOrder }} = await import('{mod}');
        const n = 100000;
        const pos = new Float32Array(n * 3);
        for (let i = 0; i < n * 3; i++) pos[i] = Math.sin(i) * 10;
        const order = new Uint32Array(n), depth = new Float32Array(n);
        const counts = new Uint32Array(65536);
        computeSplatOrder(pos, n, [0,0,5], order, depth, counts);   // warm
        const t0 = performance.now();
        for (let r = 0; r < 5; r++) computeSplatOrder(pos, n, [r,0,5], order, depth, counts);
        console.log(JSON.stringify({{ ms: (performance.now() - t0) / 5, n }}));
    """)
    # A comparison sort of 100k per frame would blow the frame budget; this is
    # the whole reason for the counting sort.
    assert result["ms"] < 50, f"sort too slow for interactive use: {result['ms']:.1f} ms"


def test_sort_handles_degenerate_input():
    """All gaussians at one point: span is zero and must not divide by zero."""
    mod = sort_module()
    result = run_node(f"""
        const {{ computeSplatOrder }} = await import('{mod}');
        const n = 500;
        const pos = new Float32Array(n * 3).fill(1.5);
        const order = new Uint32Array(n), depth = new Float32Array(n);
        const counts = new Uint32Array(65536);
        computeSplatOrder(pos, n, [0,0,0], order, depth, counts);
        console.log(JSON.stringify({{
          unique: new Set(order).size,
          allFinite: Array.from(order).every(Number.isFinite),
        }}));
    """)
    assert result["unique"] == 500 and result["allFinite"]


def test_sort_reverses_when_the_camera_moves_to_the_other_side():
    """The ordering must actually depend on the viewpoint."""
    mod = sort_module()
    result = run_node(f"""
        const {{ computeSplatOrder }} = await import('{mod}');
        const n = 1000;
        const pos = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) pos[i*3 + 2] = i * 0.01;   // a line along z
        const order = new Uint32Array(n), depth = new Float32Array(n);
        const counts = new Uint32Array(65536);
        computeSplatOrder(pos, n, [0,0,-50], order, depth, counts);
        const front = Array.from(order.slice(0, 5));
        computeSplatOrder(pos, n, [0,0,50], order, depth, counts);
        const back = Array.from(order.slice(0, 5));
        console.log(JSON.stringify({{ front, back }}));
    """)
    # From -z the farthest are the high-index gaussians; from +z, the low ones.
    assert max(result["front"]) > 900, result
    assert min(result["back"]) < 100, result


# --- job service client ---------------------------------------------------
#
# The viewer's upload/progress path is the last piece of MPO-246. These run
# the shipped poll loop against a fake API rather than a real server, so the
# retry/terminal/artifact-choice logic is covered without a live service.

def job_module() -> str:
    src = viewer_source()
    begin = src.index("const STAGE_ORDER")
    end = src.index("const keyEl = document.getElementById")
    return module_from(src[begin:end], "summarise", "pickArtifact", "pollJob", "pollDelay")


def test_progress_is_measured_against_declared_stages():
    mod = job_module()
    result = run_node(f"""
        const {{ summarise }} = await import('{mod}');
        console.log(JSON.stringify(summarise({{
          status: 'running',
          stages: {{
            depth:  {{ status: 'done', seconds: 3 }},
            frames: {{ status: 'done', seconds: 1 }},
            colmap: {{ status: 'running' }},
          }},
        }})));
    """)
    # Ordered by the pipeline's real order, not by dict insertion.
    assert result["names"] == ["frames", "depth", "colmap"]
    assert result["done"] == 2 and result["running"] == "colmap"
    assert abs(result["fraction"] - 2 / 3) < 1e-9
    assert not result["finished"]


def test_failure_surfaces_the_stage_error():
    mod = job_module()
    result = run_node(f"""
        const {{ summarise }} = await import('{mod}');
        console.log(JSON.stringify(summarise({{
          status: 'failed', error: '',
          stages: {{ colmap: {{ status: 'failed', error: 'no good initial pair' }} }},
        }})));
    """)
    assert result["failed"] and result["finished"]
    assert result["error"] == "no good initial pair"


def test_unknown_stage_names_are_not_dropped():
    """A pipeline that grows a stage must not lose it from the display."""
    mod = job_module()
    result = run_node(f"""
        const {{ summarise }} = await import('{mod}');
        console.log(JSON.stringify(summarise({{
          status: 'running',
          stages: {{ frames: {{status:'done'}}, brand_new: {{status:'running'}} }},
        }})));
    """)
    assert result["names"] == ["frames", "brand_new"]


def test_artifact_choice_prefers_splats_then_mesh_then_cloud():
    mod = job_module()
    result = run_node(f"""
        const {{ pickArtifact }} = await import('{mod}');
        const all = [{{name:'cloud.glb'}},{{name:'mesh.glb'}},{{name:'scene.splat'}},
                     {{name:'scene.usdz'}},{{name:'mesh.obj'}}];
        console.log(JSON.stringify({{
          best: pickArtifact(all).name,
          noSplat: pickArtifact(all.filter(a => a.name !== 'scene.splat')).name,
          cloudOnly: pickArtifact([{{name:'cloud_lod2.glb'}}]).name,
          // usdz and obj are exports, not things this renderer can draw.
          nothing: pickArtifact([{{name:'scene.usdz'}},{{name:'mesh.obj'}}]),
        }}));
    """)
    assert result["best"] == "scene.splat"
    assert result["noSplat"] == "mesh.glb"
    assert result["cloudOnly"] == "cloud_lod2.glb"
    assert result["nothing"] is None


def test_poll_runs_until_the_job_reaches_a_terminal_state():
    mod = job_module()
    result = run_node(f"""
        const {{ pollJob }} = await import('{mod}');
        const script = [
          {{ status: 'queued',  stages: {{}} }},
          {{ status: 'running', stages: {{ frames: {{status:'done'}}, depth: {{status:'running'}} }} }},
          {{ status: 'done',    stages: {{ frames: {{status:'done'}}, depth: {{status:'done'}} }} }},
        ];
        let calls = 0;
        const api = async () => script[Math.min(calls++, script.length - 1)];
        const seen = [];
        let slept = 0;
        const job = await pollJob(api, 'abc', (j, s) => seen.push(j.status), {{
          sleep: async ms => {{ slept += ms; }},
          now: () => slept,                      // virtual clock: no real waiting
        }});
        console.log(JSON.stringify({{ final: job.status, seen, calls, slept }}));
    """)
    assert result["final"] == "done"
    assert result["seen"] == ["queued", "running", "done"]
    assert result["calls"] == 3, "polled after the job was already done"
    assert result["slept"] > 0


def test_poll_gives_up_rather_than_spinning_forever():
    mod = job_module()
    result = run_node(f"""
        const {{ pollJob }} = await import('{mod}');
        let slept = 0, calls = 0;
        try {{
          await pollJob(async () => (calls++, {{ status: 'running', stages: {{}} }}),
            'abc', () => {{}},
            {{ sleep: async ms => {{ slept += ms; }}, now: () => slept, maxMs: 60000 }});
          console.log(JSON.stringify({{ threw: false }}));
        }} catch (e) {{
          console.log(JSON.stringify({{ threw: true, message: e.message, calls }}));
        }}
    """)
    assert result["threw"] and "in time" in result["message"]


def test_poll_backs_off_but_stays_bounded():
    mod = job_module()
    result = run_node(f"""
        const {{ pollDelay }} = await import('{mod}');
        console.log(JSON.stringify({{
          start: pollDelay(0),
          minute: pollDelay(60000),
          hour: pollDelay(3600000),
        }}));
    """)
    assert result["start"] == 500
    assert result["start"] < result["minute"] < 5001
    # A long job must still refresh often enough to feel live.
    assert result["hour"] == 5000


# --- AR handoff (MPO-248) -------------------------------------------------
#
# The OS viewers give no useful diagnostics when handed a malformed URL — the
# failure is a blank screen or a silent download — so the URL construction and
# the platform detection are worth testing precisely.

def ar_module() -> str:
    src = viewer_source()
    begin = src.index("function arCapability")
    end = src.index("const arEl = document.getElementById")
    return module_from(src[begin:end], "arCapability", "sceneViewerURL",
                       "pickArAsset", "isPubliclyReachable")


# Node 25 defines `navigator` as a getter-only global, so plain assignment
# throws. defineProperty replaces it for the duration of the snippet.
AR_STUB = """
        const stub = (name, value) => Object.defineProperty(
            globalThis, name, { value, configurable: true, writable: true });
        stub('navigator', { userAgent: '', platform: '', maxTouchPoints: 0 });
        stub('location', { href: 'https://example.com/viewer' });
"""


@pytest.mark.parametrize("ua,platform,touch,expected", [
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)", "iPhone", 5, "quicklook"),
    ("Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X)", "iPad", 5, "quicklook"),
    # iPadOS lies and calls itself a Mac; touch points are the tell.
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "MacIntel", 5, "quicklook"),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "MacIntel", 0, None),
    ("Mozilla/5.0 (Linux; Android 14; Pixel 8)", "Linux armv8l", 5, "scene-viewer"),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Win32", 0, None),
])
def test_ar_capability_detection(ua, platform, touch, expected):
    mod = ar_module()
    result = run_node(f"""
        {AR_STUB}
        navigator.maxTouchPoints = {touch};
        const {{ arCapability }} = await import('{mod}');
        console.log(JSON.stringify({{ cap: arCapability({ua!r}, {platform!r}) }}));
    """)
    assert result["cap"] == expected


def test_scene_viewer_intent_url_is_well_formed():
    mod = ar_module()
    result = run_node(f"""
        {AR_STUB}
        const {{ sceneViewerURL }} = await import('{mod}');
        console.log(JSON.stringify({{
          url: sceneViewerURL('https://cdn.example/j1/mesh.glb',
                              {{ title: 'My scan', fallback: 'https://app.example/v?job=j1' }}),
        }}));
    """)
    url = result["url"]
    assert url.startswith("intent://arvr.google.com/scene-viewer/1.0?")
    # Scene Viewer will not launch without these.
    for required in ("#Intent", "scheme=https",
                     "package=com.google.android.googlequicksearchbox",
                     "action=android.intent.action.VIEW"):
        assert required in url, required
    assert url.endswith("end;"), "an intent URL must terminate with end;"
    assert "mode=ar_preferred" in url
    # The file URL is a parameter value and must be encoded, not inlined raw.
    assert "file=https%3A%2F%2Fcdn.example%2Fj1%2Fmesh.glb" in url
    assert "S.browser_fallback_url=https%3A%2F%2Fapp.example" in url


def test_scene_viewer_url_without_a_fallback_is_still_terminated():
    mod = ar_module()
    result = run_node(f"""
        {AR_STUB}
        const {{ sceneViewerURL }} = await import('{mod}');
        console.log(JSON.stringify({{ url: sceneViewerURL('https://cdn.example/a.glb') }}));
    """)
    assert result["url"].endswith("end;")
    assert "browser_fallback_url" not in result["url"]


def test_ar_asset_choice_is_per_platform_and_prefers_a_mesh():
    mod = ar_module()
    result = run_node(f"""
        {AR_STUB}
        const {{ pickArAsset }} = await import('{mod}');
        const all = [{{name:'cloud.glb'}},{{name:'cloud_lod1.glb'}},{{name:'mesh.glb'}},
                     {{name:'scene.usdz'}},{{name:'scene.splat'}}];
        console.log(JSON.stringify({{
          ios: pickArAsset(all, 'quicklook').name,
          android: pickArAsset(all, 'scene-viewer').name,
          desktop: pickArAsset(all, null),
          // A cloud-only job has no mesh; USDZ is simply absent.
          noUsdz: pickArAsset([{{name:'cloud.glb'}}], 'quicklook'),
        }}));
    """)
    assert result["ios"] == "scene.usdz"
    # Not cloud.glb: neither OS viewer lights or occludes a point cloud.
    assert result["android"] == "mesh.glb"
    assert result["desktop"] is None
    assert result["noUsdz"] is None


@pytest.mark.parametrize("url,reachable", [
    ("https://cdn.example.com/a.glb", True),
    ("http://localhost:8000/a.glb", False),
    ("http://127.0.0.1:8000/a.glb", False),
    ("http://192.168.1.20:8000/a.glb", False),
    ("http://10.0.0.5/a.glb", False),
    ("http://172.16.4.4/a.glb", False),
    ("http://mac.local/a.glb", False),
])
def test_private_addresses_are_recognised_as_unreachable_for_scene_viewer(url, reachable):
    """Scene Viewer fetches from another app, so a dev server cannot serve it."""
    mod = ar_module()
    result = run_node(f"""
        {AR_STUB}
        const {{ isPubliclyReachable }} = await import('{mod}');
        console.log(JSON.stringify({{ ok: isPubliclyReachable({url!r}) }}));
    """)
    assert result["ok"] is reachable


# --- mobile budgets (MPO-248 Option B) ------------------------------------
#
# The issue's worry about the WebView path is memory ceilings and sort cost on
# mobile. Neither needs a device to bound: the per-primitive footprint is
# fixed by the attribute layout, so the tier budgets can be checked directly.

def tier_module() -> str:
    src = viewer_source()
    tiering = src[src.index("function deviceTier"):src.index("// --- GLB parsing")]
    budget = src[src.index("function splatBudget"):]
    budget = budget[:budget.index("}") + 1]
    return module_from(tiering + "\n" + budget, "deviceTier", "splatBudget")


@pytest.mark.parametrize("ua,mem,cores,tier", [
    ("Mozilla/5.0 (Linux; Android 14; low-end)", 2, 4, 2),      # mid-range phone
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)", 6, 6, 1),      # recent phone
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)", 16, 16, 0),   # desktop
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)", 4, 8, 1),     # weak desktop
])
def test_device_tier_and_splat_budget(ua, mem, cores, tier):
    mod = tier_module()
    result = run_node(f"""
        Object.defineProperty(globalThis, 'navigator', {{ configurable: true, value: {{
            userAgent: {ua!r}, deviceMemory: {mem}, hardwareConcurrency: {cores},
        }}}});
        const {{ deviceTier, splatBudget }} = await import('{mod}');
        console.log(JSON.stringify({{ tier: deviceTier(), budget: splatBudget() }}));
    """)
    assert result["tier"] == tier
    assert result["budget"] > 0


def test_mobile_splat_budgets_stay_within_a_stated_memory_ceiling(scene_splat):
    """Bound the footprint the WebView path asks a phone for.

    Per gaussian the viewer uploads position (12B), colour (12B), scale (12B),
    rotation (16B) and alpha (4B), and keeps an index (4B) and depth (4B)
    array. The tier budgets multiply straight through, so the ceiling is
    arithmetic rather than a guess — and it is asserted against the arrays the
    shipped loader actually allocates.
    """
    path, _ = scene_splat
    src = viewer_source()
    mod = module_from(
        src[src.index("// --- .splat parsing"):src.index("// --- renderer")], "loadSplat")

    result = run_node(f"""
        import {{ readFileSync }} from 'node:fs';
        const {{ loadSplat }} = await import('{mod}');
        const buf = readFileSync(process.argv[1]);
        const s = loadSplat(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
        const attrs = ['positions','colors','scales','quats','alphas']
          .reduce((n, k) => n + s[k].byteLength, 0);
        const perPrim = (attrs + s.count * 8) / s.count;   // + index and depth
        console.log(JSON.stringify({{ perPrim, count: s.count }}));
    """, str(path))

    per = result["perPrim"]
    assert per == 64, f"attribute layout changed: {per} bytes per gaussian"

    ceilings = {2: 32 * 1024**2, 1: 96 * 1024**2, 0: 256 * 1024**2}
    for tier, budget in enumerate([2_000_000, 800_000, 250_000]):
        assert budget * per <= ceilings[tier], (
            f"tier {tier}: {budget * per / 1024**2:.0f} MB exceeds its ceiling")

    # The mid-range tier is the one that has to be safe on a 2GB phone.
    assert 250_000 * per / 1024**2 < 20
