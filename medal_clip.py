import os
import re
import sys
import json
import errno
import shutil
import requests
import urllib.parse
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/115.0 Safari/537.36"

def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def _sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip()[:240]

def _find_video_url_from_meta(html:str, base_url: str) -> Optional[str]:
    m = re.search(r'<meta[^>]+property=["\']og:video:secure_url["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
    if not m:
        m = re.search(r'<meta[^>]+property=["\']og:video["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
    if m:
        url = m.group(1)
        return urllib.parse.urljoin(base_url, url)
    return None

def _find_video_url_from_json(html: str) -> Optional[str]:
    m = re.search(r'window\.__INITIAL_STATE__\s*=\s*({.+?})\s*;</script>', html, re.S)
    if not m:
        m = re.search(r'<script[^>]*>.*?(playerConfig|playback|clip)\s*[:=]\s*({.+?})\s*;?.*?</script>', html, re.S)
    if not m:
        return None
    try:
        payload = json.loads(m.group(1))
    except Exception:
        try:
            cleaned = m.group(1).replace("'", '"')
            payload = json.loads(cleaned)
        except Exception:
            return None

    def _walk(obj):
        if isinstance(obj, str):
            if obj.startswith("http") and obj.lower().endswith(('.mp4', '.mov', '.webm')):
                return obj
            return None
        elif isinstance(obj, dict):
            for k, v in obj.items():
                res = _walk(v)
                if res:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = _walk(item)
                if res:
                    return res
        return None
    
    found = _walk(payload)
    if found:
        return urllib.parse.urljoin("base_url", found)
    return None

def download_medal_clip(url: str, out_dir: str = "clips") -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
    }

    session = requests.Session()
    session.headers.update(headers)

    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    html = resp.text
    base_url = resp.url

    video_url = _find_video_url_from_meta(html, base_url)
    if not video_url:
        video_url = _find_video_url_from_json(html)

    if not video_url:
        raise RuntimeError("Could not find video URL in the provided Medal clip page.")
    
    # Prefer the clip title from the page; fall back to basename of the video URL
    parsed = urllib.parse.urlparse(video_url)
    ext = os.path.splitext(parsed.path)[1] or ".mp4"

    fname = os.path.basename(parsed.path) or "medal_clip" + ext
    fname = _sanitize_filename(fname)
    if not os.path.splitext(fname)[1]:
        fname += ".mp4"

    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, fname)

    with session.get(video_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get('Content-Length')
        if total is None:
            with open(out_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            total = int(total)
            chunk_size = 8192
            downloaded = 0
            with open(out_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        done = int(50 * downloaded / total)
                        print("\rDownloading: [{}{}] {}/{} bytes".format("#" * done, " " * (50 - done), downloaded, total), end="")
            print()

    return out_path

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        input_url = sys.argv[1]
    else:
        input_url = input("Medal clip URL: ").strip()

    try:
        saved = download_medal_clip(input_url)
        print(f"Saved clip to: {saved}")
    except Exception as e:
        print("Error:", e)