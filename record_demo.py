"""
Record a demo video of the FedKG-Brain platform: real inputs, real computation, live captions.

  python record_demo.py --url http://127.0.0.1:7860 --out FedKG_Brain_demo.mp4 --upload /path/to/subject_rois_cc200.1D

The platform (app.py) must already be running. The captions at the bottom of the screen come from the platform
itself (they describe each real processing step); this script only adds a title card, chapter titles at the top,
a visible cursor and the mouse movements.
Needs: playwright (+ chromium) and ffmpeg (system or imageio-ffmpeg).
"""
import argparse, os, shutil, subprocess, tempfile, time
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:7860/")
ap.add_argument("--out", default="FedKG_Brain_demo.mp4")
ap.add_argument("--upload", required=True, help=".1D scan of a subject never used for training (the INPUT)")
ap.add_argument("--author", default="Wiem Ben El Haj Salah Bouhdid · ENSI, University of Manouba · 2026")
ap.add_argument("--watermark", default="", help="red banner text (only for test previews)")
ap.add_argument("--rounds", type=int, default=15, help="federated rounds shown in the video")
ap.add_argument("--width", type=int, default=1600)
ap.add_argument("--height", type=int, default=900)
ap.add_argument("--timeout", type=int, default=900, help="max seconds to wait for one computation")
A = ap.parse_args()

OVERLAY = r"""
() => {
  if (window.__rec) return; window.__rec = true;
  const st = document.createElement('style');
  st.textContent = `
   #rec-chap{position:fixed;top:18px;left:50%;transform:translateX(-50%) translateY(-20px);z-index:999999;opacity:0;
     transition:all .5s;background:rgba(14,10,30,.92);border:1px solid rgba(167,139,250,.45);border-radius:999px;
     padding:10px 28px;color:#fff;font:600 22px 'Space Grotesk',Inter,Arial,sans-serif;letter-spacing:.5px;
     box-shadow:0 10px 30px rgba(0,0,0,.45)}
   #rec-chap.on{opacity:1;transform:translateX(-50%) translateY(0)}
   #rec-chap b{background:linear-gradient(90deg,#A78BFA,#2DD4BF);-webkit-background-clip:text;background-clip:text;color:transparent;margin-right:10px}
   #rec-cursor{position:fixed;width:22px;height:22px;border-radius:50%;background:rgba(167,139,250,.35);border:2px solid #E9D5FF;
     z-index:1000000;pointer-events:none;transform:translate(-50%,-50%);transition:width .15s,height .15s;left:-50px;top:-50px;
     box-shadow:0 0 14px rgba(167,139,250,.8)}
   #rec-cursor.down{width:36px;height:36px;background:rgba(45,212,191,.45);border-color:#5EEAD4}
   #rec-title{position:fixed;inset:0;z-index:999998;display:flex;flex-direction:column;align-items:center;justify-content:center;
     background:radial-gradient(900px 500px at 20% 10%,rgba(124,58,237,.55),transparent 60%),
       radial-gradient(800px 500px at 90% 90%,rgba(20,184,166,.40),transparent 60%),#07060F;color:#fff;transition:opacity .9s}
   #rec-title h1{font:700 92px 'Space Grotesk',Inter,Arial,sans-serif;margin:0;background:linear-gradient(90deg,#E9D5FF,#A78BFA,#2DD4BF);
     -webkit-background-clip:text;background-clip:text;color:transparent}
   #rec-title p{font:400 26px Inter,Arial,sans-serif;margin:16px 0 0;color:#CFC8E8;text-align:center;max-width:72%}
   #rec-wm{position:fixed;top:16px;right:16px;pointer-events:none;z-index:1000001;background:#C0392B;color:#fff;
     font:700 18px Arial;padding:8px 14px;border-radius:8px;opacity:.92}
  `;
  document.head.appendChild(st);
  const c = document.createElement('div'); c.id = 'rec-cursor'; document.body.appendChild(c);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  document.addEventListener('mousedown', () => c.classList.add('down'), true);
  document.addEventListener('mouseup', () => setTimeout(() => c.classList.remove('down'), 150), true);
  window.recChap = (h) => { let el = document.getElementById('rec-chap');
    if (!el) { el = document.createElement('div'); el.id = 'rec-chap'; document.body.appendChild(el); }
    if (!h) { el.classList.remove('on'); return; } el.innerHTML = h; el.classList.add('on'); };
  window.recTitle = (t, s) => { let el = document.getElementById('rec-title');
    if (!t) { if (el) { el.style.opacity = 0; setTimeout(() => el.remove(), 1000); } return; }
    el = document.createElement('div'); el.id = 'rec-title'; el.innerHTML = '<h1>' + t + '</h1><p>' + s + '</p>'; document.body.appendChild(el); };
  window.recWM = (t) => { const w = document.createElement('div'); w.id = 'rec-wm'; w.textContent = t; document.body.appendChild(w); };
}
"""


def run():
    tmp = tempfile.mkdtemp()
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": A.width, "height": A.height}, record_video_dir=tmp,
                            record_video_size={"width": A.width, "height": A.height})
        page = ctx.new_page()
        page.goto(A.url); page.wait_for_load_state("networkidle"); time.sleep(2)
        page.evaluate(OVERLAY)
        if A.watermark:
            page.evaluate("t => window.recWM(t)", A.watermark)

        def chap(num, text):
            page.evaluate("h => window.recChap(h)", f"<b>{num}</b>{text}" if num else text)

        def nochap():
            page.evaluate("() => window.recChap('')")

        def wait_done(t=A.timeout):
            """Wait for the platform's own 'computation finished' marker."""
            t0 = time.time()
            while time.time() - t0 < 6 and page.locator(".run-done").count() > 0:   # old marker from the previous run
                time.sleep(0.2)
            while time.time() - t0 < t:
                if page.locator(".run-done").count() > 0:
                    return True
                time.sleep(0.4)
            print("timeout while waiting for a result")
            return False

        def move_to(loc, click=True, dx=0.5, dy=0.5):
            loc.scroll_into_view_if_needed(timeout=15000); time.sleep(0.5)
            bb = loc.bounding_box()
            if bb is None:
                if click:
                    loc.click()
                return
            page.mouse.move(bb["x"] + bb["width"] * dx, bb["y"] + bb["height"] * dy, steps=28); time.sleep(0.35)
            if click:
                page.mouse.down(); time.sleep(0.08); page.mouse.up()

        def glide(sel, offset=-90, pause=3.0):
            """Smoothly scroll so that element `sel` sits near the top of the screen."""
            page.evaluate("""([s, o]) => { const e = document.querySelector(s); if (!e) return;
                window.scrollTo({top: e.getBoundingClientRect().top + window.scrollY + o, behavior: 'smooth'}); }""", [sel, offset])
            time.sleep(pause)

        def top(pause=1.4):
            page.evaluate("() => window.scrollTo({top:0, behavior:'smooth'})"); time.sleep(pause)

        def tab(name):
            top(1.2); page.evaluate("() => window.scrollTo(0, 0)"); time.sleep(0.2)
            t = page.get_by_role("tab", name=name)
            move_to(t); time.sleep(1.2)
            if t.get_attribute("aria-selected") != "true":
                t.click(); time.sleep(1.0)

        def rotate_brain():
            el = page.locator("#an-brain")
            el.scroll_into_view_if_needed(); time.sleep(0.8)
            bb = el.bounding_box()
            if not bb:
                return
            x, y = bb["x"] + bb["width"] * 0.42, bb["y"] + bb["height"] * 0.5
            page.mouse.move(x, y, steps=20); page.mouse.down()
            page.mouse.move(x + 300, y + 20, steps=24); time.sleep(0.4); page.mouse.move(x + 140, y - 25, steps=14)
            page.mouse.up(); time.sleep(1.5)

        def scene(fn):
            try:
                fn()
            except Exception as e:  # keep recording even if one step fails
                print("scene skipped:", fn.__name__, type(e).__name__, str(e)[:200])
                page.evaluate("() => window.scrollTo(0, 0)")

        def s0_intro():
            page.evaluate("([t, s]) => window.recTitle(t, s)", ["FedKG-Brain",
                          "Brain-connectivity graphs · graph neural networks · federated learning<br>Live demo on ABIDE I"])
            time.sleep(4.5); page.evaluate("() => window.recTitle('')"); time.sleep(1.5)
            chap("", "Input: a resting-state fMRI scan  →  Output: probability of autism + why")
            time.sleep(4)
            cb = page.locator("#capt input[type=checkbox]")
            if cb.count() and not cb.is_checked():
                cb.check()

        def s1_upload():
            chap("01", "Analyze a patient’s scan")
            move_to(page.locator("#an-file"), click=False); time.sleep(1.2)
            page.locator("#an-file input[type=file]").set_input_files(A.upload)   # the real .1D file
            time.sleep(1.0); nochap()
            wait_done(); time.sleep(2.5)
            glide("#an-steps", -120, 3.5)
            glide("#an-gauge", -160, 4.0)
            glide("#an-conn", -40, 5.0)
            rotate_brain(); time.sleep(1.5)
            glide("#an-table", -60, 4.5)
            top()

        def s2_known():
            chap("01", "A subject with a known diagnosis, never seen in training")
            move_to(page.locator("#an-random")); nochap()
            wait_done(); time.sleep(1.5)
            glide("#an-gauge", -160, 5.0)
            top()

        def s3_federated():
            tab("🌐 Federated training")
            chap("02", "Federated training: scans never leave the hospitals")
            box = page.locator("#fed-rounds input[type=number]")
            move_to(box); box.fill(str(A.rounds)); page.keyboard.press("Enter"); time.sleep(0.8)
            move_to(page.locator("#fed-go")); time.sleep(1.0); nochap()
            time.sleep(6); glide("#fed-net", -120, 6)
            t0 = time.time()
            while page.locator("#fed-curve").is_visible() is False and time.time() - t0 < 60:
                time.sleep(0.5)
            glide("#fed-curve", -420, 2)
            wait_done(); time.sleep(2)
            glide("#fed-cmp", -380, 6)
            top()

        def s4_evolution():
            tab("⏳ Connectivity evolution")
            chap("03", "Predict how the connectome evolves (last third of the scan)")
            time.sleep(2.5)
            move_to(page.locator("#evo-random")); nochap()
            wait_done(); time.sleep(1.5)
            glide("#evo-fig", -60, 7)
            top()

        def s5_model():
            tab("🧠 Model")
            chap("04", "How the model works, and the logged results")
            time.sleep(4); page.mouse.wheel(0, 380); time.sleep(5); nochap(); top()

        for fn in (s0_intro, s1_upload, s2_known, s3_federated, s4_evolution, s5_model):
            scene(fn)
        nochap()
        page.evaluate("([t, s]) => window.recTitle(t, s)", ["FedKG-Brain", A.author])
        time.sleep(4.5)
        vid = page.video.path()
        ctx.close(); b.close()
    return vid


def ffmpeg_bin():
    f = shutil.which("ffmpeg")
    if f:
        return f
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


if __name__ == "__main__":
    t0 = time.time()
    webm = run()
    ff = ffmpeg_bin()
    for crf in (28, 32, 36, 40):     # keep it under ~19.5 MB so it fits in a slide
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", A.out], check=True)
        if os.path.getsize(A.out) < 19.5e6:
            break
    still = os.path.splitext(A.out)[0] + "_still.png"
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "30", "-i", A.out, "-frames:v", "1", still], check=False)
    print(f"saved {A.out} ({os.path.getsize(A.out) / 1e6:.1f} MB) and {still} in {time.time() - t0:.0f}s")
