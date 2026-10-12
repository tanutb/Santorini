"""Play a full game against the agent by clicking the 3D canvas, checking every pick against the server.

Optional check; needs Playwright for Python and an installed Chrome:
    python scripts/browser_play.py http://127.0.0.1:8000 shots/game [desktop|mobile] [seed] [level]
Screenshots are written with the given prefix. Fails on a wrong pick, a stalled game or a page error.
"""
import random
import sys
import time
from playwright.sync_api import sync_playwright

url, prefix = sys.argv[1], sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 3 else "desktop"
rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 3)
size = (1440, 900) if device == "desktop" else (390, 844)
errors, shots = [], set()

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
    context = browser.new_context(viewport={"width": size[0], "height": size[1]}, device_scale_factor=1 if device == "desktop" else 2,
                                  is_mobile=device == "mobile", has_touch=device == "mobile")
    page = context.new_page()
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url, wait_until="networkidle")
    page.wait_for_function("window.santorini && window.santorini.board.kind === '3d'", timeout=15000)
    if len(sys.argv) > 5:
        page.wait_for_selector(f"[data-level={sys.argv[5]}]")
        page.click(f"[data-level={sys.argv[5]}]")
    page.click("#start-button")
    page.wait_for_function("window.santorini.room && window.santorini.room.game")
    print("level", page.evaluate("window.santorini.room.level"))

    def room():
        return page.evaluate("window.santorini.room")

    def point(cell):
        return page.evaluate(f"window.santorini.board.cellScreenPosition({cell})")

    def shot(name):
        if name not in shots:
            shots.add(name)
            page.wait_for_timeout(700)
            page.screenshot(path=f"{prefix}-{device}-{name}.png")

    def click(cell, hover_name=None):
        page.evaluate("document.getElementById('board-frame').scrollIntoView({block: 'start'})")
        position = point(cell)
        if position is None:
            state = room()
            page.screenshot(path=f"{prefix}-{device}-hidden.png")
            raise AssertionError(f"no clickable point for {cell}: workers={state['game']['workers']} heights={state['game']['heights']} "
                                 f"phase={state['game']['phase']} diag={page.evaluate('JSON.stringify(window.santorini.board.diagnostics)')} "
                                 "rect=" + page.evaluate("JSON.stringify(document.getElementById('board').getBoundingClientRect())"))
        if device == "mobile":
            page.touchscreen.tap(position["x"], position["y"])
        else:
            page.mouse.move(position["x"], position["y"])
            if hover_name:
                shot(hover_name)
            page.mouse.click(position["x"], position["y"])

    def wait_revision(revision, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            state = room()
            if state["game"]["revision"] != revision:
                return state
            page.wait_for_timeout(60)
        raise AssertionError("revision did not advance")

    PICK_CHECK = """(() => { const b = window.santorini.board, bad = [];
      const r = window.santorini.room, g = r.game;
      if (g.phase !== 'move') for (const c of r.legal) { if (!b.cellScreenPosition(c)) bad.push(c); }
      return bad; })()"""
    mismatches = []
    actions, picks = 0, 0
    deadline = time.time() + 180
    while time.time() < deadline:
        state = room()
        game = state["game"]
        if game["winner"] is not None:
            break
        if game["current"] != state["seat"]:
            if game["phase"] == "move" and game["revision"] > 4:
                shot("agent-thinking")
            page.wait_for_timeout(150)
            continue
        legal = state["legal"]
        bad = page.evaluate(PICK_CHECK)
        if bad:
            mismatches.append((game["revision"], bad, list(game["heights"])))
        if game["phase"] == "place":
            cell = rng.choice(legal)
            if not game["workers"][0][0] >= 0:
                shot("placement")
            click(cell, "place-hover")
            state = wait_revision(game["revision"])
        elif game["phase"] == "move":
            if game["revision"] > 6 and "hint" not in shots:
                page.click("#hint-button")
                page.wait_for_function("!document.getElementById('hint-result').hidden", timeout=8000)
                shot("hint")
                page.keyboard.press("Escape")
            wins = [a for a in legal if game["heights"][game["workers"][0][(a - 25) // 25]] == 2 and game["heights"][(a - 25) % 25] == 3]
            action = wins[0] if wins else rng.choice(legal)
            worker, target = divmod(action - 25, 25)
            click(game["workers"][0][worker])
            page.wait_for_timeout(120)
            click(target, "move-hover")
            state = wait_revision(game["revision"])
            moved = state["game"]["workers"][0][worker]
            if moved != target:
                pos = point(target)
                print("FAIL move", dict(action=action, worker=worker, target=target, workers=game["workers"], heights=game["heights"],
                      after=state["game"]["workers"], legal=legal, pick_at_target=page.evaluate(f"window.santorini.board.cellAt({pos['x']},{pos['y']})"),
                      history=state["history"][-3:]))
                page.screenshot(path=f"{prefix}-{device}-fail.png")
            assert moved == target, f"picked {moved} instead of {target}"
            picks += 1
            if state["game"]["phase"] == "build":
                shot("build-phase")
        else:
            cell = rng.choice(legal)
            click(cell, "build-hover")
            state = wait_revision(game["revision"])
            assert state["game"]["heights"][cell] == game["heights"][cell] + 1, "build landed on the wrong square"
            picks += 1
        actions += 1
    final = room()
    page.wait_for_timeout(1200)
    page.screenshot(path=f"{prefix}-{device}-end.png")
    diagnostics = page.evaluate("JSON.stringify(window.santorini.board.diagnostics)")
    print("winner", final["game"]["winner"], "reason", final["game"]["reason"], "revision", final["game"]["revision"],
          "my actions", actions, "verified picks", picks)
    print("diagnostics", diagnostics)
    print("console errors", errors or "none")
    print("hidden playable squares", mismatches or "none")
    assert not errors, errors
    browser.close()
