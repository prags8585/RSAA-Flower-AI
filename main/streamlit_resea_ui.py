# streamlit_resea_ui.py — fast, responsive control panel (Windows-friendly)
# - Optimistic chips + short polling (no hard refresh)
# - Concurrent, low-timeout status checks (no spinner)
# - Start in a process group; stop kills child server too
# - No stdout PIPE capture (prevents stalls)
#
# Usage:
#   pip install -r requirements.txt
#   pip install streamlit requests psutil
#   streamlit run streamlit_resea_ui.py

import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

# Optional deps (safe fallbacks if missing)
try:
    import requests
except Exception:
    requests = None
try:
    import psutil
except Exception:
    psutil = None

PROJECT_ROOT = Path(__file__).resolve().parent
PY_EXE = sys.executable

st.set_page_config(page_title="ReSEA Control Panel", layout="wide")
st.title("🌸 ReSEA — Control Panel")

# --------------------- ultra-fast status probes ---------------------
def check_port(port: int, host: str = "127.0.0.1", timeout: float = 0.06) -> bool:
    """Fast TCP connect; tiny timeout so UI never hangs."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def http_ok(url: str, timeout: float = 0.15) -> bool:
    """Optional HTTP probe. Defaults tiny timeouts."""
    if requests is None:
        return False
    try:
        r = requests.get(url, timeout=timeout)
        return bool(r.ok)
    except Exception:
        return False

def service_up(port: int, health_path: Optional[str] = None) -> bool:
    """Prefer blazing-fast TCP; optionally confirm with quick HTTP."""
    tcp = check_port(port)
    if not tcp:
        return False
    if health_path:
        return http_ok(f"http://127.0.0.1:{port}{health_path}")
    return True

def services_status(specs: Dict[str, Tuple[int, Optional[str]]]) -> Dict[str, bool]:
    """Concurrent status for all services (name -> (port, health_path))."""
    out: Dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futs = {ex.submit(service_up, port, hp): name for name, (port, hp) in specs.items()}
        for fut in as_completed(futs):
            out[futs[fut]] = bool(fut.result())
    return out

# --------------------- process control (group-aware) ---------------------
def start_bg(name: str, cmd, env: Optional[Dict[str, str]] = None):
    """Start uvicorn in its own process group so we can kill children on stop."""
    if "procs" not in st.session_state:
        st.session_state["procs"] = {}
    if name in st.session_state["procs"]:
        p = st.session_state["procs"][name]
        if p and p.poll() is None:
            st.warning(f"{name} already running.")
            return

    the_env = os.environ.copy()
    if env:
        the_env.update(env)

    creationflags = 0
    preexec_fn = None
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # Windows group
    else:
        import os as _os  # POSIX: new session
        preexec_fn = _os.setsid  # type: ignore[attr-defined]

    # DO NOT pipe stdout/stderr (prevents log backpressure stalls)
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=the_env,
        stdout=None,
        stderr=None,
        creationflags=creationflags,
        preexec_fn=preexec_fn,
    )
    st.session_state["procs"][name] = proc
    st.toast(f"Started {name} (pid={proc.pid})", icon="🚀")

def stop_bg(name: str, port: int):
    """Stop process group; if port owner isn't ours, try best-effort fallback."""
    p = st.session_state.get("procs", {}).get(name)
    if p and p.poll() is None:
        try:
            if os.name == "nt":
                # signal the whole group; fallback to terminate/kill
                p.send_signal(subprocess.signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                time.sleep(0.2)
                p.terminate()
            else:
                import os as _os, signal
                _os.killpg(_os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
        try:
            p.wait(timeout=2.0)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    else:
        st.info(f"{name}: no tracked process to stop.")

    # If port is still up, attempt to kill whoever owns it (optional; needs psutil)
    if check_port(port) and psutil:
        try:
            for proc in psutil.process_iter(attrs=["pid", "name", "connections"]):
                for c in proc.info.get("connections", []):
                    if getattr(c, "laddr", None) and getattr(c.laddr, "port", None) == port:
                        try:
                            proc.terminate()
                            proc.wait(timeout=1.5)
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
        except Exception:
            pass

# --------------------- UI helpers ---------------------
def chip(state: str) -> str:
    return {
        "up": "🟢 UP",
        "down": "🔴 DOWN",
        "starting": "🟡 STARTING",
        "stopping": "🟠 STOPPING",
    }.get(state, "🔘")

def optimistic_flip(ph, label: str, interim: str, done: bool, final_ok: bool):
    ph.metric(label, chip(interim))
    ph.metric(label, chip("up" if final_ok else "down"))

def wait_for(cond, timeout: float = 2.0, interval: float = 0.1) -> bool:
    t0 = time.time()
    ok = cond()
    while not ok and (time.time() - t0) < timeout:
        time.sleep(interval)
        ok = cond()
    return ok

# --------------------- layout: status chips ---------------------
SERVICES = {
    "EU":   (8001, "/health"),   # (port, optional health path)
    "US":   (8002, "/health"),
    "APAC": (8003, "/health"),
    "COORD":(8000, "/healthz"),  # your coordinator exposes /healthz
}

with st.container(border=True):
    st.subheader("Service status")

    # placeholders for live updates
    c1, c2, c3, c4 = st.columns(4)
    ph = {
        "EU": c1.empty(),
        "US": c2.empty(),
        "APAC": c3.empty(),
        "COORD": c4.empty(),
    }

    # draw initial status concurrently (fast)
    status = services_status(SERVICES)
    ph["EU"].metric("EU :8001", chip("up" if status["EU"] else "down"))
    ph["US"].metric("US :8002", chip("up" if status["US"] else "down"))
    ph["APAC"].metric("APAC :8003", chip("up" if status["APAC"] else "down"))
    ph["COORD"].metric("Coord :8000", chip("up" if status["COORD"] else "down"))
    st.caption(f"Last checked: {datetime.now().strftime('%H:%M:%S')}")

# --------------------- layout: controls ---------------------
with st.container(border=True):
    st.subheader("Launch / Stop services (instant feedback)")

    colA, colB = st.columns(2)

    # ---------- Regions ----------
    with colA:
        st.write("**Region services**")

        # EU
        a1, a2 = st.columns(2)
        if a1.button("Start EU (8001)"):
            ph["EU"].metric("EU :8001", chip("starting"))
            start_bg(
                "EU",
                [PY_EXE, "-m", "uvicorn", "region.region_service:app",
                 "--port", "8001", "--no-access-log", "--log-level", "warning"],
                env={"REGION_NAME": "EU"},
            )
            ok = wait_for(lambda: service_up(*SERVICES["EU"]))
            ph["EU"].metric("EU :8001", chip("up" if ok else "down"))

        if a2.button("Stop EU"):
            ph["EU"].metric("EU :8001", chip("stopping"))
            stop_bg("EU", port=8001)
            ok = wait_for(lambda: not service_up(*SERVICES["EU"]))
            ph["EU"].metric("EU :8001", chip("down" if ok else "up"))

        # US
        b1, b2 = st.columns(2)
        if b1.button("Start US (8002)"):
            ph["US"].metric("US :8002", chip("starting"))
            start_bg(
                "US",
                [PY_EXE, "-m", "uvicorn", "region.region_service:app",
                 "--port", "8002", "--no-access-log", "--log-level", "warning"],
                env={"REGION_NAME": "US"},
            )
            ok = wait_for(lambda: service_up(*SERVICES["US"]))
            ph["US"].metric("US :8002", chip("up" if ok else "down"))

        if b2.button("Stop US"):
            ph["US"].metric("US :8002", chip("stopping"))
            stop_bg("US", port=8002)
            ok = wait_for(lambda: not service_up(*SERVICES["US"]))
            ph["US"].metric("US :8002", chip("down" if ok else "up"))

        # APAC
        c1, c2 = st.columns(2)
        if c1.button("Start APAC (8003)"):
            ph["APAC"].metric("APAC :8003", chip("starting"))
            start_bg(
                "APAC",
                [PY_EXE, "-m", "uvicorn", "region.region_service:app",
                 "--port", "8003", "--no-access-log", "--log-level", "warning"],
                env={"REGION_NAME": "APAC"},
            )
            ok = wait_for(lambda: service_up(*SERVICES["APAC"]))
            ph["APAC"].metric("APAC :8003", chip("up" if ok else "down"))

        if c2.button("Stop APAC"):
            ph["APAC"].metric("APAC :8003", chip("stopping"))
            stop_bg("APAC", port=8003)
            ok = wait_for(lambda: not service_up(*SERVICES["APAC"]))
            ph["APAC"].metric("APAC :8003", chip("down" if ok else "up"))

    # ---------- Coordinator ----------
    with colB:
        st.write("**Coordinator (8000)**")
        d1, d2 = st.columns(2)
        if d1.button("Start Coordinator (8000)"):
            ph["COORD"].metric("Coord :8000", chip("starting"))
            start_bg(
                "COORD",
                [PY_EXE, "-m", "uvicorn", "coordinator.coordinator_service:app",
                 "--port", "8000", "--no-access-log", "--log-level", "warning"]
            )
            ok = wait_for(lambda: service_up(*SERVICES["COORD"]))
            ph["COORD"].metric("Coord :8000", chip("up" if ok else "down"))

        if d2.button("Stop Coordinator"):
            ph["COORD"].metric("Coord :8000", chip("stopping"))
            stop_bg("COORD", port=8000)
            ok = wait_for(lambda: not service_up(*SERVICES["COORD"]))
            ph["COORD"].metric("Coord :8000", chip("down" if ok else "up"))

# --------------------- Data seeding ---------------------
with st.container(border=True):
    st.subheader("Seed sample documents")
    if st.button("Index demo data (scripts/index_sample_data.py)"):
        cp = subprocess.run([PY_EXE, "scripts/index_sample_data.py"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        st.code((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else ""))
        st.success("Indexed sample data." if cp.returncode == 0 else "Indexing failed.")

# --------------------- Query ---------------------
with st.container(border=True):
    st.subheader("Ask a question")
    q = st.text_input("Your question", value="How do we scale the service for a traffic spike?")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_ollama = st.toggle("Use Ollama summarization", value=False)
    with col2:
        ollama_url = st.text_input("OLLAMA_BASE_URL", value="http://localhost:11434")
    with col3:
        ollama_model = st.text_input("OLLAMA_MODEL", value="llama3")

    if st.button("Run query via scripts/query.py"):
        env = os.environ.copy()
        if use_ollama:
            env["OLLAMA_BASE_URL"] = ollama_url
            env["OLLAMA_MODEL"] = ollama_model
        cp = subprocess.run([PY_EXE, "scripts/query.py", q], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
        st.write("**Result**" if cp.returncode == 0 else "**Error**")
        st.code(cp.stdout if cp.stdout else "(no output)")
        if cp.stderr:
            with st.expander("stderr"):
                st.code(cp.stderr)

# --------------------- Flower training ---------------------
with st.container(border=True):
    st.subheader("Train reranker federatively (Flower)")
    rounds = st.slider("Rounds", 1, 10, 3, 1)
    if st.button("Start training (fed/train_reranker_flower.py)"):
        cp = subprocess.run([PY_EXE, "fed/train_reranker_flower.py", "--rounds", str(rounds)], cwd=PROJECT_ROOT, capture_output=True, text=True)
        st.code((cp.stdout or "") + ("\n" + cp.stderr if cp.stderr else ""))
        st.success("Training finished." if cp.returncode == 0 else "Training failed.")

# --------------------- Footer ---------------------
st.caption("Buttons update chips instantly and then verify state via fast TCP/HTTP probes. "
          "If Stop doesn't go DOWN, a different process owns the port; this file tries best-effort to terminate it (psutil).")
