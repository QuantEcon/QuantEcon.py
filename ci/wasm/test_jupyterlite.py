"""
Playwright smoke tests for QuantEcon.py in the actual Emscripten/JupyterLite
environment.  A locally-built JupyterLite site (xeus-python WASM kernel) is
served on localhost:8000; these tests drive it headlessly.

Build the site first:
    jupyter lite build --XeusAddon.environment_file=ci/wasm/environment.yml \\
                       --output-dir=_site

Then run:
    pytest ci/wasm/test_jupyterlite.py --browser chromium -v
"""
import pytest
from playwright.sync_api import Browser, Page

SITE = "http://localhost:8000"

# First run must download WASM packages + compile with Numba — keep generous.
BOOT_MS = 600_000   # 10 min
EXEC_MS  = 180_000  # 3 min per cell


# ---------------------------------------------------------------------------
# Browser-page fixture: one page shared across all tests in the module so
# the WASM kernel boots only once.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def console(browser: Browser) -> Page:
    ctx  = browser.new_context()
    page = ctx.new_page()
    _open_console(page)
    yield page
    ctx.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_console(page: Page) -> None:
    """Navigate to JupyterLite and open the xeus-python console."""
    page.goto(f"{SITE}/lab/index.html", timeout=BOOT_MS)

    # Wait for the Launcher to appear
    page.wait_for_selector(".jp-Launcher", timeout=BOOT_MS)

    # Click the first Console launcher card (xeus-python)
    page.locator(".jp-LauncherCard[data-category='Console']").first.click()

    # Wait for the console widget
    page.wait_for_selector(".jp-CodeConsole", timeout=BOOT_MS)

    # Wait for the kernel to reach idle (downloads + first JIT compile)
    _wait_idle(page, BOOT_MS)


def _wait_idle(page: Page, timeout: int) -> None:
    """Block until the JupyterLab kernel status indicator reads 'Idle'."""
    page.wait_for_function(
        """() => {
            const el = document.querySelector(
                ".jp-Toolbar-kernelStatus, [data-status]");
            if (!el) return false;
            const txt = (el.textContent || el.dataset.status || "").toLowerCase();
            return txt.includes("idle");
        }""",
        timeout=timeout,
    )


def _run(page: Page, code: str, timeout: int = EXEC_MS) -> str:
    """
    Paste *code* into the active console prompt, execute it with Shift+Enter,
    wait for the kernel to return to idle, and return the last output text.
    """
    prompt = page.locator(".jp-CodeConsole-input .cm-content, "
                          ".jp-Console-promptCell .CodeMirror")
    prompt.last.click()
    page.keyboard.press("Control+a")
    page.keyboard.type(code)
    page.keyboard.press("Shift+Enter")
    _wait_idle(page, timeout)

    outputs = page.locator(".jp-OutputArea-output").all()
    return "\n".join(o.inner_text() for o in outputs[-5:]) if outputs else ""


# ---------------------------------------------------------------------------
# Tests — key items from the smoke checklist, run inside the real WASM kernel
# ---------------------------------------------------------------------------

def test_kernel_boots(console: Page):
    """xeus-python WASM kernel loads and reaches idle without error."""
    # If the fixture succeeds the kernel already booted; just assert no crash.
    assert console.url.startswith(SITE)


def test_import_quantecon(console: Page):
    """import quantecon succeeds in the Emscripten environment."""
    out = _run(console, "import quantecon as qe; print('v', qe.__version__)")
    assert "v" in out, f"unexpected output: {out!r}"


def test_tauchen(console: Page):
    """Plain lazy @njit path: tauchen discretises an AR(1) correctly."""
    code = (
        "import quantecon as qe, numpy as np\n"
        "mc = qe.tauchen(5, 0.9, 0.1)\n"
        "ok = mc.P.shape == (5,5) and np.allclose(mc.P.sum(1), 1)\n"
        "print('ok' if ok else 'FAIL')"
    )
    assert "ok" in _run(console, code), "tauchen failed"


def test_np_linalg_solve_jit(console: Page):
    """
    np.linalg.solve inside @njit works on Emscripten.  This is the clean
    proxy test for the numba_xgesv/_LAPACK mechanism that #927 depends on:
    if this passes, _numba_linalg_solve almost certainly works too.
    """
    code = (
        "from numba import njit; import numpy as np\n"
        "@njit\n"
        "def _s(A, b): return np.linalg.solve(A, b)\n"
        "x = _s(np.array([[3.,2.],[1.,-1.]]), np.array([8.,1.]))\n"
        "print('ok' if abs(x[0]-2.0)<1e-4 else 'FAIL')"
    )
    out = _run(console, code, timeout=EXEC_MS)
    assert "ok" in out, f"_LAPACK proxy failed: {out!r}"


def test_support_enumeration(console: Page):
    """End-to-end support_enumeration: exercises _numba_linalg_solve (#927)."""
    code = (
        "import quantecon as qe\n"
        "bm=[[(3,3),(3,2)],[(2,2),(5,6)],[(0,3),(6,1)]]\n"
        "g=qe.game_theory.NormalFormGame(bm)\n"
        "nes=qe.game_theory.support_enumeration(g)\n"
        "print(len(nes))"
    )
    out = _run(console, code)
    assert "3" in out, f"support_enumeration unexpected output: {out!r}"


def test_gini_fails_on_emscripten(console: Page):
    """
    gini_coefficient must raise on Emscripten: @njit(parallel=True) + prange
    is not supported by the WASM Numba build (#926).
    """
    code = (
        "import quantecon as qe, numpy as np\n"
        "try:\n"
        "    qe.gini_coefficient(np.array([1.,2.,3.]))\n"
        "    print('NO_ERROR')\n"
        "except Exception:\n"
        "    print('EXPECTED_ERROR')"
    )
    out = _run(console, code)
    assert "EXPECTED_ERROR" in out, (
        f"gini_coefficient should fail on Emscripten but did not: {out!r}"
    )
