// Hash router. Each route is `{path, render}` where render is async and
// returns a teardown function (or void). Switching routes runs teardown
// before the next render so things like WebSockets and intervals release.

const routes = new Map();
let currentTeardown = null;

export function registerRoute(path, render) {
  routes.set(path, render);
}

export function start({ defaultRoute = "/live", outlet, titleEl, navEl }) {
  async function go() {
    const hash = location.hash || `#${defaultRoute}`;
    const path = hash.replace(/^#/, "").split("?")[0];
    const render = routes.get(path) || routes.get(defaultRoute);
    if (!render) return;

    if (typeof currentTeardown === "function") {
      try { currentTeardown(); } catch (e) { console.warn("teardown failed", e); }
    }
    currentTeardown = null;

    outlet.innerHTML = "";
    if (titleEl) titleEl.textContent = path.replace(/^\//, "").replace(/^./, c => c.toUpperCase()) || "Live";

    if (navEl) {
      navEl.querySelectorAll("[data-route]").forEach(a => {
        a.classList.toggle("active", "/" + a.dataset.route === path);
      });
    }
    try {
      const teardown = await render(outlet);
      currentTeardown = teardown;
    } catch (e) {
      console.error("route render failed", e);
      outlet.innerHTML = `<div class="toast err">页面加载失败:${escapeHtml(String(e.message || e))}</div>`;
    }
  }

  window.addEventListener("hashchange", go);
  go();
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
