#!/usr/bin/env node
/* Functional UI test for llm_router/static/dashboard.html using jsdom.
   Loads the real HTML, stubs window.fetch with canned API data, and
   exercises every feature. Exits 0 if all checks pass, 1 otherwise. */
const fs = require("fs");
const path = require("path");
const { JSDOM } = require("jsdom");

const HTML = fs.readFileSync(path.join(__dirname, "..", "llm_router", "static", "dashboard.html"), "utf-8");

let failures = 0, passes = 0;
function check(name, cond) {
  if (cond) { passes++; console.log("  ok  " + name); }
  else { failures++; console.log("FAIL  " + name); }
}

function makeData(running) {
  return {
    instances: running ? {
      default: { port: 18590, path: "/cfg/dash.yaml", running: true },
      instances: [{
        port: 18590, base_url: "http://127.0.0.1:18590", reached: true, pid: 12345,
        sources: ["pidfile"], active: "qwen-local",
        router: { pp_current: 21.5, tg_current: 101.2, pp_avg: 22.1, tg_avg: 99.8 },
        router_recent: [
          { model: "qwen-local", pp: 21.5, tg: 101.2 },
          { model: "qwen-local", pp: 22.1, tg: 99.8 }
        ],
        models: [
          { name: "qwen-local", kind: "chat", external: true, port: 18501, pid: 111, active: true, running: true,
            llm: { pp_current: 21.5, tg_current: 101.2, pp_avg: 22.1, tg_avg: 99.8 } },
          { name: "embed-local", kind: "embedding", external: true, port: 18502, pid: 222, active: false, running: false,
            llm: { pp_current: null, tg_current: null, pp_avg: null, tg_avg: null } }
        ]
      }],
      suggested_port: 18591, version: "0.1"
    } : {
      default: { port: 18590, path: "/cfg/dash.yaml", running: false },
      instances: [], suggested_port: 18591, version: "0.1"
    },
    config: {
      path: "/cfg/dash.yaml", valid: true, errors: [],
      router: { host: "127.0.0.1", port: 18590, api_key: "", default_model: "qwen-local", startup_timeout: 30,
        health_interval: 2, health_path: "/health", log_dir: "/tmp/logs", shutdown_grace: 10,
        keep_inactive_stopped: true, preload: true },
      models: [
        { name: "qwen-local", command: "python mock_llm.py --port 18501", upstream_model: "qwen-local",
          host: "127.0.0.1", port: 18501, startup_timeout: 25, health_path: "/health", context_length: 65536, kind: "chat", env: {} },
        { name: "embed-local", command: "python mock_llm.py --port 18502", port: 18502, kind: "embedding",
          context_length: 8192, env: { FOO: "bar" } }
      ],
      dashboard: { host: "127.0.0.1", port: 18930, browser: false },
      raw: "router:\n  port: 18590\n"
    },
    logs: { logs: [
      { id: "router-18590", kind: "router", label: "router :18590", exists: true },
      { id: "llm-18501", kind: "llm", label: "qwen-local (:18501)", exists: true }
    ] },
    calls: []   // records (url, method, body) for action assertions
  };
}

function buildDom(data) {
  const enc = new TextEncoder();
  const dom = new JSDOM(HTML, {
    runScripts: "dangerously",
    pretendToBeVisual: true,
    url: "http://127.0.0.1/",
    beforeParse(window) {
      window.matchMedia = window.matchMedia || ((q) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {} }));
      window.confirm = () => true;   // stub the Stop-instance confirmation dialog
      window.TextDecoder = TextDecoder;
      window.TextEncoder = TextEncoder;
      window.AbortController = AbortController;
      window.fetch = async (url, opts) => {
        const method = (opts && opts.method) || "GET";
        let body;
        try { body = opts && opts.body ? JSON.parse(opts.body) : undefined; } catch (e) { body = opts && opts.body; }
        data.calls.push({ url: String(url), method, body });
        if (String(url).includes("/api/logs/stream")) {
          const chunks = [enc.encode('llama.cpp prompt eval time = 100 ms | 20.0 t/s\n')];
          let i = 0;
          return { ok: true, status: 200, body: { getReader: () => ({ read: async () => i < chunks.length ? { done: false, value: chunks[i++] } : { done: true, value: undefined } }) } };
        }
        let out;
        if (method === "PUT" && url === "/api/config") out = { ok: true, path: "/cfg/dash.yaml" };
        else if (method === "PUT" && url === "/api/config/raw") out = { ok: true, path: "/cfg/dash.yaml" };
        else if (url === "/api/instances") out = data.instances;
        else if (url === "/api/config") out = data.config;
        else if (url === "/api/logs") out = data.logs;
        else if (url === "/api/run-default") out = { ok: true, already_running: false, port: 18590 };
        else if (url === "/api/run-instance") out = { ok: true, port: 18591, launch_pid: 99 };
        else if (url === "/api/rates/clear") out = { ok: true, cleared: 2 };
        else out = { ok: true };
        return { ok: true, status: 200, json: async () => out, text: async () => JSON.stringify(out) };
      };
    }
  });
  return dom;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  /* ---------- Scenario A: default instance RUNNING ---------- */
  console.log("Scenario A: default instance running");
  const A = makeData(true);
  const dom = buildDom(A);
  const w = dom.window, d = w.document;
  await sleep(500); // let init() resolve its fetches

  check("theme applied (light by default)", (d.documentElement.getAttribute("data-theme") || "") === "light");
  check("theme btn present", !!d.querySelector("#themeBtn"));
  check("tooltip box appended", !!d.querySelector(".tipbox"));
  const tipSet = new Set(Array.from(d.querySelectorAll(".tip")).map(t => t.getAttribute("data-tip")));
  check("tooltips reduced to the intended set (5-25)", d.querySelectorAll(".tip").length >= 5 && d.querySelectorAll(".tip").length <= 25, "count=" + d.querySelectorAll(".tip").length);
  check("added: router health_interval tooltip", tipSet.has("Seconds between readiness probes to each model backend. The router polls the model Health path every this many seconds until it answers; lower = faster readiness detection, higher = less polling load."));
  check("added: router shutdown_grace tooltip", tipSet.has("Seconds the router waits for a model to exit cleanly after SIGTERM before forcing it with SIGKILL on stop/shutdown."));
  check("added: model env tooltip", tipSet.has("Extra environment variables (KEY=VALUE, one per line) passed to this model backend when it starts — e.g. FLASH_ATTN=1, NUM_THREADS=8, or an HF token."));
  check("removed: theme button tooltip gone", !d.querySelector("#themeBtn").hasAttribute("data-tip"));
  check("removed: 'Current prompt' metric tooltip gone", !tipSet.has("Most recent prompt-processing speed (tokens/sec) — how fast this instance is chewing through the prompt."));
  // Task 1: Overview -> Running Instances -> active model badge tooltip removed
  check("task1: active-model tooltip removed from tip set", !tipSet.has("This model is the instance's active model — /v1 requests are routed to it right now."));
  check("4 tabs present", d.querySelectorAll(".tabs button").length === 4);

  // overview rendering
  check("instance card rendered", d.querySelectorAll("#instances .card").length === 1);
  // header: "Opened with config: <path>" (replaces the old "default config: ...")
  check("header: shows 'Opened with config: <path>'", d.querySelector("#cfgpath").textContent === "Opened with config: /cfg/dash.yaml");
  check("header: old 'default config:' label removed", !/default config:/.test(d.querySelector("#cfgpath").textContent));
  // Running-instances auto-refresh must be 5s (not the old 2s)
  check("auto-refresh: meta says 'every 5s'", /Auto-refreshing every 5s/.test(HTML));
  check("auto-refresh: polling interval is 5000ms", /setInterval\([\s\S]*?\}, 5000\);/.test(HTML) && !/setInterval\([\s\S]*?\}, 2000\);/.test(HTML));
  check("auto-refresh: tooltip says '5 seconds'", /Refreshes every 5 seconds\./.test(HTML));
  check("auto-refresh: no leftover 'every 2s' text", !/every 2s|every 2 seconds/.test(HTML));
  const card = d.querySelector("#instances .card");
  check("card shows :18590", /:18590/.test(card.textContent));
  check("card shows active model qwen-local", /active: qwen-local/.test(card.textContent));
  check("model table has qwen-local row", /qwen-local/.test(d.querySelector("#instances table").textContent));
  check("model table shows current pp 21.50", /21\.50/.test(card.textContent));
  check("model table shows avg tg 99.80", /99\.80/.test(card.textContent));
  check("router metric block present", d.querySelectorAll("#instances .metric").length === 4);
  check("task1: router rate block renders values (not —)", (()=>{ const g=card.querySelector(".grid"); return g && /101\.2/.test(g.textContent) && /22\.10/.test(g.textContent) && !/—/.test(g.textContent); })());
  // ---- Rate "last rates" hover window: hovering the router's Avg pp / Avg tg
  // reveals a floating table (model | pp | tg) of the recent rates. ----
  const hovers = Array.from(d.querySelectorAll('#instances .metric.rate-hover'));
  check("router Avg pp/tg cells are hoverable (.rate-hover, data-port=18590)", hovers.length === 2 && hovers.every(h => h.getAttribute('data-port') === '18590'), "count=" + hovers.length);
  const rateboxEl = d.querySelector('.ratebox');
  check("ratebox floating window is in the DOM", !!rateboxEl);
  if (rateboxEl) {
    // Hover "Avg pp" -> the table shows the pp column only (model | pp), no tg.
    hovers[0].dispatchEvent(new w.MouseEvent('mouseover', { bubbles: true }));
    check("hover Avg pp -> table shows the pp column only (no tg)",
      rateboxEl.style.display === 'block' && /model/.test(rateboxEl.textContent) && /pp/.test(rateboxEl.textContent) && !/tg/.test(rateboxEl.textContent),
      "display=" + rateboxEl.style.display + " html=" + rateboxEl.innerHTML);
    check("hover Avg pp -> lists qwen-local pp 21.50 (not the tg value 101.20)",
      /qwen-local/.test(rateboxEl.textContent) && /21\.50/.test(rateboxEl.textContent) && !/101\.20/.test(rateboxEl.textContent), rateboxEl.innerHTML);
    // Hover "Avg tg" -> the table shows the tg column only (model | tg), no pp.
    hovers[1].dispatchEvent(new w.MouseEvent('mouseover', { bubbles: true }));
    check("hover Avg tg -> table shows the tg column only (no pp)",
      rateboxEl.style.display === 'block' && /model/.test(rateboxEl.textContent) && /tg/.test(rateboxEl.textContent) && !/pp/.test(rateboxEl.textContent),
      rateboxEl.innerHTML);
    check("hover Avg tg -> lists qwen-local tg 101.20 (not the pp value 21.50)",
      /qwen-local/.test(rateboxEl.textContent) && /101\.20/.test(rateboxEl.textContent) && !/21\.50/.test(rateboxEl.textContent), rateboxEl.innerHTML);
    hovers[1].dispatchEvent(new w.MouseEvent('mouseout', { bubbles: true }));
    check("mouseout hides the ratebox", rateboxEl.style.display === 'none', "display=" + rateboxEl.style.display);
  }
  check("embedding row present", /embed-local/.test(card.textContent));
  const activeBadge = d.querySelector('#instances .badge.active');
  check("task1: active badge itself carries no ? tooltip", !!activeBadge && activeBadge.querySelectorAll('.tip').length === 0);
  // Stop-instance button (new feature)
  check("running instance has Stop button", !!d.querySelector('#instances [data-stop="18590"]'));
  d.querySelector('#instances [data-stop="18590"]').click();
  await sleep(60);
  check("Stop -> POST /api/instances/stop {port:18590}", A.calls.some(c => c.url === "/api/instances/stop" && c.method === "POST" && c.body && c.body.port === 18590));
  check("Stop -> status shown on #ovStatus", /Stopped instance on :18590/.test(d.querySelector("#ovStatus").textContent));
  // Clear-rates button (new feature) — clears all stored avg pp/tg rates
  check("Clear rates button present in Overview", !!d.querySelector("#clearRatesBtn"));
  d.querySelector("#clearRatesBtn").click();
  await sleep(60);
  check("Clear rates -> POST /api/rates/clear", A.calls.some(c => c.url === "/api/rates/clear" && c.method === "POST"));
  check("Clear rates -> status shows the cleared count", /Cleared stored rates for 2 model\(s\)\./.test(d.querySelector("#ovStatus").textContent));
  check("Clear rates -> button re-enabled after the action", !d.querySelector("#clearRatesBtn").disabled);
  check("banner: default running", (d.querySelector("#banner").className || "").includes("run") && /default running/i.test(d.querySelector("#banner").textContent));
  check("no run-default btn when running (pop-up hidden)", !d.querySelector("#runDefaultBtnPop") && !d.querySelector("#noInstPop.on"));

  // run form
  check("run port prefilled with suggested 18591", d.querySelector("#runPort").value === "18591");
  check("run model checkboxes (2)", d.querySelectorAll("#runModels .runModel").length === 2);
  check("run default select populated", d.querySelectorAll("#runDefault option").length === 2);

  // config tab
  const cfgRouterInputs = d.querySelectorAll("#routerForm [data-field]").length;
  check("router form has fields (>=11)", cfgRouterInputs >= 11);
  check("router port value = 18590", d.querySelector('#routerForm [data-field="port"]').value === "18590");
  check("2 model cards in config", d.querySelectorAll("#modelsForm .model-card").length === 2);
  check("raw yaml populated", d.querySelector("#rawYaml").value.includes("port: 18590"));
  check("config marked valid", /valid/.test(d.querySelector("#cfgValid").textContent));
  check("default model is a dropdown (select)", d.querySelector('#routerForm [data-field="default_model"]').tagName === "SELECT");
  // Task 2: Preload default model had two identical ? tooltips; keep only the upper (field-label) one
  const preloadInput = d.querySelector('#routerForm .switch input[data-field="preload"]');
  const preloadSpan = preloadInput ? preloadInput.nextElementSibling : null;
  check("task2: preload switch span has NO ? tooltip (lower duplicate removed)", !!preloadSpan && preloadSpan.querySelectorAll('.tip').length === 0 && preloadSpan.textContent === 'Preload default model');
  const preloadLabel = Array.from(d.querySelectorAll('#routerForm .field')).find(l => l.textContent.includes('Preload default model'));
  check("task2: preload field label keeps its ? tooltip (upper one kept)", !!preloadLabel && preloadLabel.querySelectorAll('.tip').length === 1);
  check("model card: host field removed", !d.querySelector('#modelsForm [data-mf="host"]'));
  check("model card: port field removed", !d.querySelector('#modelsForm [data-mf="port"]'));
  check("model card: command field present", !!d.querySelector('#modelsForm [data-mf="command"]'));
  // Task 3: all checkboxes are now iOS-style green/red sliders (native checkbox, restyled)
  const cssText = Array.from(d.querySelectorAll('style')).map(s => s.textContent).join('');
  check("task3: slider CSS present (appearance:none + rounded track)", /input\[type=checkbox\]\{[^}]*appearance:none/.test(cssText) && /input\[type=checkbox\]\{[^}]*border-radius:12px/.test(cssText));
  check("task3: on-state is green (#34c759)", /input\[type=checkbox\]:checked\{background:#34c759\}/.test(cssText));
  check("task3: off-state is red (#ff453a)", /input\[type=checkbox\]\{[^}]*background:#ff453a/.test(cssText));
  check("task3: knob slides right when on (translateX(18px))", /input\[type=checkbox\]:checked::before\{transform:translateX\(18px\)\}/.test(cssText));
  check("task3: checkboxes remain semantic checkboxes (state intact)", d.querySelectorAll('#routerForm input[type="checkbox"]').length === 2 && d.querySelector('#routerForm input[data-field="preload"]').checked === true && d.querySelector('#routerForm input[data-field="keep_inactive_stopped"]').checked === true);
  // "No instances running" pop-up CSS: in-flow card below the page selector; hidden by
  // default (display:none), .on puts it in the flow, .shown fades/slides the inner in
  check("popup CSS: in-flow (display:none by default; .on -> display:block)", /#noInstPop\{[^}]*display:none/.test(cssText) && /#noInstPop\.on\{[^}]*display:block/.test(cssText));
  check("popup CSS: shown state fades the inner card in (opacity:1)", /#noInstPop\.shown \.pop-inner\{[^}]*opacity:1/.test(cssText));
  check("popup element present in DOM", !!d.querySelector("#noInstPop"));
  // Task 4: empty text inputs show informative placeholders; optional fields marked "(optional)"
  const apiKeyInput = d.querySelector('#routerForm [data-field="api_key"]');
  check("task4: empty api_key shows (optional) placeholder", !!apiKeyInput && apiKeyInput.value === '' && /optional/i.test(apiKeyInput.getAttribute('placeholder')));
  const hostInput = d.querySelector('#routerForm [data-field="host"]');
  check("task4: filled host still has placeholder attr (hidden by value)", !!hostInput && !!hostInput.getAttribute('placeholder') && hostInput.value === '127.0.0.1');
  const embedCard = Array.from(d.querySelectorAll('#modelsForm .model-card')).find(c => (c.querySelector('[data-mf="name"]') || {}).value === 'embed-local');
  const upInput = embedCard && embedCard.querySelector('[data-mf="upstream_model"]');
  check("task4: empty model upstream_model shows (optional) placeholder", !!upInput && upInput.value === '' && /optional/i.test(upInput.getAttribute('placeholder')));
  const cmd0 = d.querySelector('#modelsForm [data-mf="command"]');
  check("task4: command input has a placeholder attr", !!cmd0 && !!cmd0.getAttribute('placeholder'));
  const env0 = d.querySelector('#modelsForm [data-mf="env"]');
  check("task4: env textarea has (optional) placeholder", !!env0 && /optional/i.test(env0.getAttribute('placeholder')));
  check("task4: runPort has a placeholder attr", !!d.querySelector('#runPort') && !!d.querySelector('#runPort').getAttribute('placeholder'));
  // Task 3: Command moved to its own full-width line (out of the grid) sitting above Env
  const mc0 = d.querySelector('#modelsForm .model-card');
  check("command NOT in the model grid (full-width now)", !!mc0 && !mc0.querySelector('.fgrid [data-mf="command"]'));
  const cmdEl = mc0 && mc0.querySelector('[data-mf="command"]');
  const envEl = mc0 && mc0.querySelector('[data-mf="env"]');
  check("command input is monospace (wider command line)", !!cmdEl && /monospace/.test(cmdEl.style.fontFamily));
  check("command is above env (DOM order)", !!cmdEl && !!envEl && !!(envEl.compareDocumentPosition(cmdEl) & w.Node.DOCUMENT_POSITION_PRECEDING));
  // dashboard section (new feature)
  check("dashboard form present with 3 fields", d.querySelectorAll("#dashboardForm [data-df]").length === 3);
  check("dashboard port value = 18930", d.querySelector('#dashboardForm [data-df="port"]').value === "18930");
  check("dashboard host value = 127.0.0.1", d.querySelector('#dashboardForm [data-df="host"]').value === "127.0.0.1");
  const dbCheck = d.querySelector('#dashboardForm [data-df="browser"]');
  check("dashboard browser is a checkbox", dbCheck && dbCheck.type === "checkbox");
  check("dashboard browser unchecked (config browser:false)", dbCheck && dbCheck.checked === false);
  // Task 1: the 3 Dashboard field tooltips (Host/Port/browser) are removed
  check("dashboard fields have NO ? tooltips", d.querySelectorAll("#dashboardForm .tip").length === 0);
  // Task 2: dynamically-built tips (built after page load) must show on hover (event delegation)
  const tipbox = d.querySelector('.tipbox');
  const dynRouterTip = d.querySelector('#routerForm .tip');
  check("router form has a dynamic .tip", !!dynRouterTip);
  if (dynRouterTip && tipbox) {
    dynRouterTip.dispatchEvent(new w.MouseEvent('mouseover', { bubbles: true }));
    check("hover dynamic router tip -> tipbox shows with text", tipbox.style.display === 'block' && tipbox.textContent.length > 10);
    dynRouterTip.dispatchEvent(new w.MouseEvent('mouseout', { bubbles: true }));
    check("mouseout hides the tipbox", tipbox.style.display === 'none');
  }
  const dynModelTip = d.querySelector('#modelsForm .tip');
  check("model card has a dynamic .tip", !!dynModelTip);
  if (dynModelTip && tipbox) {
    dynModelTip.dispatchEvent(new w.MouseEvent('mouseover', { bubbles: true }));
    check("hover dynamic model tip -> tipbox shows", tipbox.style.display === 'block');
    dynModelTip.dispatchEvent(new w.MouseEvent('mouseout', { bubbles: true }));
  }

  // tab switching
  const tabBtn = (t) => Array.from(d.querySelectorAll(".tabs button")).find((b) => b.dataset.tab === t);
  tabBtn("config").click(); await sleep(30);
  check("click Config -> #tab-config active", d.querySelector("#tab-config").classList.contains("active"));
  tabBtn("run").click(); await sleep(30);
  check("click Run -> #tab-run active", d.querySelector("#tab-run").classList.contains("active"));
  tabBtn("overview").click(); await sleep(30);
  check("click Overview -> #tab-overview active", d.querySelector("#tab-overview").classList.contains("active"));
  tabBtn("logs").click(); await sleep(30);
  check("click Logs -> #tab-logs active", d.querySelector("#tab-logs").classList.contains("active"));

  // logs
  check("log list rendered (2 items)", d.querySelectorAll("#logsList .log-item").length === 2);
  check("log item has Watch button", /Watch/.test(d.querySelector("#logsList .log-item button").textContent));
  // watch the router log (first item)
  d.querySelector("#logsList .log-item button").click();
  await sleep(150);
  check("log viewer opened", d.querySelector("#viewer").classList.contains("on"));
  check("task3: Auto-scroll is in the bubble (.log-fab)", !!d.querySelector("#viewer .log-fab #autoScroll"));
  check("task3b: .log-fab pinned to top-right corner (top+right, no bottom/left)", (function(){ const css=Array.from(d.querySelectorAll('style')).map(s=>s.textContent).join(''); const m=css.match(/\.viewer\s*\.log-fab\{[^}]*\}/); return !!(m && /top:/.test(m[0]) && /right:/.test(m[0]) && !/bottom:/.test(m[0]) && !/left:/.test(m[0])); })());
  check("task3: in-header 'Stop' button removed", !d.querySelector("#viewer #stopWatch"));
  check("log body has streamed line", /t\/s/.test(d.querySelector("#logbody").textContent));
  check("stream fetch called", A.calls.some((c) => c.url.includes("/api/logs/stream")));
  // toggle: after the first click the log-item button is now "Stop watching" (danger)
  const watchBtn2 = d.querySelector("#logsList .log-item button");
  check("watch toggle: button now 'Stop watching' (danger)", /Stop watching/.test(watchBtn2.textContent) && watchBtn2.classList.contains("danger"));
  // stopping now happens via the list's "Stop watching" button (the in-viewer Stop button was removed)
  watchBtn2.click(); await sleep(60);
  check("stop (via list 'Stop watching') closes viewer", !d.querySelector("#viewer").classList.contains("on"));

  // task 2: the live-log watch auto-closes when its target stops (e.g. `llm-router stop`)
  await sleep(80); // let refreshLogs() re-render the list back to "Watch" buttons
  d.querySelector("#logsList .log-item button").click(); await sleep(120);
  check("task2: viewer reopened for auto-close test", d.querySelector("#viewer").classList.contains("on"));
  check("task2: watched id still in list -> watch kept", w.maybeAutoCloseWatch({ logs: [{ id: "router-18590" }] }) === false && d.querySelector("#viewer").classList.contains("on"));
  check("task2: watched id gone (target stopped) -> watch auto-closes", w.maybeAutoCloseWatch({ logs: [] }) === true && !d.querySelector("#viewer").classList.contains("on"));

  // config: mode toggle to raw
  tabBtn("config").click(); await sleep(30);
  const rawBtn = Array.from(d.querySelectorAll(".mode-toggle button")).find((b) => b.dataset.mode === "raw");
  rawBtn.click();
  check("raw mode: raw pane visible", d.querySelector("#cfgRaw").style.display !== "none");
  check("raw mode: structured pane hidden", d.querySelector("#cfgStructured").style.display === "none");

  // add model
  const structBtn = Array.from(d.querySelectorAll(".mode-toggle button")).find((b) => b.dataset.mode === "structured");
  structBtn.click();
  const before = d.querySelectorAll("#modelsForm .model-card").length;
  d.querySelector("#addModelBtn").click();
  check("add model -> card count increments", d.querySelectorAll("#modelsForm .model-card").length === before + 1);

  // save structured config
  d.querySelector("#saveCfgBtn").click();
  await sleep(60);
  const put = A.calls.find((c) => c.url === "/api/config" && c.method === "PUT");
  check("save -> PUT /api/config with {router,models}", !!put && !!put.body.router && Array.isArray(put.body.models));
  check("save -> PUT body includes dashboard", !!put && !!put.body.dashboard && put.body.dashboard.port === 18930 && put.body.dashboard.browser === false);
  check("save status shown", /Saved to/.test(d.querySelector("#cfgStatus").textContent));

  // theme toggle
  const beforeTheme = d.documentElement.getAttribute("data-theme");
  d.querySelector("#themeBtn").click();
  check("theme toggles", d.documentElement.getAttribute("data-theme") !== beforeTheme);

  dom.window.close();

  /* ---------- Scenario B: default instance NOT running ---------- */
  console.log("Scenario B: no instances running");
  const B = makeData(false);
  const domB = buildDom(B);
  const d2 = domB.window.document;
  await sleep(500);
  check("empty-state message shown", /No router instances/.test(d2.querySelector("#instances").textContent));
  // Pop-up replaces the inline "not running" banner when nothing is running
  check("popup: shown when no instances (class 'on')", d2.querySelector("#noInstPop").classList.contains("on"));
  check("popup: says 'No instances running' + 'One-click start'", /No instances running/.test(d2.querySelector("#noInstPop").textContent) && /One-click start/.test(d2.querySelector("#noInstPop").textContent));
  check("popup: has Run default button", !!d2.querySelector("#runDefaultBtnPop"));
  check("popup: 'no instances' badge restored on the left", !!d2.querySelector("#noInstPop .badge") && /no instances/.test(d2.querySelector("#noInstPop .badge").textContent));
  check("popup: aria-hidden cleared while shown", d2.querySelector("#noInstPop").getAttribute("aria-hidden") === "false");
  check("inline banner hidden when nothing running", d2.querySelector("#banner").style.display === "none");
  // Regression: the pop-up must NOT auto-start the instance on load.
  // (Old bug: updateNoInstPop called startDefaultInstance(), which immediately
  //  POSTed /api/run-default — so `stop` was impossible, because the 5s poller
  //  rebuilt the pop-up and re-fired run-default. The instance now starts only
  //  on an explicit click of the button.)
  check("popup: does NOT auto-start on load (no /api/run-default POST yet)",
        !B.calls.some((c) => c.url === "/api/run-default" && c.method === "POST"));
  // click the pop-up's run-default button
  d2.querySelector("#runDefaultBtnPop").click();
  await sleep(80);
  check("run-default -> POST /api/run-default called", B.calls.some((c) => c.url === "/api/run-default" && c.method === "POST"));
  domB.window.close();

  console.log("\nScenario C: shared-port external model detection (dashboard rendering)");
  // Mirrors the user-reported bug: two models share one backend port (:18095);
  // qwen-q4km is the one loaded EXTERNALLY there. After the fix the dashboard
  // must show qwen-q4km as active + external and qwen-q4kxl as stopped.
  const C = makeData(true);
  C.instances = {
    default: { port: 18795, path: "/cfg/adopt.yaml", running: true },
    instances: [{
      port: 18795, base_url: "http://127.0.0.1:18795", reached: true, pid: 777,
      sources: ["pidfile"], active: "qwen-q4km",
      models: [
        { name: "qwen-q4km", kind: "chat", external: true, port: 18095, pid: null, active: true, running: true,
          llm: { pp_current: null, tg_current: null, pp_avg: null, tg_avg: null } },
        { name: "qwen-q4kxl", kind: "chat", external: false, port: 18095, pid: null, active: false, running: false,
          llm: { pp_current: null, tg_current: null, pp_avg: null, tg_avg: null } }
      ]
    }],
    suggested_port: 18796, version: "0.1"
  };
  const domC = buildDom(C);
  const dC = domC.window.document;
  await sleep(500);
  const rowsC = Array.from(dC.querySelectorAll("#instances table tr"));
  const rowKm = rowsC.find(tr => tr.textContent.includes("qwen-q4km"));
  const rowKxl = rowsC.find(tr => tr.textContent.includes("qwen-q4kxl"));
  check("adopt: qwen-q4km row rendered", !!rowKm);
  check("adopt: qwen-q4km shows 'active' badge", !!rowKm && !!rowKm.querySelector(".badge.active"));
  check("adopt: qwen-q4km shows 'external' hint", !!rowKm && /external/.test(rowKm.textContent));
  check("adopt: qwen-q4kxl row rendered", !!rowKxl);
  check("adopt: qwen-q4kxl shows 'stopped' badge", !!rowKxl && !!rowKxl.querySelector(".badge.stopped"));
  check("adopt: qwen-q4kxl has NO 'active' badge", !!rowKxl && !rowKxl.querySelector(".badge.active"));
  check("adopt: qwen-q4kxl has NO 'external' hint", !!rowKxl && !/external/.test(rowKxl.textContent));
  check("adopt: instance active-model badge shows qwen-q4km", /active: qwen-q4km/.test(dC.querySelector("#instances").textContent));
  domC.window.close();

  /* ---------- Scenario D: pop-up dismisses when an instance starts ---------- */
  console.log("Scenario D: pop-up dismisses (slides up) when an instance starts");
  const D = makeData(false);           // starts with nothing running
  const domD = buildDom(D);
  const dD = domD.window.document;
  await sleep(500);
  check("dismiss: pop-up shown initially (no instances)", dD.querySelector("#noInstPop").classList.contains("on"));
  // simulate an instance coming up: the next /api/instances poll now returns one
  D.instances.default.running = true;
  D.instances.instances = [
    { port: 18590, base_url: "http://127.0.0.1:18590", reached: true, pid: 12345, sources: ["pidfile"], active: "qwen-local",
      models: [ { name: "qwen-local", kind: "chat", external: false, port: 18501, pid: 111, active: true, running: true,
        llm: { pp_current: null, tg_current: null, pp_avg: null, tg_avg: null } } ] }
  ];
  await domD.window.eval("refreshOverview()");
  await sleep(400);   // let the 300ms fade-out timer finish so the pop-up leaves the flow
  check("dismiss: pop-up faded out (no longer .shown) after instance start", !dD.querySelector("#noInstPop").classList.contains("shown"));
  check("dismiss: pop-up out of the flow (no longer .on) after fade", !dD.querySelector("#noInstPop").classList.contains("on"));
  check("dismiss: running banner shown after instance start", (dD.querySelector("#banner").className || "").includes("run"));
  domD.window.close();

  console.log(`\nRESULT: ${passes} passed, ${failures} failed`);
  process.exit(failures ? 1 : 0);
})().catch((e) => { console.error("test crashed:", e); process.exit(1); });
