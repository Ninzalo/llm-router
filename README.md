# llm-router

An **OpenAI-compatible router + lifecycle manager for local LLM backends**, built to
plug into [hermes-agent](https://hermes-agent.nousresearch.com) as a `custom` endpoint.

You point hermes at the router once. The router:

- serves a **faithful OpenAI-compatible API** — the whole `/v1/*` surface
  (`/v1/chat/completions`, `/v1/embeddings`, …), with streaming SSE passthrough, not a chat-only stub
- **lists your preconfigured chat models** to hermes's `/model` picker (`GET /v1/models`)
- **advertises each model's `context_length` and default `reasoning_effort`** to hermes
- **starts a default model on boot** — in the background, so the router is reachable while a big model loads
- **switches the active model on demand** (preconfigured only; unknown → 400)
- **stops the old model completely *before* starting the new one** — one big model in RAM at a time
- **forwards each request** to the correct local LLM and **returns its answer**
- **stops a model even while it is still loading** when you stop the router — no orphaned backends

It installs as a single **`llm-router` executable** on your `PATH`, so you can start,
configure, and inspect it from any directory — no `cd`, no venv activation. The
top-level command and **every subcommand** print a full **markdown `--help`**.

```
hermes-agent ──(OpenAI)──▶  llm-router :8787/v1  ──▶  active local LLM (e.g. splash :8080)
   /model picker ◀─ GET /v1/models          stops the old model *first*,
                                            starts the requested model,
                                            rewrites the `model` field to
                                            what the backend expects,
                                            streams the answer back
```

## Quick Setup

The fastest path from a clean machine to a working hermes endpoint.

### 1. Install (one line)

```bash
curl -fsSL https://raw.githubusercontent.com/Ninzalo/llm-router/main/install.sh | bash
```

This clones the repo to `~/llm-router`, creates a virtualenv, installs the `llm-router`
command, links it into `~/.local/bin`, and drops a starter `config.yaml`. Requirements:
`bash`, `git`, and Python 3.10+ (uses `uv` if it's on your PATH). It's idempotent —
re-run it any time to update.

<details>
<summary>Prefer to do it by hand?</summary>

```bash
git clone git@github.com:Ninzalo/llm-router.git ~/llm-router && cd ~/llm-router
python3 -m venv .venv
./.venv/bin/pip install -e .
ln -sf "$HOME/llm-router/.venv/bin/llm-router" "$HOME/.local/bin/llm-router"
cp config.example.yaml config.yaml        # then edit it
```

</details>

### 2. Configure your models

```bash
llm-router config        # opens config.yaml in your $EDITOR
llm-router check         # validate — nothing is started
```

At minimum, give `router.default_model` the `name` of a model in the `models:` list, and
give each model a `command` that starts an OpenAI-compatible backend.

### 3. Run it

```bash
llm-router run           # background daemon; the default model loads in the background
llm-router status        # confirm the instance + active model
```

The public OpenAI-compatible base is `http://127.0.0.1:8787/v1`.

### 4. Point hermes at it

Add a `custom` provider to your hermes config that targets the router (see the
[Wiring it into hermes-agent](#wiring-it-into-hermes-agent-the-custom-endpoint) section):

```yaml
model_aliases:
  my-model:
    model: my-model               # must match a `name` in the router config
    provider: custom
    base_url: "http://127.0.0.1:8787/v1"
```

### 5. Dashboard (optional)

```bash
llm-router dashboard     # opens http://127.0.0.1:8799 in your browser
```

> **Want to try it without real models?** `bash run_test.sh` boots the router against
> fake backends on dedicated `18xxx` ports (never touching your real servers), runs the
> full suite, and shuts down. See [Quick start (isolated, no real models)](#quick-start-isolated-no-real-models).

## Feature set

| Area | What it does |
|------|--------------|
| **OpenAI proxy** | Faithful proxy for the whole `/v1/*` API (chat, embeddings, …). Streaming SSE passthrough. The router rewrites the request `model` field to what the local backend expects and passes the answer (JSON or SSE) back unchanged. |
| **Model picker** | `GET /v1/models` lists your preconfigured **chat** models — the source of hermes's `/model` picker. `embedding`-kind models are hidden. |
| **context_length** | Each model advertises `context_length` (and `max_context_length`) on `/v1/models`; hermes reads these for custom endpoints. Booting a `chat` model below hermes's 64,000 minimum logs a warning. |
| **reasoning_effort** | Per-model default `reasoning_effort` is advertised on `/v1/models`, injected into forwarded requests when the client omits it, and passed through unchanged when the client provides it. |
| **kind** | `chat` (default) models are picker-visible; `embedding` models are fully router-managed (start/stop, `/v1/embeddings`) but hidden from hermes so they can't be mis-selected as the active chat model. |
| **Memory-safe switching** | The active model is stopped **completely** (whole process group SIGTERM'd, then awaited; SIGKILL after `shutdown_grace`) *before* the new one is spawned — the two never share RAM. |
| **HF-style names** | Name a model exactly as its Hugging Face id, e.g. `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`. The name is used verbatim as the `/v1/models` id and for routing; only the per-model log filename is flattened by `safe_filename()`. |
| **Daemon lifecycle** | `run` (background + log file), `serve` (foreground), `stop` (all instances + their models **and any `watch`/`watch --llm` tailers**), `stop-model`, `switch`, `status`, `models`, `check`, `config`. |
| **Robust stop** | `stop` stops the model even when it is still loading — a model in its startup window is a real process and is always stopped on shutdown (regression-tested). |
| **Quiet logging** | No per-request `httpx`/`httpcore` chatter, no duplicate startup lines. Per-model logs + a router log that rotates on each `run`. |
| **Watch** | `watch` tails the router log; `watch --llm` tails the **active model's own** log and **auto-switches** to the new model's log when the active model changes. Both require a **running** router (else `cannot reach a router …`). Ctrl+C stops the watcher, never the router. |
| **Web dashboard** | `llm-router dashboard` serves a self-contained web UI: running instances (each with a one-click **Stop** button) + their LLMs (active labelled), current + average pg/tg rates per instance and per LLM, a structured **and** raw-YAML config editor, live log watching with a **Watch / Stop-watching** toggle, and a dark/light theme. Opens your browser automatically (`--no-browser` to skip); Ctrl+C stops it immediately. It is its own process and **never auto-starts a router** (one-click button to start the default). |
| **Help** | `llm-router --help` and `llm-router <cmd> --help` print a full markdown description + arguments for the command. |

## The CLI

`llm-router` (no args) or `llm-router --help` prints the full command reference.
Every subcommand also takes `--help` / `-h` for its own markdown page.

```
llm-router run        [--config P] [--host H] [--port N] [--preload|--no-preload]   # BACKGROUND daemon (returns at once; logs to file)
llm-router serve      [--config P] [--host H] [--port N] [--preload|--no-preload]   # FOREGROUND (what `run` daemonizes; used by launchd)
llm-router watch      [--port N] [--lines N] [--llm]          # follow the router log, OR (--llm) the active model's own log
llm-router stop       [--port N]                              # stop ALL router instances + their active model(s) + any watch tailers
llm-router stop-model <model> [--config P] ...                # stop one model via a running router (router keeps running)
llm-router switch     <model>   [--config P] ...              # activate a preconfigured model (memory-safe)
llm-router status     [--port N] [--base-url U] [--json]       # status of ALL running instances (or one)
llm-router models     [--config P]                            # list preconfigured models (name, kind, target, context, reasoning)
llm-router check      [--config P]                            # validate + print the resolved specs (no server)
llm-router config     [--path P] [--print]                    # open the config in $EDITOR (or --print the path)
llm-router dashboard  [--host H] [--port N] [-c P] [-d] [--no-browser]   # web dashboard (opens your browser; -d = background)
llm-router dashboard stop                                                  # stop a running dashboard
llm-router --version                                                       # print the version
```

Examples:

```bash
# start the router as a background daemon on 127.0.0.1:8787, from anywhere:
llm-router run --host 127.0.0.1 --port 8787
#   -> returns immediately; logs to ~/.config/llm-router/router-<port>.log
#   -> the default model loads in the background while the router is already reachable

# follow the ROUTER log (Ctrl+C stops the watcher, NOT the router):
llm-router watch --port 8787

# follow the ACTIVE MODEL's OWN log (auto-switches when the model changes):
llm-router watch --llm --port 8787

# stop it (all instances + the active model + any watch tailers — even if it's still loading):
llm-router stop

# show the status of every running instance (port, pid, active model, per-model state):
llm-router status
#   just one instance, or machine-readable JSON:
llm-router status --port 8787
llm-router status --json

# start from an explicit config file:
llm-router run --config /path/to/config

# run in the foreground instead (Ctrl+C stops the router + its model):
llm-router serve --host 127.0.0.1 --port 8787

# open the (default) config in your editor / just print its path:
llm-router config
llm-router config --print

# full markdown help for any command:
llm-router --help
llm-router watch --help
llm-router stop --help

# open the web dashboard — it launches your default browser (new window) automatically:
llm-router dashboard                 # http://127.0.0.1:8799 (foreground; Ctrl+C stops it immediately)
llm-router dashboard --no-browser    # don't open the browser (just print the URL)
llm-router dashboard -d              # background (logs to dashboard.log); still opens the browser
llm-router dashboard stop            # stop it
```

### Config resolution (so it works from any cwd)

`--config`/`--path` > `$LLM_ROUTER_CONFIG` > `~/.config/llm-router/config.yaml`
(if present) > `<repo>/config.yaml` (if present) > `~/.config/llm-router/config.yaml`.

Relative paths inside the config (e.g. `log_dir: ./logs`) and any relative `command`
resolve against **the config file's directory**, not the shell's cwd. `llm-router config`
creates a starter file (and opens it) if the resolved config doesn't exist yet.

Router **state** (pid files + the router log) lives in `~/.config/llm-router/`
(override with `$LLM_ROUTER_STATE_DIR`), so `stop` / `watch` work regardless of cwd or
which config you used. On each `run`, the previous run's router log is rotated to
`router-<port>.log.1`, so every boot begins a fresh, readable log.

## Install (one time)

Easiest — the [Quick Setup](#quick-setup) one-liner installs everything for you:

```bash
curl -fsSL https://raw.githubusercontent.com/Ninzalo/llm-router/main/install.sh | bash
```

Or, if you already have the repo and your own venv, install it into that venv:

```bash
cd ~/llm-router
~/.local/bin/uv pip install -e . --python ./.venv/bin/python     # or: ./.venv/bin/pip install -e .
ln -sf "$HOME/llm-router/.venv/bin/llm-router" "$HOME/.local/bin/llm-router"
llm-router --version
```

That's it — `llm-router` is now a global command that runs on the project's own venv
(the editable install's console script already points at `.venv/bin/python`).

## Quick start (isolated, no real models)

Uses `mock_llm.py` (a fake OpenAI backend) on dedicated ports (18xxx) so it never
touches your real splash/llama servers:

```bash
bash run_test.sh                 # boots the test router, runs all 27 checks, shuts it down
./test_stop_startup.sh           # regression: `stop` while the model is mid-startup
./test_stop_watchers.sh          # regression: `stop` stops running `watch`/`watch --llm` tailers
```

Or by hand: `llm-router run --config config.test.yaml` and then point an OpenAI client at
`http://127.0.0.1:18787/v1`.

## Running for real

```bash
llm-router check                 # validate the config first (nothing starts)
llm-router run --config ~/llm-router/config.yaml
# public base:  http://127.0.0.1:8787/v1
```

## Config reference

```yaml
router:
  host: 127.0.0.1        # where the router listens
  port: 8787             # where the router listens
  api_key: ""            # if set, require "Authorization: Bearer <key>" on data paths
  default_model: unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M   # the default model (else first model)
  preload: true            # start `default_model` in the background on `run`/`serve`
                           #   (override per-run with --preload / --no-preload; off = lazy, first request loads it)
  startup_timeout: 600   # seconds to wait for a model to load
  health_interval: 1.0   # seconds between readiness probes
  health_path: /health   # primary readiness probe path (also tries /v1/models)
  log_dir: ./logs        # per-model stdout+stderr logs
  keep_inactive_stopped: true   # the core behavior: only the active model runs
  shutdown_grace: 10     # seconds after SIGTERM before SIGKILL

dashboard:               # defaults for the web dashboard (`llm-router dashboard`)
  host: 127.0.0.1        # where the dashboard binds (127.0.0.1 = this machine only)
  port: 8799             # where the dashboard listens
  browser: true          # auto-open the browser on start (a new window)

models:
  - name: unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M          # hermes-facing id (what you type in /model)
    upstream_model: unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M  # optional; sent to the backend. default = name
    host: 127.0.0.1                   # optional CONNECT host (defaults to --host in the command, else 127.0.0.1)
    port: 8080                        # optional; defaults to --port parsed from the command
    context_length: 131072            # advertised to hermes via /v1/models (match your backend's ctx)
    reasoning_effort: medium          # optional; default reasoning effort (injected if the client omits it)
    kind: chat                        # optional: chat (default, shown in /v1/models) | embedding (managed, hidden)
    command: splash serve --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --host 0.0.0.0 --port 8080 --max-context 131072 --idle-release 1800
    # extra per-model knobs (optional):
    # startup_timeout: 900
    # health_path: /health
    # env: {SOME_VAR: "x"}
```

Per-model fields:

- **`name`** — the id hermes sees: what you type in `/model` and what `GET /v1/models`
  returns. It may be **any string**, including a Hugging Face id with `/` and `:`
  (e.g. `unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`). It is used verbatim for routing and in
  the API; only the per-model **log file** is flattened (`unsloth-Qwen3.8-27B-GGUF-UD-Q4_K_M.log`).
- **`command`** — the full startup command *with arguments*. Port/host are parsed from
  `--port`/`--host` unless you set them explicitly.
- **`upstream_model`** — the model id placed in the forwarded request body (what the local
  server actually expects). Defaults to `name`. This is how hermes can send one id while the
  backend receives a different one.
- **`context_length`** — the model's context window, advertised to hermes in the `/v1/models`
  object (hermes reads `context_length` first, then `context_window`/`context_size`/
  `max_context_length`). Set it to your backend's `--max-context` / `--ctx-size`.
- **`reasoning_effort`** — the model's default reasoning effort
  (`none/minimal/low/medium/high/xhigh/max`). Advertised on `GET /v1/models`, injected into
  forwarded requests when the client omits `reasoning_effort`, and passed through unchanged
  when it's provided — so hermes (or any OpenAI client) can change the effort per-request.
- **`kind`** — `chat` (default) models are listed on `GET /v1/models` and appear in hermes's
  `/model` picker. `embedding` models are still fully router-managed (start/stop,
  `/v1/embeddings`) but **hidden from the hermes picker** — so an embeddings-only backend
  can't be mis-selected as the active chat model and trip hermes's 64K minimum-context check.
- **`host` / `port`** — the address the router *connects to*. By default these are parsed
  from the model's `command` (`--host`/`--port`); set `host` explicitly when the backend
  binds to a different address (e.g. splash binds `0.0.0.0` but you connect via
  `127.0.0.1`). They're valid in the YAML file (used for adopted/external models), but the
  dashboard's Config editor deliberately doesn't expose them — they live in the `command`.
- **`startup_timeout`, `health_path`, `env`** — per-model overrides.

The optional **`dashboard:`** section holds the web dashboard's defaults (`host`, `port`,
`browser`) — what `llm-router dashboard` binds to and whether it auto-opens the browser.
It is edited the same way as the rest of the config: in the dashboard's **Config →
Dashboard** section (structured) or in Raw YAML. CLI flags on `llm-router dashboard`
(`--host`/`--port`/`--no-browser`) override these for a single run; a changed `port`
takes effect the next time the dashboard starts. If the section is absent, the built-in
defaults (`127.0.0.1:8799`, browser on) apply.

## Wiring it into hermes-agent (the custom endpoint)

The router is a standard OpenAI-compatible endpoint, so hermes consumes it through its
`custom` provider. Register it once under `custom_providers` and point hermes at it:

```yaml
model:
  default: unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M   # must match a router model `name`
  provider: custom
  base_url: http://127.0.0.1:8787/v1

custom_providers:
  - name: llm-router
    base_url: http://127.0.0.1:8787/v1
    model: unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
    models_discovered: true        # hermes pulls the model list from the router's GET /v1/models
```

- **`models_discovered: true`** is the cleanest: hermes queries the router's
  `GET /v1/models` to populate its picker, so you never have to list the models by hand.
  Because the router only lists `chat` models, hermes's picker can only ever offer chat
  models — the 64K minimum-context rejection can't happen from an embeddings backend.
- If you prefer to pin them explicitly, list each under `models:` instead of using
  `models_discovered`:
  ```yaml
  custom_providers:
    - name: llm-router
      base_url: http://127.0.0.1:8787/v1
      model: unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M
      models:
        unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M: {}
        unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL: {}
  ```
- **You don't set a context length or reasoning effort in hermes** — hermes reads both from
  the router's `/v1/models` response for custom endpoints, which is where the router
  advertises them.
- If you set `router.api_key`, mirror it as the provider's `api_key` in hermes.

Then in hermes: `/model` to pick, or `/model <name>` to switch. Every subsequent request
carries that model, and the router keeps exactly that model running.

## Web dashboard

A self-contained web UI for managing the router — instances, their LLMs, live rates,
config, and logs. It is its **own process** and **never auto-starts a router**; you start
instances from the UI (a one-click **"Run default instance"** button, or the Run Instance tab).

### Running it

```bash
llm-router dashboard                 # foreground (Ctrl+C stops it immediately); host/port from the config's dashboard: section (default 127.0.0.1:8799)
llm-router dashboard --port 9000 --host 0.0.0.0
llm-router dashboard -c /path/config.yaml
llm-router dashboard --no-browser    # print the URL but don't open a browser window
llm-router dashboard -d              # detached (background); logs to dashboard.log
llm-router dashboard stop            # stop it (the -d one and any foreground one)
```

| flag                | description |
|---|---|
| `--host H`          | dashboard bind host (overrides `dashboard.host`; default `127.0.0.1`) |
| `--port N`          | dashboard port (overrides `dashboard.port`; default `8799`) |
| `-c, --config P`    | config file the dashboard edits/starts from (default: auto-resolved) |
| `-d, --detached`    | run in the background (log to `dashboard.log`); default is foreground |
| `--no-browser`      | don't auto-open a browser window this run (overrides `dashboard.browser`; just print the URL) |
| `stop`              | stop a running dashboard |

By default `llm-router dashboard` **opens the dashboard in your default browser** (a new
window) as soon as it comes up — unless `dashboard.browser` is set to `false` in the config
or you pass `--no-browser`. Where it binds (`host`/`port`) comes from the config's
`dashboard:` section, with the CLI flags overriding for a single run. The dashboard discovers running router instances the same
way `status` does (pidfiles + a process scan), so it works from any cwd. In the foreground,
**Ctrl+C stops it immediately** (it drops in-flight requests and the live log stream rather
than waiting for the browser page to disconnect); `llm-router dashboard stop` does the same
for a detached run.

### What's in the UI

- **Overview** — every running instance with its configured LLMs; the **active** model is
  labelled. Shows **current** and **average** prompt (pg) / generation (tg) rate **per
  instance** (i.e. its active model — an instance-level figure, not a per-LLM measurement)
  **and per LLM** (each model's own, read from its log). A red **Stop** button stops that
  running instance (SIGTERM, then SIGKILL). A **"Run default instance"** button starts the
  default config's instance with one click if it isn't already running.
- **Run Instance** — start a **new** instance on a fresh (auto-suggested) port, reusing LLMs
  already in the default config: tick which LLMs to include and pick the default/active one.
  Preload is optional.
- **Config** — a **structured form** for every router field (host, port, api_key,
  **default_model** — a **dropdown** of the configured models — startup/health/shutdown
  timeouts, log_dir, preload, keep_inactive_stopped) and every model field (name, command,
  upstream_model, startup_timeout, health_path, context_length, reasoning_effort, kind, env),
  with **Add model** / remove. A model's host/port are **not** editable here — they come from
  the model's `command` (`--host`/`--port`). Plus a **Raw YAML** mode for editing the whole
  file as text. Saves are **validated before writing** and a `.bak` backup of the previous
  config is kept.
- **Logs** — list every running instance's router log and every running LLM's own log.
  **Watch** streams any of them live (the LLM log is where the pg/tg timing lines come from);
  once started, the button turns into a red **Stop watching** — click it again to stop.
  Stopping detaches the stream without killing anything.
- **Dark/light theme** (persisted; follows the OS by default). The less-obvious controls carry
  a **`?` tooltip** — hover for a one-line description (e.g. the health-interval / health-path
  / shutdown-grace / preload knobs, per-model startup/health/env). The self-explanatory ones
  (the theme toggle, the rate metrics, and most run/config controls) don't.

Rates are parsed from llama.cpp-style timing lines (`prompt eval time …`, `eval time = …`)
in each model's log file, so they reflect what the backends are actually doing.

## Admin / lifecycle API

| Method | Path | Effect |
|--------|------|--------|
| GET | `/v1/models` | chat models only (embedding models hidden) + `context_length` + `reasoning_effort` |
| GET | `/health` | router health + active model + `ready` + managed/external counts |
| GET | `/admin/status` | active + per-model state (running/managed/external/active/pid/context_length) |
| GET | `/admin/active` | lightweight `{active, ready}` probe (used by `watch --llm`; kept out of the access log) |
| GET | `/admin/models` | active + per-model detail (upstream, base_url, port, context_length) |
| POST | `/admin/switch` | body `{"model":"<name>"}` — activate it (stop old first, then start it) |
| POST | `/admin/models/{name}/start` | activate a model |
| POST | `/admin/models/{name}/stop` | stop one model |
| any | `/v1/...`, `/...` | transparent proxy to the active model |

The same operations are available as CLI subcommands (`llm-router status/switch/stop` /
`stop-model`), which talk to a running router over this admin API.

```bash
llm-router status
llm-router switch unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
curl http://127.0.0.1:8787/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model":"unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M","messages":[{"role":"user","content":"hi"}]}'
```

## Owned vs adopted (important for memory-safe stopping)

The router can only **stop** a model it **started**. This keeps it from ever killing a
model it didn't launch:

- **Owned** — the router launched it (via `command`). It is stopped when it becomes
  inactive and on router shutdown. This is the full-lifecycle path. **To use it, let the
  router be the thing that starts your models** (stop any pre-running splash/llama first,
  then start the router).
- **Adopted (external)** — a model already listening on its port when the router boots.
  The router routes to it but **never kills it** (safe if your models are already up).

So: if you want "stop the inactive one" to actually stop it, make sure the router started
it. If a model was already running, it's adopted and left alone.

## Operations

- **Daemon start**: `llm-router run` detaches the router (its own process group), returns
  immediately, and writes the router log to `~/.config/llm-router/router-<port>.log`. By default
  the default model is preloaded **in the background**, so the router is reachable while a big
  model loads; `/health` reports `ready: true` once it's active. Control this with `router.preload`
  (config) or the `--preload` / `--no-preload` flags — the flags win, and with neither present
  `router.preload` decides (default `true`). With preload off, nothing loads on boot and the
  **first request** (or `llm-router switch`) starts the model lazily. Use `llm-router serve` for a
  foreground run.
- **Watch**: `llm-router watch [--port N] [--lines N]` tails the router log.
  `llm-router watch --llm [--host H] [--port N] [--lines N]` tails the **active model's own**
  log (`<log_dir>/<safe_filename>.log`) and **auto-switches** to the new model's log whenever the
  active model changes (via `llm-router switch` or hermes changing models). **Ctrl+C stops the
  watcher only** — the router keeps running either way. Both modes require a **running** router
  on the target port: with none reachable, `watch` reports `cannot reach a router …` (a leftover
  log file from a stopped run is not tailed).
- **Status**: `llm-router status` reports every **running** instance (discovered from the
  `router-<port>.pid` pidfiles plus any stray `serve`/`run` processes): its port, pid, how it was
  started, the active model (and whether it is ready), and each model's running/managed/external
  state. Add `--port N` (or `--base-url U`) for just one instance, or `--json` for machine output.
- **Stop**: `llm-router stop [--port N]` stops every router instance (SIGTERM). Each instance's
  **graceful shutdown stops the model it started** — including a model that is still loading
  (a model in its startup window is a real process and is always stopped, never orphaned).
  `llm-router stop-model <name>` stops one model while the router keeps running.
- **Logs**: per-model `stdout+stderr` in `log_dir` (`<safe_filename>.log`); router log as above.
  On each `run`, the previous router log is rotated to `router-<port>.log.1`. Lifecycle logging
  is quiet by design: per-request `httpx`/`httpcore` chatter and the `ready at` / `active model is
  now` lines are suppressed (WARNING/debug) so the log only shows meaningful events.
- **Readiness**: after (re)starting a model the router polls `health_path` (then `/v1/models`)
  until OK or `startup_timeout`; a request during a switch waits for the model to be ready. Big
  models can take minutes to load, so the *first* request after a switch may be slow.
- **Crash recovery**: if the active model dies, the next request re-detects it and re-starts it.

## Auto-start the router on login (optional)

Use the included `llm-router.plist` (launchd) or a systemd unit / cron `@reboot`.
The plist uses `llm-router run` (foreground) so launchd can supervise and restart the
router if it crashes. Adjust the paths and the `--config` argument.

## Security

Bind to `127.0.0.1` (default). If you expose it beyond localhost, set `router.api_key`
and mirror it in the hermes provider's `api_key`.

## Layout

```
llm-router/
  pyproject.toml         # packaging: installs the `llm-router` console command
  package.json           # jsdom devDependency + `test:ui` script (for the Node dashboard UI test)
  install.sh             # one-line installer: curl ... | bash  (venv + editable install + PATH + starter config)
  config.example.yaml    # documented starter config (copy to config.yaml and edit)
  llm_router/            # the package
    config.py            # config load + validation + port auto-parse + context_length + safe_filename()
    manager.py           # process lifecycle (start/stop/ensure_active/shutdown)
    app.py               # FastAPI: OpenAI surface + admin API + proxy
    dashboard.py         # web dashboard backend (FastAPI): discovery, rate parsing, config editor, log tail
    static/
      dashboard.html     # the dashboard frontend (self-contained: dark/light, 4 tabs, tooltips, pop-up, 5s auto-refresh)
    __main__.py          # CLI: run/serve/watch/stop/stop-model/switch/status/models/check/config/dashboard + markdown --help
  mock_llm.py            # fake OpenAI backend for isolated testing (supports --delay, logs pg/tg timing at INFO)
  config.yaml            # REAL config (your splash/llama) — NOT committed; see config.example.yaml
  config.test.yaml       # TEST config (mocks, isolated ports 18xxx)
  config.adopt.yaml      # TEST config for the external-detection suite (mock, isolated port)
  config.dash.yaml       # DASHBOARD TEST config (mocks, isolated ports 185xx)
  test_suite.py          # end-to-end assertions (27 checks)
  test_external_detect.py# test: router reads a model's real id from its /v1/models (20 checks)
  dash_e2e.py            # dashboard backend end-to-end driver (isolated ports)
  dash_ctrlc.py          # dashboard live Ctrl+C / SSE-shutdown driver
  ui_test.js             # dashboard frontend jsdom test (110 checks; `npm install` then `node ui_test.js`)
  test_stop_startup.sh   # regression: `stop` must stop a model that is still loading
  test_stop_watchers.sh  # regression: `stop` stops running `watch`/`watch --llm` tailers
  run_test.sh            # boot the test router + run the suite, isolated from real servers
  llm-router.plist       # optional launchd auto-start template (not committed — has machine paths)
  .gitignore             # ignores .venv/, __pycache__/, *.egg-info/, logs/, real config.yaml, etc.
  .venv/                 # isolated venv (uv, Python 3.11) — not committed
```

## How this was validated

`test_suite.py` runs against `config.test.yaml` — mock backends on **18080/18081/18082/18083**,
router on **18787** — **isolated from the real `:8080`/`:8081` servers**, which are only probed
read-only and never started or stopped by the tests. It asserts **27 checks**, including:

- per-model `context_length` + `reasoning_effort` are advertised on `GET /v1/models`
  (chat models only) and `/admin/status` (all models).
- **embedding models are hidden from `GET /v1/models`** (hermes can't pick one) but are
  still switchable/routable by the router.
- **`reasoning_effort`**: the model default is advertised, injected when the client omits
  it, and passed through when the client provides it.
- **memory-safe switching**: the daemon log proves `stopping model '<old>'` (which awaits the
  process exit) *precedes* `starting model '<new>'` — the old model's RAM is freed before the
  new one is spawned.
- **HF-style names** (`org/repo:tag`) are routable verbatim, and the per-model log is written
  to a **flat** filename (no nested `unsloth/` directory).
- streaming (SSE passthrough) and the base-url-without-`/v1` path normalization.
- graceful shutdown (via `llm-router stop`) stops the active model.

`test_stop_startup.sh` is a standalone regression for the "stop while loading" case: it starts a
router whose default model has a long load (`mock_llm.py --delay`), calls `llm-router stop` while
the model is still in its startup window, and asserts the model process is gone. (This was
verified by A/B: with the old shutdown path — `except Exception`, which does not catch
`asyncio.CancelledError`, a `BaseException` — the mid-startup model was orphaned; with the fix
— catching `CancelledError` so `mgr.shutdown()` always runs — it is always stopped.)

`test_stop_watchers.sh` is a standalone regression for watcher cleanup: it boots a router, starts
a `watch` and a `watch --llm` in the background (other-terminal style), calls `llm-router stop`, and
asserts **both** watcher processes are gone — not just the router and its model.

`run_test.sh` boots the test router, runs the 27-check suite, and shuts it down
(`RESULT: PASS (27/27)`). The `llm-router` executable runs **from any working directory**:
`run` returns immediately as a background daemon, `watch` / `watch --llm` follow the right log
(Ctrl+C stops the watcher only), and `stop` stops the router + its model + any watch tailers.

The **web dashboard** is validated three ways, all isolated on 18xxx/185xx ports (never the real
`8080`/`8081` servers). All three drivers ship in the repo root and are path-portable (they
resolve `dashboard.html` and `config.dash.yaml` relative to themselves):

- a **backend end-to-end driver** (`dash_e2e.py`) boots mock backends + a router + the
  dashboard and asserts, over HTTP: instance discovery, the active-LLM label, **current and
  average pg/tg rates per instance and per LLM** (populated from the mock backends' timing
  lines), run-default, **run-instance** (a new instance on a fresh port reusing a subset of
  the config's LLMs), **stop-instance** (`POST /api/instances/stop` — stops only the target
  port; the other instance stays up; 400 for a bad port, 404 for none running), structured
  config save (+ `.bak`), **invalid-config rejection**, raw-YAML save, and live log streaming.
- a **jsdom frontend test** (`ui_test.js`) loads the real `dashboard.html`, stubs the fetch
  APIs, and exercises the UI end-to-end: theme init + toggle, the reduced **and** added
  tooltips (asserts specific removed ones are gone and specific added ones are present), all
  four tabs, overview rendering (instance card, active label, per-LLM + per-instance rate
  cells), the **Stop-instance** button (and its POST to `/api/instances/stop`), the
  run-default button, the run-instance form, the structured **and** raw config editors
  (**default model as a dropdown; model host/port not editable**), add-model, save (correct
  PUT body), the **Watch / Stop-watching** log toggle, **5-second auto-refresh** (polls every
  5s, never 2s), the **"No default instance" pop-up** (shown only when nothing is running,
  dismissed + slides up after any instance starts), and the **"Opened with config:"** header —
  **110 checks**. (Run it with `npm install && node ui_test.js` — jsdom is the only dependency.)
- a **live Ctrl+C test** (`dash_ctrlc.py`) starts a router + the dashboard, holds the
  dashboard's `/api/logs/stream` (the Logs watch) open and polls, then sends SIGINT and
  asserts the dashboard exits within a second — proving Ctrl+C stops it immediately even
  while the browser page is streaming (uvicorn's graceful-shutdown timeout is `0`, so it
  drops the in-flight requests and the SSE stream instead of waiting).
