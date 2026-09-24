# Agent Scheduler Neo

This Fork was created to fix the Agent Scheduler Extension to work with gradio 4.x so it can be used again in Forge Neo
I fixed it by using Claude.

Big Thanks to https://github.com/supersonic13 for its initial work!

## Compatibility

This version of AgentScheduler is compatible with latest versions of Forge Neo:

- Forge Neo: https://github.com/Haoming02/sd-webui-forge-classic/releases/tag/2.15

NOTE: If it doesn't work immediately. Try pressing F5 before first generation. sometimes there is a timing issue within the UI.

## Installation

### Using the built-in extension list

1. Open the Extensions tab
2. Open the "Install From URL" sub-tab
3. Paste the repo url: https://github.com/cataclisma/sd-webui-agent-scheduler-neo.git
4. Click "Install"

### Manual clone

```bash
git clone https://github.com/cataclisma/sd-webui-agent-scheduler-neo.git" extensions/agent-scheduler
```

(The second argument specifies the name of the folder, you can choose whatever you like).

## Stored tasks and queue import

Task script params are stored as pickles, and loading a pickle can run code, so
this build signs them with a per-install key (`agent_scheduler_signing.key`,
next to the task database). Queue exports carry the signature; `/import`
accepts only tasks exported from the same install.

Tasks saved by older versions have no signature and will not run (they fail
with a message saying so). If nobody else could have imported tasks into your
database, start once with `--agent-scheduler-trust-unsigned-params` to sign
them, then remove the flag; tasks that failed for being unsigned can then be
requeued from the History tab.

Keep `agent_scheduler_signing.key` together with the task database when you
move the install or a Docker volume. If the key file cannot be read (e.g. it is
locked at startup), the queue is paused instead of failing stored tasks; fix
the file and restart. The task database and key are never served through
Gradio's `/file=` route.

## Access control

- With `--api-auth`, the API uses HTTP Basic auth.
- Without it, but with `--gradio-auth`, every `/agent-scheduler/v1/*` route
  requires the Gradio login (the UI's own requests already carry it).
- State-changing requests from another site are refused. Browser front-ends on
  another origin must be allowed with `--cors-allow-origins`.
- `callback_url` must be http(s); loopback and LAN targets are allowed, cloud
  metadata / link-local targets are not, and redirects are not followed.
- Image URLs in API tasks follow Forge's *Settings → API* request policy.
- A task runs at most once at a time, even across a UI reload or a manual
  **Run** while the queue is running.
