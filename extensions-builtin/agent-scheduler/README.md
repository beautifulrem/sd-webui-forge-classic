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
them, then remove the flag; failed ones can then be requeued.
