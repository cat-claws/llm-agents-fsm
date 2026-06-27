# llm-agents-fsm

Research/prototype agents that combine LLM tool use with explicit planning
traces and finite-state property checks. The repository currently contains two
agent domains:

- `git-system/`: terminal Git agents with basic and FSM variants.
- `shrdlu-block/`: OpenAI-compatible agents for a standalone SHRDLU
  blocks-world simulator.
- `utils/`: shared session serialization, planning-tree helpers, and TLA+/TLC
  verification utilities.

The Git scripts can be run directly from source. The supported launch command is
`run-agents`, installed by the editable setup below. The editable install also
maps the `shrdlu_agents` import name to the `shrdlu-block/` source directory.

## Requirements

- Python 3.10 or newer.
- `openai` for all LLM-backed agents.
- Git for the `git-system` agents.
- Optional: Java plus `tla2tools.jar` for TLC verification. Set
  `TLA2TOOLS_JAR` to the jar path. If it is unavailable, the TLC runner reports
  verification as skipped.
- For SHRDLU agents: an already-running `shrdlu_blocks` simulator. If the
  simulator package is not installed, keep its checkout on `PYTHONPATH`.

Basic local setup:

```bash
cd /home/robot/llm-agents-fsm
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install openai
python3 -m pip install -e . --no-deps
```

Drop `--no-deps` if you want pip to resolve and install dependencies itself.
No local `shrdlu_agents` symlink is needed; the mapping lives in
`pyproject.toml`.

## Package Naming

The SHRDLU source directory is named `shrdlu-block/` to match the simulator
domain, but Python imports use the valid package name `shrdlu_agents`. The
editable install provides that mapping:

```toml
[tool.setuptools.package-dir]
shrdlu_agents = "shrdlu-block"
```

Do not create a local `shrdlu_agents` directory or symlink in the checkout.

Common model settings:

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:30000/v1
```

For TLC-backed runs:

```bash
export TLA2TOOLS_JAR=/path/to/tla2tools.jar
```

## Launching Agents

After `pip install -e`, use `run-agents` to launch the four canonical agents:

```bash
# Run these from the Git repository the agent should operate on.
run-agents git-basic
run-agents git-fsm

# Run these with SHRDLU_SIMULATOR_URL pointing at a running simulator.
run-agents shrdlu-basic -- --result-dir "$PWD/results"
run-agents shrdlu-fsm -- --result-dir "$PWD/results"
```

Both domains also expose planning-mode targets:

```bash
run-agents git-plan
run-agents git-advisory

run-agents shrdlu-plan -- --result-dir "$PWD/results"
run-agents shrdlu-advisory -- --result-dir "$PWD/results"
```

The same choices are available through option form:

```bash
run-agents --domain git --agent basic
run-agents --domain git --agent fsm
run-agents --domain git --agent plan
run-agents --domain git --agent advisory

run-agents --domain shrdlu --agent basic -- --result-dir "$PWD/results"
run-agents --domain shrdlu --agent fsm -- --result-dir "$PWD/results"
run-agents --domain shrdlu --agent plan -- --result-dir "$PWD/results"
run-agents --domain shrdlu --agent advisory -- --result-dir "$PWD/results"
```

Use `run-agents --list` to print the supported targets. Arguments after `--`
are passed through to the selected agent. `run_agents` with an underscore is
installed as the same command for shells/scripts that prefer that spelling.
No separate SHRDLU launcher command is installed.

## Interactive Terminal

The Git and SHRDLU interactive agents share the same terminal loop. In a real
TTY, normal line editing is enabled through Python `readline`: left/right arrows
move within the current line, up/down arrows recall previous entries, and
history is saved per terminal under
`$XDG_STATE_HOME/llm-agents-fsm/` or `~/.local/state/llm-agents-fsm/`.

Common commands:

- `/help`: show available commands.
- `/exit`, `/quit`, `exit`, or `quit`: exit the terminal.
- `/config`: show runtime planning settings for FSM agents.
- `/mode <fsm|plan|advisory>`: switch FSM planning mode.
- `/granularity <step|batch>`: switch planning granularity.
- `/violations <retry|ignore|advisory>`: switch violation handling.
- `/retries <n>`: switch planning retries.

Compatibility aliases are kept: `/planning-mode` is the same as `/mode`, and
`/planning` is the same as `/granularity`.

## Git Agents

Run these from the Git repository you want the agent to operate on:

```bash
cd /path/to/target/git/repo
run-agents git-basic
run-agents git-fsm
run-agents git-plan
run-agents git-advisory
```

Variants:

- `git-agent-basic.py`: basic terminal agent with allowlisted Git and shell
  commands.
- `git-agent-fsm.py`: observes atomic propositions, plans before executing,
  checks finite traces against property resources, then executes the accepted
  plan.

`git-plan` and `git-advisory` run the Git FSM implementation with the selected
planning mode.

Git-specific commands include `/model <name>` and `/cwd`. The FSM variant also
adds `/props`. The basic Git agent adds `/reset`, `/save`, and `/verbose`.
Saved basic-agent sessions are written to `.git-agent-sessions/` in the target
working directory.

## SHRDLU Block Agents

The SHRDLU agents talk to an already-running simulator HTTP service. Start the
simulator from the simulator project, for example:

```bash
cd /path/to/shrdlu-block-world
python3 -m shrdlu_blocks.simulator --headless
```

This checkout stores the SHRDLU source in `shrdlu-block/`. The editable install
exposes that directory as the `shrdlu_agents` Python package. If the sibling
simulator package is not installed, keep its checkout on `PYTHONPATH`.

Then run one of the canonical agent strategies through `run-agents`:

```bash
export SHRDLU_SIMULATOR_URL=http://127.0.0.1:8000
export SHRDLU_OPENAI_BASE_URL=http://127.0.0.1:30000/v1
export SHRDLU_OPENAI_API_KEY=EMPTY
export SHRDLU_OPENAI_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507

run-agents shrdlu-basic -- --result-dir "$PWD/results"
run-agents shrdlu-fsm -- --result-dir "$PWD/results"
```

The SHRDLU FSM supports the same planning modes as the Git FSM:

```bash
# FSM mode: plan toward the goal and retry/backtrack on property violations.
run-agents shrdlu-fsm -- \
  --planning-mode fsm \
  --max-branch-retries 3

# Plan mode: plan toward the goal, record property violations, and continue.
run-agents shrdlu-fsm -- \
  --planning-mode plan

# Advisory mode: include properties in the planning prompt, record violations,
# and continue.
run-agents shrdlu-fsm -- \
  --planning-mode advisory
```

The unified launcher also exposes these as `run-agents shrdlu-plan` and
`run-agents shrdlu-advisory`, matching Git's `run-agents git-plan` and
`run-agents git-advisory`.

Runtime controls:

- `/help`: print the current command list.
- `/state`: print the current simulator snapshot.
- `/reset`: reset the simulator.
- `/events`: print recent simulator action events.
- `/config`, `/mode`, `/granularity`, `/violations`, `/retries`: adjust live
  FSM planning settings when running `shrdlu-fsm`, `shrdlu-plan`, or
  `shrdlu-advisory`.
- `/quit`: exit.

Useful SHRDLU settings:

```bash
export SHRDLU_OPENAI_TEMPERATURE=0.2
export SHRDLU_OPENAI_MAX_TOKENS=512
export SHRDLU_AGENT_MAX_STEPS=50
export SHRDLU_AGENT_MAX_BRANCH_RETRIES=3
export SHRDLU_AGENT_FSM_PLANNING_MODE=fsm  # fsm | plan | advisory
export SHRDLU_AGENT_RESULT_DIR=/path/to/results
```

Set `SHRDLU_AGENT_RESULT_DIR=` to disable result writing.

## Source Layout

```text
git-system/
  git-agent-basic.py        Basic Git terminal agent
  git-agent-fsm.py          FSM Git planning agent
  resources/                Git atomic-proposition and property catalogs

shrdlu-block/                 Installed as the shrdlu_agents package
  __main__.py               Package module entry point
  shrdlu_agent_basic.py     Basic agent
  shrdlu_agent_fsm.py       FSM SHRDLU planning agent
  property_verifier.py      SHRDLU property checks
  resources/                SHRDLU atomic-proposition and property catalogs

utils/
  chat_terminal.py          Shared interactive terminal loop
  planning_terminal.py      Shared runtime planning-mode commands
  run_agents.py             Unified launcher for Git and SHRDLU agents
  planning_modes.py         Shared planning-mode vocabulary
  session.py                Shared JSON session schema
  planning_tree.py          Planning-tree helpers
  tla_verifier.py           TLA+/TLC spec generation and runner
```

## Generated Files

Do not commit local environments, Python caches, LLM trace output, saved
`.git-agent-sessions/`, or secrets. These are covered by `.gitignore`.
