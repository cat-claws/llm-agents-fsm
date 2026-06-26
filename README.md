# llm-agents-fsm

Research/prototype agents that combine LLM tool use with explicit planning
traces and finite-state property checks. The repository currently contains two
agent domains:

- `git-system/`: terminal Git agents with basic and merged FSM variants.
- `shrdlu-block/`: OpenAI-compatible agents for a standalone SHRDLU
  blocks-world simulator.
- `utils/`: shared session serialization, planning-tree helpers, and TLA+/TLC
  verification utilities.

The Git scripts can be run directly from source. The unified launcher and
SHRDLU entry points are intended to run from an editable install, which maps the
`shrdlu_agents` import name to the `shrdlu-block/` source directory.

## Requirements

- Python 3.10 or newer.
- `openai` for all LLM-backed agents.
- Git for the `git-system` agents.
- Optional: Java plus `tla2tools.jar` for TLC verification. Set
  `TLA2TOOLS_JAR` to the jar path. If it is unavailable, the TLC runner reports
  verification as skipped.
- Optional for SHRDLU FSM modes: the sibling `shrdlu_blocks` simulator
  package on `PYTHONPATH`.

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

## Unified Launcher

After `pip install -e`, `run-agents` covers the four canonical agents:

```bash
run-agents git-basic
run-agents git-fsm
run-agents shrdlu-reactive -- --trace-dir "$PWD/agent_traces"
run-agents shrdlu-fsm -- --trace-dir "$PWD/agent_traces"
```

Equivalent option form:

```bash
run-agents --domain git --agent basic
run-agents --domain git --agent fsm
run-agents --domain shrdlu --agent reactive -- --trace-dir "$PWD/agent_traces"
run-agents --domain shrdlu --agent fsm -- --trace-dir "$PWD/agent_traces"
```

Use `run-agents --list` to print the supported targets. Arguments after `--`
are passed through to the selected agent. `run_agents` with an underscore is
installed as the same command for shells/scripts that prefer that spelling.

## Git Agents

Run these from the Git repository you want the agent to operate on:

```bash
cd /path/to/target/git/repo
run-agents git-basic
run-agents git-fsm
```

Variants:

- `git-agent-basic.py`: reactive terminal agent with allowlisted Git and shell
  commands.
- `git-agent-fsm.py`: observes atomic propositions, plans before executing,
  checks finite traces against property resources, then executes the accepted
  plan.

The merged FSM can emulate the old plan-first behavior by changing parameters:

```bash
GIT_AGENT_FSM_PLANNING_GRANULARITY=batch \
GIT_AGENT_FSM_VIOLATION_POLICY=ignore \
GIT_AGENT_FSM_MAX_RETRIES=1 \
python3 /home/robot/llm-agents-fsm/git-system/git-agent-fsm.py
```

Interactive commands include `/help`, `/model <name>`, `/config`,
`/planning <step|batch>`, `/violations <retry|ignore>`, `/retries <n>`,
and `/quit`.
Saved sessions are written to `.git-agent-sessions/` in the target working
directory.

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

Then run one of the canonical agent strategies:

```bash
export SHRDLU_SIMULATOR_URL=http://127.0.0.1:8000
export SHRDLU_OPENAI_BASE_URL=http://127.0.0.1:30000/v1
export SHRDLU_OPENAI_API_KEY=EMPTY
export SHRDLU_OPENAI_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507

run-agents shrdlu-reactive -- --trace-dir "$PWD/agent_traces"
run-agents shrdlu-fsm -- --trace-dir "$PWD/agent_traces"
```

The merged FSM exposes the old plan/FSM distinction as parameters:

```bash
# FSM-style: plan a suffix and retry/backtrack on property violations.
run-agents shrdlu-fsm -- \
  --planning-granularity batch \
  --violation-policy retry \
  --max-branch-retries 3

# Plan-style: plan a suffix, record property violations, and continue.
run-agents shrdlu-fsm -- \
  --planning-granularity batch \
  --violation-policy ignore \
  --max-branch-retries 1

# Stepwise planning with FSM retry behavior.
run-agents shrdlu-fsm -- \
  --planning-granularity step \
  --violation-policy retry
```

Legacy `--agent preplanned`, `--agent predictive`, and `--agent suffix` names
are accepted by the SHRDLU launcher as aliases for `--agent fsm` with matching
parameter presets. The unified launcher also exposes these as
`run-agents shrdlu-preplanned`, `run-agents shrdlu-predictive`, and
`run-agents shrdlu-suffix`.

Runtime controls:

- `/state`: print the current simulator snapshot.
- `/reset`: reset the simulator.
- `/events`: print recent simulator action events.
- `/quit`: exit.

Useful SHRDLU settings:

```bash
export SHRDLU_OPENAI_TEMPERATURE=0.2
export SHRDLU_OPENAI_MAX_TOKENS=512
export SHRDLU_AGENT_MAX_STEPS=50
export SHRDLU_AGENT_MAX_BRANCH_RETRIES=3
export SHRDLU_AGENT_FSM_PLANNING_GRANULARITY=batch
export SHRDLU_AGENT_FSM_VIOLATION_POLICY=retry
export SHRDLU_AGENT_TRACE_DIR=/path/to/agent_traces
```

Set `SHRDLU_AGENT_TRACE_DIR=` to disable trace writing.

## Source Layout

```text
git-system/
  git-agent-basic.py        Basic Git terminal agent
  git-agent-fsm.py          Merged plan/FSM Git agent
  resources/                Git atomic-proposition and property catalogs

shrdlu-block/                 Installed as the shrdlu_agents package
  shrdlu_agent_basic.py     Reactive/basic agent
  shrdlu_agent_fsm.py       Merged predictive plan/FSM agent
  property_verifier.py      SHRDLU property checks
  resources/                SHRDLU atomic-proposition and property catalogs

utils/
  run_agents.py             Unified launcher for Git and SHRDLU agents
  session.py                Shared JSON session schema
  planning_tree.py          Planning-tree helpers
  tla_verifier.py           TLA+/TLC spec generation and runner
```

## Generated Files

Do not commit local environments, Python caches, LLM trace output, saved
`.git-agent-sessions/`, or secrets. These are covered by `.gitignore`.
