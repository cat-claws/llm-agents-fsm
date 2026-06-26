# llm-agents-fsm

Research/prototype agents that combine LLM tool use with explicit planning
traces and finite-state property checks. The repository currently contains two
agent domains:

- `git-system/`: terminal Git agents with basic and merged FSM variants.
- `shrdlu-block/`: OpenAI-compatible agents for a standalone SHRDLU
  blocks-world simulator.
- `utils/`: shared session serialization, planning-tree helpers, and TLA+/TLC
  verification utilities.

The code is source-only at the moment: there is no package metadata and no
pinned requirements file in this checkout.

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
```

Common model settings:

```bash
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:30000/v1
```

For TLC-backed runs:

```bash
export TLA2TOOLS_JAR=/path/to/tla2tools.jar
```

## Git Agents

Run these from the Git repository you want the agent to operate on:

```bash
cd /path/to/target/git/repo
python3 /home/robot/llm-agents-fsm/git-system/git-agent-basic.py
python3 /home/robot/llm-agents-fsm/git-system/git-agent-fsm.py
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

This checkout stores the SHRDLU source in `shrdlu-block/`, while the code
imports it as `shrdlu_agents`. Until the project is packaged or the directory is
renamed, create a local import shim before running the agents:

```bash
cd /home/robot/llm-agents-fsm
ln -sfn shrdlu-block shrdlu_agents
export PYTHONPATH="$PWD:/path/to/shrdlu-block-world:${PYTHONPATH}"
```

Then run one of the canonical agent strategies:

```bash
export SHRDLU_SIMULATOR_URL=http://127.0.0.1:8000
export SHRDLU_OPENAI_BASE_URL=http://127.0.0.1:30000/v1
export SHRDLU_OPENAI_API_KEY=EMPTY
export SHRDLU_OPENAI_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507

python3 -m shrdlu_agents.run_agent --agent reactive
python3 -m shrdlu_agents.run_agent --agent fsm
```

The merged FSM exposes the old plan/FSM distinction as parameters:

```bash
# FSM-style: plan a suffix and retry/backtrack on property violations.
python3 -m shrdlu_agents.run_agent --agent fsm \
  --planning-granularity batch \
  --violation-policy retry \
  --max-branch-retries 3

# Plan-style: plan a suffix, record property violations, and continue.
python3 -m shrdlu_agents.run_agent --agent fsm \
  --planning-granularity batch \
  --violation-policy ignore \
  --max-branch-retries 1

# Stepwise planning with FSM retry behavior.
python3 -m shrdlu_agents.run_agent --agent fsm \
  --planning-granularity step \
  --violation-policy retry
```

Legacy `--agent preplanned`, `--agent predictive`, and `--agent suffix` names
are accepted as aliases for `--agent fsm` with matching parameter presets.

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

shrdlu-block/
  run_agent.py              SHRDLU launcher
  shrdlu_agent_basic.py     Reactive/basic agent
  shrdlu_agent_fsm.py       Merged predictive plan/FSM agent
  property_verifier.py      SHRDLU property checks
  resources/                SHRDLU atomic-proposition and property catalogs

utils/
  session.py                Shared JSON session schema
  planning_tree.py          Planning-tree helpers
  tla_verifier.py           TLA+/TLC spec generation and runner
```

## Generated Files

Do not commit local environments, Python caches, LLM trace output, saved
`.git-agent-sessions/`, secrets, or local import shims. These are covered by
`.gitignore`.
