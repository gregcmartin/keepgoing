# keepgoing

A long-running autonomous agent harness built in Go. Designed to run 24/7 for weeks on any given task, inspired by [Anthropic's harness patterns](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) for long-running agents.

Uses a local 27B parameter model via Apple Silicon MLX for inference — no cloud API keys required.

## Quick Start

### Prerequisites

- Go 1.21+
- Python 3.12+ with pip
- Homebrew (macOS)

### Install Dependencies

```bash
# Local LLM server + TurboQuant KV cache compression
pip install mlx-lm
pip install git+https://github.com/rachittshah/mlx-turboquant.git

# Or install all Python deps at once
pip install -r requirements.txt

# Node.js (for firecrawl CLI)
brew install node
```

### Build & Run

```bash
# Build
go build -o keepgoing .

# Start the model server with TurboQuant (3-5x KV cache compression)
python scripts/start_server.py

# Or without TurboQuant
python scripts/start_server.py --no-turboquant

# Or directly via mlx_lm
mlx_lm.server --model nightmedia/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx --port 8000

# Create and run a new task
./keepgoing -task "your research goal here"

# Resume the latest unfinished task (after crash/restart)
./keepgoing

# Run with crash-resilient wrapper (auto-restarts with exponential backoff)
./run.sh -task "your research goal here"
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-task` | | New task goal (omit to resume latest) |
| `-db` | `keepgoing.db` | SQLite database path |
| `-endpoint` | `http://localhost:8000` | LLM server endpoint |
| `-model` | auto | Model name override |
| `-workdir` | `.` | Working directory |

## Architecture

### Core Loop (ReAct Pattern)

The agent follows a Think → Act → Observe cycle, persisted to SQLite so it can resume after any crash:

```
┌─────────────────────────────────────────────────┐
│                   Agent Loop                     │
│                                                  │
│  1. Orient   → Load history from SQLite          │
│               → Read progress file               │
│               → Recover incomplete tool calls     │
│                                                  │
│  2. Think    → Send conversation + tools to LLM  │
│                                                  │
│  3. Act      → Execute tool calls (bash, search) │
│               → Save results to DB               │
│                                                  │
│  4. Compact  → Summarize old messages if > 40    │
│               → Update progress file             │
│                                                  │
│  5. Repeat   → Until "TASK COMPLETE"             │
└─────────────────────────────────────────────────┘
```

### Key Design Decisions

Informed by Anthropic's research on [effective harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) and [harness design](https://www.anthropic.com/engineering/harness-design-long-running-apps):

- **File-based state for cross-session continuity** — Progress files (`.keepgoing/task-N-progress.txt`) enable fast orientation on resume, rather than relying solely on conversation replay
- **Single-task focus** — Each agent session works on one sub-task at a time to prevent context exhaustion
- **Crash recovery via conversation replay** — On restart, the agent loads its full conversation from SQLite and re-executes any incomplete tool calls
- **Rolling context compaction** — When messages exceed 40, the oldest 20 are summarized by the LLM into a single message, preserving key findings

### Skills System

Pluggable tools the agent can invoke:

| Skill | Description | Timeout |
|-------|-------------|---------|
| `bash` | Execute any shell command on macOS | 60s |
| `firecrawl_scrape` | Scrape a URL to markdown via Firecrawl CLI | 120s |
| `firecrawl_search` | Web search via Firecrawl CLI | 120s |
| `save_finding` | Persist structured research results to SQLite | — |
| `spawn_subtask` | Spawn an independent sub-agent (max 3 concurrent) | — |
| `check_subtask` | Check sub-agent status and results | — |

### Sub-Agent System

The parent agent can spawn up to 3 concurrent sub-agents, each running in its own goroutine with a separate conversation history. Sub-agents are child rows in the `tasks` table linked by `parent_id`.

```
Root Task (research CISOs)
├── Sub-task 1 (research Company A) → running
├── Sub-task 2 (research Company B) → done
└── Sub-task 3 (research Company C) → running
```

### SQLite Schema

Four tables:

- **`tasks`** — Hierarchical task tree (parent_id for sub-agents), tracks status (pending/running/done/failed)
- **`agent_state`** — Conversation log per task (role, content, tool_calls), ordered by step number
- **`results`** — Structured research findings as JSON blobs (kind: company/ciso/contact/note)
- **`kv`** — Key-value store for config and bookkeeping

WAL mode enabled for concurrent sub-agent writes.

### Resilient Runner (`run.sh`)

Bash wrapper that:
- Builds the Go binary
- Checks model server connectivity
- Restarts agent on crash with exponential backoff (5s → 300s cap)
- Stops after 10 consecutive failures
- Resets failure count when model server is healthy

### TurboQuant KV Cache Compression

Optional integration with [mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) for PolarQuant KV cache quantization. This compresses the KV cache 3-5x, enabling much longer effective context on the same hardware.

```bash
# Start with 4-bit KV cache (default, best quality)
python scripts/start_server.py --bits 4

# More aggressive compression (3-bit)
python scripts/start_server.py --bits 3
```

| Bits | Compression | Quality (cosine sim) |
|------|------------|---------------------|
| 4    | ~4.6x      | 0.995+              |
| 3    | ~4.6x      | 0.995+              |
| 2    | ~4.0x      | ~0.97               |

For a 27B model with 256K context, TurboQuant can reduce KV cache memory from ~40GB to ~8-10GB, making full context window usage practical on machines with 64GB+ unified memory.

## Project Structure

```
keepgoing/
├── main.go                      # Entry point, CLI flags, wiring
├── run.sh                       # Crash-resilient runner
├── scripts/
│   └── start_server.py          # Model server with TurboQuant integration
├── requirements.txt             # Python dependencies
├── internal/
│   ├── agent/
│   │   ├── agent.go             # Core ReAct loop with crash recovery
│   │   └── subagent.go          # Sub-agent spawning and lifecycle
│   ├── context/
│   │   ├── context.go           # Rolling summarization compaction
│   │   └── fileutil.go          # Progress file I/O
│   ├── db/
│   │   ├── db.go                # SQLite connection, migrations, queries
│   │   └── schema.sql           # Embedded schema (4 tables)
│   ├── llm/
│   │   └── client.go            # OpenAI-compatible HTTP client
│   └── skills/
│       ├── skills.go            # Skill interface and registry
│       ├── bash.go              # Shell command execution
│       ├── firecrawl.go         # Web scraping/search
│       └── savefinding.go       # Persist findings to DB
├── go.mod
└── go.sum
```

## Monitoring a Running Agent

```bash
# Watch live logs
tail -f <log output path>

# Check saved findings
sqlite3 keepgoing.db "SELECT kind, data FROM results;"

# Check task status
sqlite3 keepgoing.db "SELECT id, status, substr(goal, 1, 80) FROM tasks;"

# Read progress file
cat .keepgoing/task-1-progress.txt

# Count conversation steps
sqlite3 keepgoing.db "SELECT COUNT(*) FROM agent_state WHERE task_id = 1;"
```

## Model

Uses [`nightmedia/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx`](https://huggingface.co/nightmedia/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx) — a 27B parameter model quantized for Apple Silicon MLX. Served locally via `mlx_lm.server` with an OpenAI-compatible API on port 8000. ~15GB download, runs in ~15GB unified memory.
