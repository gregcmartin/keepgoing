# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
go build -o keepgoing .                    # Build
./keepgoing -task "your goal here"         # Create and run a new task
./keepgoing                                # Resume the latest unfinished task
./run.sh -task "your goal here"            # Run with crash-resilient wrapper
```

The model server must be running separately:
```bash
mlx_lm.server --model nightmedia/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx --port 8000
```

Dependencies: `github.com/mattn/go-sqlite3` (CGo — requires C compiler).

## Architecture

**keepgoing** is a long-running autonomous agent harness in Go, designed to run 24/7 for weeks. It follows the Anthropic harness pattern: file-based state for cross-session continuity, single-task focus per session to prevent context exhaustion, and crash recovery via conversation replay.

### Core Loop (ReAct pattern — `internal/agent/agent.go`)

1. **Orient**: Load conversation history from SQLite + read progress file from `.keepgoing/`
2. **Recover**: Detect incomplete tool calls from crashes, re-execute them
3. **Think**: Send conversation + tool definitions to LLM
4. **Act**: Execute tool calls (skills), save results to DB
5. **Compact**: Rolling summarization when messages exceed threshold (40 msgs → summarize oldest 20)
6. **Repeat** until agent says "TASK COMPLETE"

### Key Components

- `internal/db/` — SQLite with WAL mode. Tables: `tasks` (hierarchical via parent_id), `agent_state` (conversation log per task), `results` (structured findings as JSON), `kv` (config)
- `internal/llm/` — Thin OpenAI-compatible HTTP client for mlx_vlm. Retry with exponential backoff (3 attempts). 120s timeout per request.
- `internal/skills/` — `Skill` interface with `Name/Description/Parameters/Execute`. Built-in: `bash` (60s timeout), `firecrawl_scrape`, `firecrawl_search` (120s timeout), `save_finding` (persists to results table)
- `internal/agent/` — Agent loop + sub-agent spawning. Sub-agents are child tasks running in goroutines (max 3 concurrent). Tools: `spawn_subtask`, `check_subtask`
- `internal/context/` — Rolling summarization compaction + progress file read/write for cross-session state
- `run.sh` — Bash wrapper with exponential backoff restart (5s→300s cap, max 10 consecutive failures)

### State Model

All conversation state lives in SQLite (`agent_state` table). On restart, the agent replays its full conversation history. Progress files in `.keepgoing/` provide human-readable state and enable fast orientation on resume.
