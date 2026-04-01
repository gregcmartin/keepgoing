package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"

	"keepgoing/internal/db"
	"keepgoing/internal/llm"
)

const maxConcurrentSubAgents = 3

// subAgentTracker manages running sub-agents.
type subAgentTracker struct {
	mu      sync.Mutex
	agents  map[int64]context.CancelFunc
}

var tracker = &subAgentTracker{
	agents: make(map[int64]context.CancelFunc),
}

// spawnSubtaskParams matches the tool call arguments.
type spawnSubtaskParams struct {
	Goal string `json:"goal"`
}

type checkSubtaskParams struct {
	TaskID int64 `json:"task_id"`
}

// SubAgentTools returns the tool definitions for sub-agent management.
func SubAgentTools() []llm.Tool {
	return []llm.Tool{
		{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "spawn_subtask",
				Description: "Spawn an independent sub-agent to work on a specific sub-task in parallel. The sub-agent has the same tools and capabilities. Use for tasks that can be researched independently (e.g., researching a specific company). Max 3 concurrent sub-agents.",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {
						"goal": {
							"type": "string",
							"description": "The specific goal for the sub-agent to accomplish"
						}
					},
					"required": ["goal"]
				}`),
			},
		},
		{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        "check_subtask",
				Description: "Check the status and result of a previously spawned sub-task.",
				Parameters: json.RawMessage(`{
					"type": "object",
					"properties": {
						"task_id": {
							"type": "integer",
							"description": "The task ID returned from spawn_subtask"
						}
					},
					"required": ["task_id"]
				}`),
			},
		},
	}
}

func (a *Agent) handleSpawnSubtask(ctx context.Context, params json.RawMessage) (string, error) {
	var p spawnSubtaskParams
	if err := json.Unmarshal(params, &p); err != nil {
		return "", fmt.Errorf("parse spawn params: %w", err)
	}

	// Check concurrency limit
	running, err := a.db.CountRunningChildren(a.taskID)
	if err != nil {
		return "", fmt.Errorf("count running children: %w", err)
	}
	if running >= maxConcurrentSubAgents {
		return fmt.Sprintf("Cannot spawn sub-agent: already %d/%d running. Wait for one to complete or check their status.", running, maxConcurrentSubAgents), nil
	}

	// Create sub-task
	parentID := a.taskID
	childID, err := a.db.CreateTask(&parentID, p.Goal)
	if err != nil {
		return "", fmt.Errorf("create sub-task: %w", err)
	}

	log.Printf("[agent] Spawning sub-agent for task %d: %s", childID, truncate(p.Goal, 80))

	// Launch sub-agent in goroutine
	subCtx, cancel := context.WithCancel(ctx)
	tracker.mu.Lock()
	tracker.agents[childID] = cancel
	tracker.mu.Unlock()

	subAgent := New(Config{
		DB:          a.db,
		LLM:         a.llm,
		Skills:      a.skills,
		CtxManager:  a.ctxManager,
		TaskID:      childID,
		WorkDir:     a.workDir,
		ProgressDir: a.progressDir,
	})

	go func() {
		defer func() {
			tracker.mu.Lock()
			delete(tracker.agents, childID)
			tracker.mu.Unlock()
		}()

		if err := subAgent.Run(subCtx); err != nil {
			log.Printf("[sub-agent %d] Error: %v", childID, err)
			failMsg := err.Error()
			a.db.UpdateTaskStatus(childID, "failed", &failMsg)
		}
	}()

	return fmt.Sprintf("Sub-agent spawned (task_id=%d). Use check_subtask with this ID to monitor progress.", childID), nil
}

func (a *Agent) handleCheckSubtask(params json.RawMessage) (string, error) {
	var p checkSubtaskParams
	if err := json.Unmarshal(params, &p); err != nil {
		return "", fmt.Errorf("parse check params: %w", err)
	}

	task, err := a.db.GetTask(p.TaskID)
	if err != nil {
		return "", fmt.Errorf("get sub-task: %w", err)
	}
	if task == nil {
		return fmt.Sprintf("Sub-task %d not found", p.TaskID), nil
	}

	// Gather any findings from this sub-task
	results, err := a.db.GetResults(p.TaskID)
	if err != nil {
		return "", fmt.Errorf("get results: %w", err)
	}

	var sb fmt.Stringer = &statusBuilder{task: task, results: results}
	return sb.String(), nil
}

type statusBuilder struct {
	task    *db.Task
	results []db.Result
}

func (sb *statusBuilder) String() string {
	var b []byte
	b = append(b, fmt.Sprintf("Sub-task %d:\n  Status: %s\n  Goal: %s\n", sb.task.ID, sb.task.Status, sb.task.Goal)...)
	if sb.task.Result != nil {
		b = append(b, fmt.Sprintf("  Result: %s\n", *sb.task.Result)...)
	}
	if len(sb.results) > 0 {
		b = append(b, fmt.Sprintf("  Findings: %d saved\n", len(sb.results))...)
		for _, r := range sb.results {
			b = append(b, fmt.Sprintf("    - [%s] %s\n", r.Kind, truncate(string(r.Data), 200))...)
		}
	}
	return string(b)
}
