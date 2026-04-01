package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"path/filepath"
	"strings"
	"time"

	kgctx "keepgoing/internal/context"
	"keepgoing/internal/db"
	"keepgoing/internal/llm"
	"keepgoing/internal/skills"
)

const systemPrompt = `You are a long-running autonomous research agent. You operate methodically and persistently, designed to work for days on complex research tasks.

## How You Work
1. **Orient**: At the start of each session, review your progress file and recent history to understand where you left off.
2. **Plan**: Break complex goals into concrete sub-tasks. Work on one sub-task at a time to avoid context exhaustion.
3. **Execute**: Use your tools to search the web, scrape pages, run shell commands, and save findings.
4. **Record**: Save every meaningful finding using save_finding IMMEDIATELY when you discover it.
5. **Persist**: You may be interrupted and restarted. Always leave clear state so you can resume.

## Critical Rules
- NEVER say "TASK COMPLETE" until you have met the target number of findings specified in the task.
- Save each CISO as a separate save_finding call with kind="ciso" as soon as you identify them.
- Move FAST: research one company, save the CISO finding, immediately move to the next company. Do NOT over-research a single company.
- Use LISTS: search for "list of Fortune 500 CISOs", "top CISOs in healthcare", "CISOs in defense industry", etc.
- For each CISO, save: name, title, company, industry, LinkedIn URL (if found), and why they'd want on-premise AI security.
- If a tool call fails, try an alternative approach rather than repeating the same action.
- When scraping, be respectful of rate limits.
- Use bash (curl, grep) for web research since firecrawl may not be available.
- Use save_finding to persist structured results to the database.
- Keep moving. Breadth over depth. Find the CISO, save them, move on.

You have been running for a long time and may continue for days. Stay focused and methodical.`

// Agent drives the core ReAct loop for a single task.
type Agent struct {
	db             *db.DB
	llm            *llm.Client
	skills         *skills.Registry
	ctxManager     *kgctx.Manager
	taskID         int64
	step           int
	workDir        string
	progressDir    string
	targetFindings int // minimum findings before task can complete (0 = no minimum)
}

// Config holds agent initialization parameters.
type Config struct {
	DB             *db.DB
	LLM            *llm.Client
	Skills         *skills.Registry
	CtxManager     *kgctx.Manager
	TaskID         int64
	WorkDir        string
	ProgressDir    string
	TargetFindings int
}

func New(cfg Config) *Agent {
	return &Agent{
		db:             cfg.DB,
		llm:            cfg.LLM,
		skills:         cfg.Skills,
		ctxManager:     cfg.CtxManager,
		taskID:         cfg.TaskID,
		workDir:        cfg.WorkDir,
		progressDir:    cfg.ProgressDir,
		targetFindings: cfg.TargetFindings,
	}
}

// Run executes the agent loop until the task is done or context is cancelled.
// Follows the Anthropic harness pattern:
//   - On startup, orient by reading progress file and conversation history
//   - Crash recovery: detect incomplete tool calls and re-execute them
//   - Single-task focus to prevent context exhaustion
//   - File-based state (progress file) for cross-session continuity
func (a *Agent) Run(ctx context.Context) error {
	task, err := a.db.GetTask(a.taskID)
	if err != nil {
		return fmt.Errorf("get task: %w", err)
	}
	if task == nil {
		return fmt.Errorf("task %d not found", a.taskID)
	}

	log.Printf("[agent] Starting task %d: %s", a.taskID, task.Goal)

	// Mark task as running
	if err := a.db.UpdateTaskStatus(a.taskID, "running", nil); err != nil {
		return fmt.Errorf("update task status: %w", err)
	}

	// Load existing conversation or initialize fresh
	msgs, err := a.db.GetMessages(a.taskID)
	if err != nil {
		return fmt.Errorf("get messages: %w", err)
	}

	if len(msgs) == 0 {
		// Fresh task — initialize with system prompt and goal
		if err := a.initConversation(task); err != nil {
			return fmt.Errorf("init conversation: %w", err)
		}
	} else {
		// Resuming — find current step and check for incomplete tool calls
		a.step = msgs[len(msgs)-1].Step + 1
		if err := a.recoverIncompleteToolCalls(ctx, msgs); err != nil {
			return fmt.Errorf("recover tool calls: %w", err)
		}
	}

	// Main ReAct loop
	for {
		select {
		case <-ctx.Done():
			a.saveProgress("Agent interrupted. Will resume on restart.")
			return ctx.Err()
		default:
		}

		// Build messages for LLM from conversation history
		chatMsgs, err := a.buildChatMessages()
		if err != nil {
			return fmt.Errorf("build messages: %w", err)
		}

		// Call LLM — combine skill tools with sub-agent management tools
		allTools := append(a.skills.Tools(), SubAgentTools()...)
		log.Printf("[agent] Step %d: calling LLM (%d messages, %d tools)", a.step, len(chatMsgs), len(allTools))
		resp, err := a.llm.Chat(ctx, chatMsgs, allTools)
		if err != nil {
			log.Printf("[agent] LLM error: %v, waiting before retry", err)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(10 * time.Second):
			}
			continue
		}

		log.Printf("[agent] Step %d: finish_reason=%s, tool_calls=%d, tokens=%d",
			a.step, resp.FinishReason, len(resp.ToolCalls), resp.Usage.TotalTokens)

		// Save assistant message
		var toolCallsJSON *string
		if len(resp.ToolCalls) > 0 {
			tc, _ := json.Marshal(resp.ToolCalls)
			s := string(tc)
			toolCallsJSON = &s
		}
		content := resp.Content
		if content == "" && len(resp.ToolCalls) > 0 {
			content = "[tool calls]"
		}
		if _, err := a.db.AppendMessage(a.taskID, a.step, "assistant", content, toolCallsJSON, nil); err != nil {
			return fmt.Errorf("save assistant message: %w", err)
		}

		// If no tool calls, the agent is done thinking — check if task is complete
		if len(resp.ToolCalls) == 0 {
			// Check actual finding count before allowing completion
			results, _ := a.db.GetResults(a.taskID)
			cisoCount := 0
			for _, r := range results {
				if r.Kind == "ciso" {
					cisoCount++
				}
			}

			if a.isTaskComplete(resp.Content) && cisoCount >= a.targetFindings {
				result := fmt.Sprintf("Found %d CISOs. %s", cisoCount, resp.Content)
				if err := a.db.UpdateTaskStatus(a.taskID, "done", &result); err != nil {
					return fmt.Errorf("update task done: %w", err)
				}
				a.saveProgress(fmt.Sprintf("Task completed: %d CISOs found", cisoCount))
				log.Printf("[agent] Task %d completed with %d CISOs", a.taskID, cisoCount)
				return nil
			}
			// Agent responded without tool calls but isn't done — nudge it with progress
			a.step++
			nudge := fmt.Sprintf(
				"You have saved %d CISOs so far. You need %d total. Keep going! "+
					"Search for more companies and CISOs. Try searching for: "+
					"'Fortune 500 CISO list', 'healthcare company CISO', 'defense contractor CISO', "+
					"'financial services CISO', 'energy company CISO'. "+
					"Find the CISO, save them with save_finding (kind='ciso'), and move to the next company.",
				cisoCount, a.targetFindings,
			)
			if _, err := a.db.AppendMessage(a.taskID, a.step, "user", nudge, nil, nil); err != nil {
				return fmt.Errorf("save nudge: %w", err)
			}
			a.step++
			continue
		}

		// Execute tool calls
		a.step++
		for _, tc := range resp.ToolCalls {
			result, err := a.executeToolCall(ctx, tc)
			if err != nil {
				result = fmt.Sprintf("Error: %v", err)
			}

			// Truncate very long results — even with 256K context,
			// raw HTML dumps waste tokens. Keep useful content.
			if len(result) > 6000 {
				result = result[:6000] + "\n\n[output truncated at 6000 chars]"
			}

			tcID := tc.ID
			if _, err := a.db.AppendMessage(a.taskID, a.step, "tool", result, nil, &tcID); err != nil {
				return fmt.Errorf("save tool result: %w", err)
			}
		}

		// Context compaction check
		if err := a.ctxManager.CompactIfNeeded(ctx, a.taskID); err != nil {
			log.Printf("[agent] Context compaction error: %v", err)
		}

		// Periodic nudge to save findings and update progress
		if a.step%10 == 0 {
			results, _ := a.db.GetResults(a.taskID)
			cisoCount := 0
			for _, r := range results {
				if r.Kind == "ciso" {
					cisoCount++
				}
			}

			a.saveProgress(fmt.Sprintf("Step %d. CISOs found: %d / %d target.", a.step, cisoCount, a.targetFindings))
			log.Printf("[agent] Progress: %d / %d CISOs found", cisoCount, a.targetFindings)

			// Always remind with progress count
			reminder := fmt.Sprintf(
				"PROGRESS UPDATE: You have saved %d / %d CISOs. "+
					"Keep going! For each new company: (1) search for the CISO name, "+
					"(2) call save_finding with kind='ciso' immediately, "+
					"(3) move to the next company. Speed is important — don't over-research.",
				cisoCount, a.targetFindings,
			)
			if _, err := a.db.AppendMessage(a.taskID, a.step, "user", reminder, nil, nil); err != nil {
				log.Printf("[agent] Error saving reminder: %v", err)
			}
			a.step++
		}

		a.step++
	}
}

func (a *Agent) initConversation(task *db.Task) error {
	// System prompt
	if _, err := a.db.AppendMessage(a.taskID, 0, "system", systemPrompt, nil, nil); err != nil {
		return err
	}

	// Read progress file if resuming across sessions
	progress, _ := kgctx.ReadProgressFile(a.progressPath())
	userMsg := fmt.Sprintf("Your task:\n\n%s", task.Goal)
	if progress != "" {
		userMsg += fmt.Sprintf("\n\n## Previous Progress\n\n%s", progress)
	}

	if _, err := a.db.AppendMessage(a.taskID, 1, "user", userMsg, nil, nil); err != nil {
		return err
	}
	a.step = 2
	return nil
}

// recoverIncompleteToolCalls detects if the last assistant message had tool calls
// that never got results (crash mid-execution) and re-executes them.
func (a *Agent) recoverIncompleteToolCalls(ctx context.Context, msgs []db.Message) error {
	if len(msgs) == 0 {
		return nil
	}

	// Find the last assistant message with tool calls
	var lastAssistant *db.Message
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "assistant" && msgs[i].ToolCalls != nil {
			lastAssistant = &msgs[i]
			break
		}
	}
	if lastAssistant == nil {
		return nil
	}

	// Parse tool calls
	var toolCalls []llm.ToolCall
	if err := json.Unmarshal([]byte(*lastAssistant.ToolCalls), &toolCalls); err != nil {
		return nil // Can't parse, skip recovery
	}

	// Check which tool calls have results
	completedIDs := make(map[string]bool)
	for _, msg := range msgs {
		if msg.Role == "tool" && msg.ToolCallID != nil {
			completedIDs[*msg.ToolCallID] = true
		}
	}

	// Re-execute any incomplete tool calls
	for _, tc := range toolCalls {
		if completedIDs[tc.ID] {
			continue
		}
		log.Printf("[agent] Recovering incomplete tool call: %s(%s)", tc.Function.Name, tc.ID)
		result, err := a.executeToolCall(ctx, tc)
		if err != nil {
			result = fmt.Sprintf("Error (recovery): %v", err)
		}
		tcID := tc.ID
		if _, err := a.db.AppendMessage(a.taskID, a.step, "tool", result, nil, &tcID); err != nil {
			return err
		}
	}
	return nil
}

func (a *Agent) executeToolCall(ctx context.Context, tc llm.ToolCall) (string, error) {
	name := tc.Function.Name
	params := json.RawMessage(tc.Function.Arguments)

	log.Printf("[agent] Executing tool: %s", name)

	// Handle built-in agent management tools
	switch name {
	case "spawn_subtask":
		return a.handleSpawnSubtask(ctx, params)
	case "check_subtask":
		return a.handleCheckSubtask(params)
	default:
		return a.skills.Execute(ctx, name, params)
	}
}

func (a *Agent) buildChatMessages() ([]llm.Message, error) {
	msgs, err := a.db.GetMessages(a.taskID)
	if err != nil {
		return nil, err
	}

	chatMsgs := make([]llm.Message, 0, len(msgs))
	for _, m := range msgs {
		cm := llm.Message{
			Role:    m.Role,
			Content: m.Content,
		}
		if m.ToolCalls != nil {
			var tcs []llm.ToolCall
			json.Unmarshal([]byte(*m.ToolCalls), &tcs)
			cm.ToolCalls = tcs
		}
		if m.ToolCallID != nil {
			cm.ToolCallID = *m.ToolCallID
		}
		chatMsgs = append(chatMsgs, cm)
	}
	return chatMsgs, nil
}

func (a *Agent) isTaskComplete(content string) bool {
	upper := strings.ToUpper(content)
	return strings.Contains(upper, "TASK COMPLETE")
}

func (a *Agent) progressPath() string {
	return filepath.Join(a.progressDir, fmt.Sprintf("task-%d-progress.txt", a.taskID))
}

func (a *Agent) saveProgress(note string) {
	task, err := a.db.GetTask(a.taskID)
	if err != nil {
		log.Printf("[agent] Error getting task for progress: %v", err)
		return
	}
	if err := kgctx.WriteProgressFile(a.progressPath(), a.taskID, task.Goal, note); err != nil {
		log.Printf("[agent] Error writing progress: %v", err)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
