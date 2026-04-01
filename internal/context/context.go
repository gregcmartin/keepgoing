package context

import (
	"context"
	"fmt"
	"log"
	"strings"

	"keepgoing/internal/db"
	"keepgoing/internal/llm"
)

const (
	// MaxMessages triggers compaction when exceeded.
	// Despite the model supporting 256K context, mlx_lm.server crashes
	// at high token counts. Keep this conservative.
	MaxMessages = 20
	// CompactWindow is how many old messages get summarized at once.
	CompactWindow = 10
)

// Manager handles context window compaction via rolling summarization.
// Following the Anthropic harness pattern: rather than relying solely on
// compaction, agents also maintain a progress file for explicit state
// reconstruction across sessions.
type Manager struct {
	db  *db.DB
	llm *llm.Client
}

func NewManager(database *db.DB, client *llm.Client) *Manager {
	return &Manager{db: database, llm: client}
}

// CompactIfNeeded checks if a task's conversation has grown too long
// and summarizes older messages to stay within the context window.
func (m *Manager) CompactIfNeeded(ctx context.Context, taskID int64) error {
	msgs, err := m.db.GetMessages(taskID)
	if err != nil {
		return fmt.Errorf("get messages: %w", err)
	}

	if len(msgs) <= MaxMessages {
		return nil
	}

	log.Printf("[context] Task %d has %d messages, compacting...", taskID, len(msgs))

	// Find non-system messages to summarize (keep system prompt intact)
	var toSummarize []db.Message
	var toSummarizeIDs []int64
	for _, msg := range msgs {
		if msg.Role == "system" {
			continue
		}
		toSummarize = append(toSummarize, msg)
		toSummarizeIDs = append(toSummarizeIDs, msg.ID)
		if len(toSummarize) >= CompactWindow {
			break
		}
	}

	if len(toSummarize) < 5 {
		return nil // Not enough to summarize
	}

	// Build the conversation text for summarization, sanitizing non-text content
	var sb strings.Builder
	for _, msg := range toSummarize {
		content := sanitizeForSummary(msg.Content)
		if len(content) > 2000 {
			content = content[:2000] + "...[truncated]"
		}
		sb.WriteString(fmt.Sprintf("[%s]: %s\n", msg.Role, content))
	}

	// Ask the LLM to summarize
	summaryMsgs := []llm.Message{
		{
			Role: "system",
			Content: "You are a summarization assistant. Summarize the following conversation history into a concise but complete summary. " +
				"Preserve all key decisions, findings, action items, and results. " +
				"Include specific data points, names, and URLs that were discovered. " +
				"The summary will replace these messages in the conversation, so nothing important should be lost.",
		},
		{
			Role:    "user",
			Content: "Summarize this conversation history:\n\n" + sb.String(),
		},
	}

	resp, err := m.llm.Chat(ctx, summaryMsgs, nil)
	if err != nil {
		return fmt.Errorf("summarize: %w", err)
	}

	// Delete old messages and insert summary
	if err := m.db.DeleteMessagesByIDs(toSummarizeIDs); err != nil {
		return fmt.Errorf("delete old messages: %w", err)
	}

	// Find the lowest step number from the summarized messages for ordering
	minStep := toSummarize[0].Step

	summary := "[CONTEXT SUMMARY - Earlier conversation compressed]\n\n" + resp.Content
	if _, err := m.db.AppendMessage(taskID, minStep, "system", summary, nil, nil); err != nil {
		return fmt.Errorf("insert summary: %w", err)
	}

	log.Printf("[context] Compacted %d messages into summary for task %d", len(toSummarize), taskID)
	return nil
}

// WriteProgressFile updates the progress file on disk for cross-session state.
// This follows the Anthropic pattern of using file-based state artifacts
// (like claude-progress.txt) for rapid context reconstruction.
func WriteProgressFile(path string, taskID int64, goal string, progress string) error {
	content := fmt.Sprintf("# Agent Progress\n\nTask ID: %d\nGoal: %s\n\n## Progress\n\n%s\n",
		taskID, goal, progress)

	return writeFile(path, content)
}

// ReadProgressFile reads the current progress state from disk.
func ReadProgressFile(path string) (string, error) {
	return readFile(path)
}

// sanitizeForSummary removes non-printable characters and HTML that confuse summarization.
func sanitizeForSummary(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r == '\n' || r == '\t' || (r >= 32 && r < 127) {
			b.WriteRune(r)
		} else if r >= 128 && r < 65536 {
			b.WriteRune(r) // Keep valid unicode
		}
		// Skip control chars and invalid bytes
	}
	result := b.String()
	// Strip HTML tags that bloat the summary
	result = stripHTMLTags(result)
	return result
}

// stripHTMLTags removes HTML tags but keeps text content.
func stripHTMLTags(s string) string {
	var b strings.Builder
	inTag := false
	for _, r := range s {
		if r == '<' {
			inTag = true
			continue
		}
		if r == '>' {
			inTag = false
			b.WriteRune(' ')
			continue
		}
		if !inTag {
			b.WriteRune(r)
		}
	}
	return b.String()
}
