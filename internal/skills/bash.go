package skills

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

const bashTimeout = 60 * time.Second

type BashSkill struct{}

type bashParams struct {
	Command string `json:"command"`
}

func (b *BashSkill) Name() string { return "bash" }

func (b *BashSkill) Description() string {
	return "Execute a bash command on macOS. Returns combined stdout and stderr. Use for file operations, system commands, data processing, and any CLI tool."
}

func (b *BashSkill) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"command": {
				"type": "string",
				"description": "The bash command to execute"
			}
		},
		"required": ["command"]
	}`)
}

func (b *BashSkill) Execute(ctx context.Context, params json.RawMessage) (string, error) {
	var p bashParams
	if err := json.Unmarshal(params, &p); err != nil {
		return "", fmt.Errorf("parse params: %w", err)
	}
	if strings.TrimSpace(p.Command) == "" {
		return "", fmt.Errorf("empty command")
	}

	cmdCtx, cancel := context.WithTimeout(ctx, bashTimeout)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, "bash", "-c", p.Command)
	output, err := cmd.CombinedOutput()
	result := string(output)

	if err != nil {
		if cmdCtx.Err() == context.DeadlineExceeded {
			return result, fmt.Errorf("command timed out after %s", bashTimeout)
		}
		// Return output even on non-zero exit — the agent needs to see stderr
		return fmt.Sprintf("%s\n[exit code: %s]", result, err.Error()), nil
	}

	return result, nil
}
