package skills

import (
	"context"
	"encoding/json"
	"fmt"

	"keepgoing/internal/llm"
)

// Skill is a capability the agent can invoke via tool calls.
type Skill interface {
	Name() string
	Description() string
	Parameters() json.RawMessage
	Execute(ctx context.Context, params json.RawMessage) (string, error)
}

// Registry holds all available skills and converts them to LLM tool definitions.
type Registry struct {
	skills map[string]Skill
}

func NewRegistry() *Registry {
	return &Registry{skills: make(map[string]Skill)}
}

func (r *Registry) Register(s Skill) {
	r.skills[s.Name()] = s
}

// Get returns a skill by name.
func (r *Registry) Get(name string) (Skill, bool) {
	s, ok := r.skills[name]
	return s, ok
}

// Tools returns all skills as LLM tool definitions.
func (r *Registry) Tools() []llm.Tool {
	tools := make([]llm.Tool, 0, len(r.skills))
	for _, s := range r.skills {
		tools = append(tools, llm.Tool{
			Type: "function",
			Function: llm.ToolFunction{
				Name:        s.Name(),
				Description: s.Description(),
				Parameters:  s.Parameters(),
			},
		})
	}
	return tools
}

// Execute runs a skill by name with the given parameters.
func (r *Registry) Execute(ctx context.Context, name string, params json.RawMessage) (string, error) {
	s, ok := r.skills[name]
	if !ok {
		return "", fmt.Errorf("unknown skill: %s", name)
	}
	return s.Execute(ctx, params)
}

// Names returns all registered skill names.
func (r *Registry) Names() []string {
	names := make([]string, 0, len(r.skills))
	for name := range r.skills {
		names = append(names, name)
	}
	return names
}
