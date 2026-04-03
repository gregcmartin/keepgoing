package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

const (
	defaultBaseURL = "http://localhost:8000"
	defaultModel   = "mlx-community/gemma-4-31b-it-4bit"
	requestTimeout = 600 * time.Second // 10 min — local 27B model with reasoning can be slow
	maxRetries     = 3
)

// Client talks to an OpenAI-compatible API (mlx_vlm server).
type Client struct {
	baseURL    string
	model      string
	httpClient *http.Client
}

// Message represents a chat message in OpenAI format.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// ToolCall represents a function call requested by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall contains the function name and arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool defines a function the model can call.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a callable function.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

// Response is what we get back from the model.
type Response struct {
	Content      string
	Reasoning    string // For reasoning models that separate thinking from output
	ToolCalls    []ToolCall
	FinishReason string
	Usage        Usage
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// chatRequest is the request body for /v1/chat/completions.
type chatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
}

// chatResponseMessage extends Message with the reasoning field some models return.
type chatResponseMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	Reasoning  string     `json:"reasoning"`
	ToolCalls  []ToolCall `json:"tool_calls"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// chatResponse is the response body from /v1/chat/completions.
type chatResponse struct {
	Choices []struct {
		Message      chatResponseMessage `json:"message"`
		FinishReason string              `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

func NewClient(baseURL, model string) *Client {
	if baseURL == "" {
		baseURL = defaultBaseURL
	}
	if model == "" {
		model = defaultModel
	}
	return &Client{
		baseURL: baseURL,
		model:   model,
		httpClient: &http.Client{
			Timeout: requestTimeout,
		},
	}
}

// Chat sends messages to the model and returns its response.
func (c *Client) Chat(ctx context.Context, messages []Message, tools []Tool) (*Response, error) {
	reqBody := chatRequest{
		Model:    c.model,
		Messages: messages,
		Tools:    tools,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
			}
		}

		resp, err := c.doRequest(ctx, body)
		if err != nil {
			lastErr = err
			continue
		}
		return resp, nil
	}
	return nil, fmt.Errorf("after %d retries: %w", maxRetries, lastErr)
}

func (c *Client) doRequest(ctx context.Context, body []byte) (*Response, error) {
	url := c.baseURL + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(respBody))
	}

	var chatResp chatResponse
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	choice := chatResp.Choices[0]

	// Some reasoning models put output in 'reasoning' instead of 'content'.
	// Merge them so the agent loop always sees the full response.
	content := choice.Message.Content
	reasoning := choice.Message.Reasoning
	if content == "" && reasoning != "" {
		content = reasoning
	}

	toolCalls := choice.Message.ToolCalls

	// Gemma 4 embeds tool calls in content using special tokens:
	//   <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>
	// Parse these into standard ToolCall structs.
	if len(toolCalls) == 0 && strings.Contains(content, "<|tool_call>") {
		parsed := parseGemmaToolCalls(content)
		if len(parsed) > 0 {
			toolCalls = parsed
			// Remove tool call tokens from content
			content = gemmaToolCallRe.ReplaceAllString(content, "")
			content = strings.TrimSpace(content)
			// Override finish reason since the model made tool calls
			choice.FinishReason = "tool_calls"
		}
	}

	return &Response{
		Content:      content,
		Reasoning:    reasoning,
		ToolCalls:    toolCalls,
		FinishReason: choice.FinishReason,
		Usage:        chatResp.Usage,
	}, nil
}

// Ping checks if the model server is reachable.
func (c *Client) Ping(ctx context.Context) error {
	url := c.baseURL + "/v1/models"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("model server unreachable: %w", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("model server returned status %d", resp.StatusCode)
	}
	return nil
}

// Model returns the configured model name.
func (c *Client) Model() string {
	return c.model
}

// --- Gemma 4 tool call parsing ---

// Matches: <|tool_call>call:func_name{args}<tool_call|>
// (?s) enables dotall mode so . matches newlines (commands often contain \n)
var gemmaToolCallRe = regexp.MustCompile(`(?s)<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>`)

// Matches key-value pairs inside tool call args:
//   key:<|"|>value<|"|>  or  key:value
var gemmaArgRe = regexp.MustCompile(`(\w+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))`)

func parseGemmaToolCalls(content string) []ToolCall {
	matches := gemmaToolCallRe.FindAllStringSubmatch(content, -1)
	if len(matches) == 0 {
		return nil
	}

	var calls []ToolCall
	for i, match := range matches {
		funcName := match[1]
		argsStr := match[2]

		// First, try to convert Gemma's format directly to valid JSON.
		// Replace <|"|> tokens with proper quotes and fix the structure.
		jsonStr := cleanGemmaArgs(argsStr)

		// Validate it's proper JSON
		var testObj interface{}
		if json.Unmarshal([]byte(jsonStr), &testObj) != nil {
			// Fallback: parse key-value pairs manually
			args := make(map[string]interface{})
			argMatches := gemmaArgRe.FindAllStringSubmatch(argsStr, -1)
			for _, am := range argMatches {
				key := am[1]
				value := am[2]
				if value == "" {
					value = strings.TrimSpace(am[3])
				}
				// Clean any remaining Gemma tokens from values
				value = strings.ReplaceAll(value, `<|"|>`, "")
				args[key] = value
			}
			jsonBytes, _ := json.Marshal(args)
			jsonStr = string(jsonBytes)
		}

		calls = append(calls, ToolCall{
			ID:   fmt.Sprintf("gemma-tc-%d", i),
			Type: "function",
			Function: FunctionCall{
				Name:      funcName,
				Arguments: jsonStr,
			},
		})
	}
	return calls
}

// cleanGemmaArgs converts Gemma's key-value format to valid JSON.
// Input:  data:{company:<|"|>Apple<|"|>,name:<|"|>John<|"|>},kind:<|"|>ciso<|"|>
// Output: {"data":{"company":"Apple","name":"John"},"kind":"ciso"}
func cleanGemmaArgs(s string) string {
	// Step 1: Replace Gemma quote tokens with actual quotes
	s = strings.ReplaceAll(s, `<|"|>`, `"`)

	// Step 2: Wrap in braces if needed
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "{") {
		s = "{" + s + "}"
	}

	// Step 3: Quote all bare keys (word followed by : that isn't already quoted)
	// Process repeatedly since nested objects need multiple passes
	quoteKeysRe := regexp.MustCompile(`([{,])\s*([a-zA-Z_]\w*)\s*:`)
	for i := 0; i < 3; i++ {
		prev := s
		s = quoteKeysRe.ReplaceAllString(s, `$1"$2":`)
		if s == prev {
			break
		}
	}

	// Step 4: Quote bare values (unquoted strings between : and , or })
	// Match :somevalue, or :somevalue} but not :{  or :"
	bareValRe := regexp.MustCompile(`:([^"{}\[\],][^,}]*)([,}])`)
	s = bareValRe.ReplaceAllStringFunc(s, func(match string) string {
		// Extract the value part
		colonIdx := strings.Index(match, ":")
		val := match[colonIdx+1 : len(match)-1]
		end := match[len(match)-1:]
		val = strings.TrimSpace(val)
		if val == "" || val == "null" || val == "true" || val == "false" {
			return match
		}
		// Check if it's already quoted
		if strings.HasPrefix(val, `"`) {
			return match
		}
		// Check if it's a number
		if _, err := fmt.Sscanf(val, "%f", new(float64)); err == nil {
			return match
		}
		return `:"` + val + `"` + end
	})

	return s
}
