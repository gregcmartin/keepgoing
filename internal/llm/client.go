package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	defaultBaseURL = "http://localhost:8000"
	defaultModel   = "nightmedia/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-qx64-hi-mlx"
	requestTimeout = 120 * time.Second
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

// chatResponse is the response body from /v1/chat/completions.
type chatResponse struct {
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
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
	return &Response{
		Content:      choice.Message.Content,
		ToolCalls:    choice.Message.ToolCalls,
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
