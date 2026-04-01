package skills

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"time"
)

const firecrawlTimeout = 120 * time.Second

// FirecrawlScrapeSkill scrapes a URL and returns its content as markdown.
type FirecrawlScrapeSkill struct{}

type scrapeParams struct {
	URL string `json:"url"`
}

func (f *FirecrawlScrapeSkill) Name() string { return "firecrawl_scrape" }

func (f *FirecrawlScrapeSkill) Description() string {
	return "Scrape a web page and return its content as clean markdown. Use for reading articles, company pages, LinkedIn profiles, and any web content."
}

func (f *FirecrawlScrapeSkill) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"url": {
				"type": "string",
				"description": "The URL to scrape"
			}
		},
		"required": ["url"]
	}`)
}

func (f *FirecrawlScrapeSkill) Execute(ctx context.Context, params json.RawMessage) (string, error) {
	var p scrapeParams
	if err := json.Unmarshal(params, &p); err != nil {
		return "", fmt.Errorf("parse params: %w", err)
	}
	if p.URL == "" {
		return "", fmt.Errorf("empty url")
	}

	cmdCtx, cancel := context.WithTimeout(ctx, firecrawlTimeout)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, "npx", "firecrawl", "scrape", p.URL)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Sprintf("%s\n[error: %s]", string(output), err.Error()), nil
	}
	return string(output), nil
}

// FirecrawlSearchSkill searches the web and returns results.
type FirecrawlSearchSkill struct{}

type searchParams struct {
	Query string `json:"query"`
}

func (f *FirecrawlSearchSkill) Name() string { return "firecrawl_search" }

func (f *FirecrawlSearchSkill) Description() string {
	return "Search the web for a query and return results with titles, URLs, and snippets. Use for finding companies, people, contact information, and research."
}

func (f *FirecrawlSearchSkill) Parameters() json.RawMessage {
	return json.RawMessage(`{
		"type": "object",
		"properties": {
			"query": {
				"type": "string",
				"description": "The search query"
			}
		},
		"required": ["query"]
	}`)
}

func (f *FirecrawlSearchSkill) Execute(ctx context.Context, params json.RawMessage) (string, error) {
	var p searchParams
	if err := json.Unmarshal(params, &p); err != nil {
		return "", fmt.Errorf("parse params: %w", err)
	}
	if p.Query == "" {
		return "", fmt.Errorf("empty query")
	}

	cmdCtx, cancel := context.WithTimeout(ctx, firecrawlTimeout)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, "npx", "firecrawl", "search", p.Query)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Sprintf("%s\n[error: %s]", string(output), err.Error()), nil
	}
	return string(output), nil
}
