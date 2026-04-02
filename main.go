package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"keepgoing/internal/agent"
	kgctx "keepgoing/internal/context"
	"keepgoing/internal/db"
	"keepgoing/internal/llm"
	"keepgoing/internal/skills"
)

func main() {
	var (
		dbPath   = flag.String("db", "keepgoing.db", "SQLite database path")
		endpoint = flag.String("endpoint", "http://localhost:8000", "LLM server endpoint")
		model    = flag.String("model", "", "Model name (default: auto-detected)")
		task     = flag.String("task", "", "New task goal (omit to resume latest)")
		workDir  = flag.String("workdir", ".", "Working directory")
		target   = flag.Int("target", 0, "Minimum findings before task can complete (0 = no minimum)")
	)
	flag.Parse()

	log.SetFlags(log.Ldate | log.Ltime | log.Lmsgprefix)
	log.SetPrefix("[keepgoing] ")

	// Open database
	database, err := db.Open(*dbPath)
	if err != nil {
		log.Fatalf("Failed to open database: %v", err)
	}
	defer database.Close()

	// Create LLM client
	client := llm.NewClient(*endpoint, *model)

	// Check model server connectivity
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := client.Ping(ctx); err != nil {
		log.Printf("WARNING: %v — agent will retry on each call", err)
	} else {
		log.Printf("Model server connected: %s", client.Model())
	}

	// Set up progress directory
	progressDir := filepath.Join(*workDir, ".keepgoing")
	os.MkdirAll(progressDir, 0755)

	// Register skills
	registry := skills.NewRegistry()
	registry.Register(&skills.BashSkill{})
	// Firecrawl skills omitted — not installed. Agent uses bash + curl instead.

	// Context manager
	ctxMgr := kgctx.NewManager(database, client)

	// Determine task ID
	var taskID int64
	if *task != "" {
		// Create new root task
		id, err := database.CreateTask(nil, *task)
		if err != nil {
			log.Fatalf("Failed to create task: %v", err)
		}
		taskID = id
		log.Printf("Created new task %d: %s", taskID, *task)
	} else {
		// Resume latest unfinished root task
		t, err := database.GetLatestRootTask()
		if err != nil {
			log.Fatalf("Failed to get latest task: %v", err)
		}
		if t == nil {
			fmt.Println("No active tasks. Create one with: keepgoing -task \"your goal here\"")
			os.Exit(0)
		}
		taskID = t.ID
		log.Printf("Resuming task %d: %s", taskID, t.Goal)
	}

	// Register task-specific skills
	registry.Register(&skills.SaveFindingSkill{DB: database, TaskID: taskID})

	// Handle graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Printf("Received %v, shutting down gracefully...", sig)
		cancel()
	}()

	// Create and run agent
	a := agent.New(agent.Config{
		DB:             database,
		LLM:            client,
		Skills:         registry,
		CtxManager:     ctxMgr,
		TaskID:         taskID,
		WorkDir:        *workDir,
		ProgressDir:    progressDir,
		TargetFindings: *target,
	})

	if err := a.Run(ctx); err != nil {
		if ctx.Err() != nil {
			log.Printf("Agent stopped: %v", err)
			os.Exit(0)
		}
		log.Fatalf("Agent failed: %v", err)
	}

	log.Println("Agent completed successfully.")
}
