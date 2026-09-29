package main

import (
	"flag"
	"log"
	"net/http"
	"path/filepath"
	"runtime"
	"time"

	"mhier-agent/internal/orchestrator"
)

func defaultPythonBin() string {
	if runtime.GOOS == "windows" {
		return filepath.Join(".venv", "Scripts", "python.exe")
	}
	return filepath.Join(".venv", "bin", "python")
}

func main() {
	addr := flag.String("addr", ":8090", "HTTP listen address")
	repoRoot := flag.String("repo-root", ".", "MHier-Agent repository root")
	pythonBin := flag.String("python", defaultPythonBin(), "Python executable used to run MHier scripts")
	maxWorkers := flag.Int("workers", 1, "Number of concurrent workers")
	queueSize := flag.Int("queue-size", 32, "Maximum queued jobs")
	stepTimeout := flag.Duration("step-timeout", 30*time.Minute, "Default timeout for each pipeline step")
	killGrace := flag.Duration("kill-grace", 10*time.Second, "Grace period before force killing a process tree")
	flag.Parse()

	cfg := orchestrator.Config{
		Addr:               *addr,
		RepoRoot:           *repoRoot,
		PythonBin:          *pythonBin,
		MaxWorkers:         *maxWorkers,
		QueueSize:          *queueSize,
		DefaultStepTimeout: *stepTimeout,
		KillGrace:          *killGrace,
	}

	manager, err := orchestrator.NewManager(cfg)
	if err != nil {
		log.Fatalf("create orchestrator: %v", err)
	}
	manager.Start()

	server := &http.Server{
		Addr:              cfg.Addr,
		Handler:           orchestrator.NewServer(manager),
		ReadHeaderTimeout: 10 * time.Second,
	}

	log.Printf("MHier-Agent orchestrator listening on %s", cfg.Addr)
	log.Fatal(server.ListenAndServe())
}
