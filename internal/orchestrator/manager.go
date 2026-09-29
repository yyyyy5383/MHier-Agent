package orchestrator

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type Manager struct {
	cfg Config

	mu          sync.RWMutex
	jobs        map[string]*Job
	cancels     map[string]context.CancelFunc
	subscribers map[string]map[chan Event]struct{}
	queue       chan *Job
}

func NewManager(cfg Config) (*Manager, error) {
	if cfg.MaxWorkers <= 0 {
		cfg.MaxWorkers = 1
	}
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 32
	}
	if cfg.DefaultStepTimeout <= 0 {
		cfg.DefaultStepTimeout = 30 * time.Minute
	}
	if cfg.KillGrace <= 0 {
		cfg.KillGrace = 10 * time.Second
	}
	root, err := filepath.Abs(cfg.RepoRoot)
	if err != nil {
		return nil, err
	}
	cfg.RepoRoot = root

	return &Manager{
		cfg:         cfg,
		jobs:        map[string]*Job{},
		cancels:     map[string]context.CancelFunc{},
		subscribers: map[string]map[chan Event]struct{}{},
		queue:       make(chan *Job, cfg.QueueSize),
	}, nil
}

func (m *Manager) Start() {
	for i := 0; i < m.cfg.MaxWorkers; i++ {
		go m.worker(i + 1)
	}
}

func (m *Manager) Submit(req JobRequest) (*Job, error) {
	if err := validateRequest(req); err != nil {
		return nil, err
	}

	id := m.newJobID(req.PDFPath)
	artifactDir := filepath.Join(m.cfg.RepoRoot, "outputs", "jobs", id)
	job := &Job{
		ID:          id,
		Status:      StatusQueued,
		CurrentStep: StepNone,
		Request:     req,
		ArtifactDir: artifactDir,
		CreatedAt:   time.Now(),
	}

	if err := ensureDir(artifactDir); err != nil {
		return nil, err
	}
	if err := writeJSON(filepath.Join(artifactDir, "request.json"), req); err != nil {
		return nil, err
	}
	if err := writeJSON(filepath.Join(artifactDir, "status.json"), job); err != nil {
		return nil, err
	}

	m.mu.Lock()
	m.jobs[id] = job
	m.mu.Unlock()

	select {
	case m.queue <- job:
		m.publish(Event{JobID: id, Event: "job_queued", Status: StatusQueued, Message: "Job queued", Timestamp: time.Now()})
		return job, nil
	default:
		m.mu.Lock()
		delete(m.jobs, id)
		m.mu.Unlock()
		return nil, fmt.Errorf("job queue is full")
	}
}

func (m *Manager) Get(id string) (*Job, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	job, ok := m.jobs[id]
	if !ok {
		return nil, false
	}
	cp := *job
	return &cp, true
}

func (m *Manager) Cancel(id string) bool {
	m.mu.Lock()
	job, ok := m.jobs[id]
	if !ok {
		m.mu.Unlock()
		return false
	}
	if job.Status == StatusQueued {
		now := time.Now()
		job.Status = StatusCanceled
		job.CurrentStep = StepFinalize
		job.FinishedAt = &now
		job.Error = "job canceled before execution"
		_ = writeJSON(filepath.Join(job.ArtifactDir, "status.json"), job)
		m.mu.Unlock()
		m.publish(Event{JobID: id, Event: "job_canceled", Status: StatusCanceled, Message: job.Error, Timestamp: time.Now()})
		return true
	}
	cancel := m.cancels[id]
	m.mu.Unlock()

	if cancel != nil {
		cancel()
		return true
	}
	return false
}

func (m *Manager) Subscribe(jobID string) (chan Event, func(), bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.jobs[jobID]; !ok {
		return nil, nil, false
	}
	ch := make(chan Event, 64)
	if m.subscribers[jobID] == nil {
		m.subscribers[jobID] = map[chan Event]struct{}{}
	}
	m.subscribers[jobID][ch] = struct{}{}
	cleanup := func() {
		m.mu.Lock()
		defer m.mu.Unlock()
		delete(m.subscribers[jobID], ch)
		close(ch)
	}
	return ch, cleanup, true
}

func (m *Manager) publish(event Event) {
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}
	_ = appendJSONL(filepath.Join(m.cfg.RepoRoot, "outputs", "jobs", event.JobID, "events.jsonl"), event)

	m.mu.RLock()
	defer m.mu.RUnlock()
	for ch := range m.subscribers[event.JobID] {
		select {
		case ch <- event:
		default:
		}
	}
}

func (m *Manager) setRunning(job *Job, step StepName, message string) {
	now := time.Now()
	m.mu.Lock()
	if job.StartedAt == nil {
		job.StartedAt = &now
	}
	job.Status = StatusRunning
	job.CurrentStep = step
	_ = writeJSON(filepath.Join(job.ArtifactDir, "status.json"), job)
	m.mu.Unlock()
	m.publish(Event{JobID: job.ID, Event: "step_started", Step: step, Status: StatusRunning, Message: message, Timestamp: now})
}

func (m *Manager) finish(job *Job, status JobStatus, step StepName, errText string, exitCode *int) {
	now := time.Now()
	m.mu.Lock()
	job.Status = status
	job.CurrentStep = StepFinalize
	job.FinishedAt = &now
	if job.StartedAt != nil {
		job.Duration = now.Sub(*job.StartedAt)
	}
	job.Error = errText
	if status == StatusSucceeded {
		job.FailedStep = StepNone
		job.ExitCode = nil
	} else {
		job.FailedStep = step
		job.ExitCode = exitCode
	}
	delete(m.cancels, job.ID)
	_ = writeJSON(filepath.Join(job.ArtifactDir, "status.json"), job)
	m.mu.Unlock()
	m.publish(Event{JobID: job.ID, Event: "job_finished", Step: step, Status: status, Message: errText, ExitCode: exitCode, Timestamp: now})
}

func (m *Manager) newJobID(pdfPath string) string {
	random := make([]byte, 3)
	_, _ = rand.Read(random)
	stem := safeIDPart(filepath.Base(pdfPath))
	return fmt.Sprintf("job_%s_%s_%s", time.Now().Format("20060102_150405"), stem, hex.EncodeToString(random))
}

func boolValue(value *bool, fallback bool) bool {
	if value == nil {
		return fallback
	}
	return *value
}

func (m *Manager) stageSamples(job *Job) error {
	samples := samplesForRequest(job.Request)
	jobSamplesPath := filepath.Join(job.ArtifactDir, "samples.json")
	return writeJSON(jobSamplesPath, samples)
}

func (m *Manager) collectOutputs(job *Job) {
	dataDir := job.ArtifactDir
	_ = copyIfExists(filepath.Join(dataDir, "sample-with-retrieval-results.json"), filepath.Join(job.ArtifactDir, "sample-with-retrieval-results.json"))

	resultsDir := filepath.Join(m.cfg.RepoRoot, "results", "mhier", job.ID)
	if fileExists(resultsDir) {
		_ = filepath.WalkDir(resultsDir, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return nil
			}
			if filepath.Ext(path) == ".json" || filepath.Ext(path) == ".html" {
				rel, relErr := filepath.Rel(resultsDir, path)
				if relErr == nil {
					_ = copyIfExists(path, filepath.Join(job.ArtifactDir, "results", rel))
				}
			}
			return nil
		})
	}
}

func (m *Manager) datasetArg(job *Job) string {
	return "dataset.data_dir=./" + filepath.ToSlash(filepath.Join("outputs", "jobs", job.ID))
}
