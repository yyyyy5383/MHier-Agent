package orchestrator

import (
	"context"
	"fmt"
	"path/filepath"
	"time"
)

func (m *Manager) worker(workerID int) {
	for job := range m.queue {
		if m.skipCanceled(job) {
			continue
		}
		m.runJob(workerID, job)
	}
}

func (m *Manager) skipCanceled(job *Job) bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return job.Status == StatusCanceled
}

func (m *Manager) runJob(workerID int, job *Job) {
	ctx, cancel := context.WithCancel(context.Background())
	m.mu.Lock()
	m.cancels[job.ID] = cancel
	m.mu.Unlock()
	defer cancel()

	m.publish(Event{
		JobID:     job.ID,
		Event:     "worker_assigned",
		Status:    StatusRunning,
		Message:   fmt.Sprintf("Worker %d assigned", workerID),
		Timestamp: time.Now(),
	})

	if err := m.stageSamples(job); err != nil {
		m.finish(job, StatusFailed, StepNone, err.Error(), nil)
		return
	}

	if boolValue(job.Request.RunPrepare, true) {
		if !m.runStep(ctx, job, StepPrepare, []string{"prepare_data_mhier.py"}) {
			return
		}
	}
	if boolValue(job.Request.RunRetrieve, true) {
		if !m.runStep(ctx, job, StepRetrieve, []string{
			filepath.Join("MDocAgent", "scripts", "retrieve.py"),
			"retrieval=mhier",
			"dataset.name=mhier",
			m.datasetArg(job),
		}) {
			return
		}
	}
	if boolValue(job.Request.RunPredict, true) {
		if !m.runStep(ctx, job, StepPredict, []string{
			filepath.Join("MDocAgent", "scripts", "predict.py"),
			"retrieval=mhier",
			"dataset.name=mhier",
			m.datasetArg(job),
			"run-name=" + job.ID,
		}) {
			return
		}
	}

	m.collectOutputs(job)
	m.finish(job, StatusSucceeded, StepFinalize, "", nil)
}

func (m *Manager) runStep(parent context.Context, job *Job, step StepName, args []string) bool {
	timeout := m.cfg.DefaultStepTimeout
	if job.Request.TimeoutSeconds > 0 {
		timeout = time.Duration(job.Request.TimeoutSeconds) * time.Second
	}

	stepCtx, cancel := context.WithTimeout(parent, timeout)
	defer cancel()

	m.setRunning(job, step, fmt.Sprintf("Starting %s", step))
	exitCode, err := m.runProcess(stepCtx, job, step, args)
	if err == nil {
		m.publish(Event{JobID: job.ID, Event: "step_finished", Step: step, Status: StatusRunning, Message: fmt.Sprintf("%s finished", step), ExitCode: &exitCode, Timestamp: time.Now()})
		return true
	}

	status := StatusFailed
	if parent.Err() == context.Canceled {
		status = StatusCanceled
	} else if stepCtx.Err() == context.DeadlineExceeded {
		status = StatusTimeout
	}
	m.collectOutputs(job)
	m.finish(job, status, step, err.Error(), &exitCode)
	return false
}
