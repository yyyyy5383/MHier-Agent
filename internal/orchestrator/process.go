package orchestrator

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

func (m *Manager) runProcess(ctx context.Context, job *Job, step StepName, scriptArgs []string) (int, error) {
	logPath := filepath.Join(job.ArtifactDir, fmt.Sprintf("%s.log", step))
	logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return -1, err
	}
	defer logFile.Close()

	args := append([]string{}, scriptArgs...)
	cmd := exec.Command(m.cfg.PythonBin, args...)
	cmd.Dir = m.cfg.RepoRoot
	cmd.Env = os.Environ()
	prepareProcessGroup(cmd)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return -1, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return -1, err
	}
	if err := cmd.Start(); err != nil {
		return -1, err
	}

	m.publish(Event{
		JobID:     job.ID,
		Event:     "process_started",
		Step:      step,
		Status:    StatusRunning,
		Message:   fmt.Sprintf("pid=%d", cmd.Process.Pid),
		Timestamp: time.Now(),
	})

	var wg sync.WaitGroup
	wg.Add(2)
	go m.pipeLogs(&wg, job, step, "stdout", stdout, logFile)
	go m.pipeLogs(&wg, job, step, "stderr", stderr, logFile)

	waitCh := make(chan error, 1)
	go func() {
		waitCh <- cmd.Wait()
	}()

	var waitErr error
	select {
	case waitErr = <-waitCh:
	case <-ctx.Done():
		killProcessTree(cmd.Process.Pid, m.cfg.KillGrace)
		waitErr = <-waitCh
	}

	wg.Wait()

	exitCode := 0
	if waitErr != nil {
		var exitErr *exec.ExitError
		if errors.As(waitErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = -1
		}
	}

	if ctx.Err() != nil {
		return exitCode, ctx.Err()
	}
	if waitErr != nil {
		return exitCode, waitErr
	}
	return exitCode, nil
}

func (m *Manager) pipeLogs(wg *sync.WaitGroup, job *Job, step StepName, stream string, reader io.Reader, writer io.Writer) {
	defer wg.Done()
	scanner := bufio.NewScanner(reader)
	buffer := make([]byte, 0, 64*1024)
	scanner.Buffer(buffer, 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		_, _ = fmt.Fprintf(writer, "[%s] %s\n", stream, line)
		m.publish(Event{
			JobID:     job.ID,
			Event:     "step_log",
			Step:      step,
			Status:    StatusRunning,
			Stream:    stream,
			Message:   line,
			Timestamp: time.Now(),
		})
	}
}
