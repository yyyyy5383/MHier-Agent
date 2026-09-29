//go:build !windows

package orchestrator

import (
	"os/exec"
	"syscall"
	"time"
)

func prepareProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func killProcessTree(pid int, grace time.Duration) {
	if pid <= 0 {
		return
	}
	_ = syscall.Kill(-pid, syscall.SIGTERM)
	time.Sleep(grace)
	_ = syscall.Kill(-pid, syscall.SIGKILL)
}
