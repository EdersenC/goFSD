//go:build windows

package actuator

import (
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/CB2Moon/vgamepad-go/pkg/commons"
	"github.com/CB2Moon/vgamepad-go/pkg/vgamepad"
)

var ErrViGEmBusNotInstalled = fmt.Errorf(
	"ViGEmBus is not installed. Install the ViGEmBus driver manually, then restart the backend as Administrator",
)

type xbox360Controller struct {
	gamepad *vgamepad.VX360Gamepad
}

func newController() (controller, error) {
	installed, err := hasViGEmBusInstalled()
	if err != nil {
		return nil, err
	}
	if !installed {
		return nil, ErrViGEmBusNotInstalled
	}

	gamepad, err := vgamepad.NewVX360Gamepad()
	if err != nil {
		return nil, err
	}

	gamepad.PressButton(commons.XUSB_GAMEPAD_A)
	if err := gamepad.Update(); err != nil {
		gamepad.Close()
		return nil, err
	}
	time.Sleep(50 * time.Millisecond)
	gamepad.ReleaseButton(commons.XUSB_GAMEPAD_A)
	if err := gamepad.Update(); err != nil {
		gamepad.Close()
		return nil, err
	}

	return &xbox360Controller{gamepad: gamepad}, nil
}

func (c *xbox360Controller) Apply(next controlState) error {
	c.gamepad.LeftJoystickFloat(next.Steer, 0)
	c.gamepad.RightTriggerFloat(next.Throttle)
	c.gamepad.LeftTriggerFloat(next.Brake)

	if next.Handbrake {
		c.gamepad.PressButton(commons.XUSB_GAMEPAD_RIGHT_SHOULDER)
	} else {
		c.gamepad.ReleaseButton(commons.XUSB_GAMEPAD_RIGHT_SHOULDER)
	}

	return c.gamepad.Update()
}

func (c *xbox360Controller) Close() {
	c.gamepad.Reset()
	_ = c.gamepad.Update()
	c.gamepad.Close()
}

func hasViGEmBusInstalled() (bool, error) {
	queries := [][]string{
		{"query", `HKLM\SYSTEM\CurrentControlSet\Services\ViGEmBus`},
		{"query", `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall`, "/s"},
	}

	for index, args := range queries {
		cmd := exec.Command("reg", args...)
		output, err := cmd.CombinedOutput()
		if err != nil {
			if index == 0 {
				continue
			}
			return false, fmt.Errorf("failed to verify ViGEmBus installation: %s", strings.TrimSpace(string(output)))
		}

		lower := strings.ToLower(string(output))
		if strings.Contains(lower, "vigembus") || strings.Contains(lower, "virtual gamepad emulation bus") {
			return true, nil
		}
	}

	return false, nil
}
