package parkingcontrol

import "testing"

func TestBoundedPIRejectsIntegralWindupAtOutputLimits(t *testing.T) {
	controller := boundedPI{
		ki:          1,
		integralMin: -10,
		integralMax: 10,
		outputMin:   -0.5,
		outputMax:   0.5,
	}

	for range 10 {
		if got := controller.step(10, 1); got != 0.5 {
			t.Fatalf("expected saturated positive output, got=%f", got)
		}
	}
	if controller.integral != 0 {
		t.Fatalf("expected saturation to reject windup, integral=%f", controller.integral)
	}
	if got := controller.step(-10, 1); got != -0.5 {
		t.Fatalf("expected immediate braking authority after saturation, got=%f", got)
	}
	if controller.integral != 0 {
		t.Fatalf("expected lower saturation to reject windup, integral=%f", controller.integral)
	}

	if got := controller.step(0.1, 1); got != 0.1 {
		t.Fatalf("expected unsaturated error to integrate, got=%f", got)
	}
	if controller.integral != 0.1 {
		t.Fatalf("unexpected committed integral: %f", controller.integral)
	}
}
