package stopsigncontrol

type boundedPI struct {
	kp          float64
	ki          float64
	integralMin float64
	integralMax float64
	outputMin   float64
	outputMax   float64
	integral    float64
}

func (c *boundedPI) step(err, dtSeconds float64) float64 {
	candidateIntegral := clamp(c.integral+err*dtSeconds, c.integralMin, c.integralMax)
	unsaturated := c.kp*err + c.ki*candidateIntegral
	output := clamp(unsaturated, c.outputMin, c.outputMax)
	pushesUpperLimit := unsaturated > c.outputMax && err > 0
	pushesLowerLimit := unsaturated < c.outputMin && err < 0
	if !pushesUpperLimit && !pushesLowerLimit {
		c.integral = candidateIntegral
	}
	return output
}

func (c *boundedPI) reset() {
	c.integral = 0
}
