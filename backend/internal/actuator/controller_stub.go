//go:build !windows

package actuator

func newController() (controller, error) {
	return nil, ErrUnsupportedPlatform
}
