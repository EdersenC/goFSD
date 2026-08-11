package parkingbatch

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

// PlanFingerprint returns a stable identity for the exact validated plan
// definition. It intentionally uses raw poses and variation definitions so the
// result does not depend on cross-runtime trigonometric rounding.
func PlanFingerprint(plan Plan) string {
	writer := planFingerprintWriter{}
	writer.writeString("parking-plan-fingerprint-v1")
	writer.writeString(strings.TrimSpace(plan.ID))
	writer.writeString(strings.TrimSpace(plan.Seed))
	writer.writeUint32(uint32(len(plan.Entries)))
	for _, entry := range plan.Entries {
		writer.writeString(strings.TrimSpace(entry.ID))
		writer.writePose(entry.ParkDest)
		writer.writePose(entry.StartDest)
		writer.writeUint32(uint32(entry.CollectionAmount))
		variations := entry.Variations
		if len(variations) == 0 {
			variations = []Variation{{ID: "base"}}
		}
		writer.writeUint32(uint32(len(variations)))
		for _, variation := range variations {
			writer.writeString(strings.TrimSpace(variation.ID))
			writer.writeOptionalPose(variation.ParkDest)
			writer.writeOptionalPose(variation.StartDest)
			writer.writeBool(variation.StartVariation != nil)
			if variation.StartVariation != nil {
				writer.writeFloat64(variation.StartVariation.DistanceDeltaM)
				writer.writeFloat64(variation.StartVariation.LateralDeltaM)
				writer.writeFloat64(variation.StartVariation.HeadingDeltaDeg)
			}
		}
	}
	sum := sha256.Sum256(writer.bytes)
	return fmt.Sprintf("sha256:%x", sum)
}

type planFingerprintWriter struct {
	bytes []byte
}

func (writer *planFingerprintWriter) writeUint32(value uint32) {
	var encoded [4]byte
	binary.BigEndian.PutUint32(encoded[:], value)
	writer.bytes = append(writer.bytes, encoded[:]...)
}

func (writer *planFingerprintWriter) writeBool(value bool) {
	if value {
		writer.bytes = append(writer.bytes, 1)
		return
	}
	writer.bytes = append(writer.bytes, 0)
}

func (writer *planFingerprintWriter) writeString(value string) {
	encoded := []byte(value)
	writer.writeUint32(uint32(len(encoded)))
	writer.bytes = append(writer.bytes, encoded...)
}

func (writer *planFingerprintWriter) writeFloat64(value float64) {
	if value == 0 {
		value = 0
	}
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], math.Float64bits(value))
	writer.bytes = append(writer.bytes, encoded[:]...)
}

func (writer *planFingerprintWriter) writePose(pose Pose) {
	writer.writeFloat64(pose.X)
	writer.writeFloat64(pose.Y)
	writer.writeFloat64(pose.Z)
	writer.writeFloat64(pose.Heading)
}

func (writer *planFingerprintWriter) writeOptionalPose(pose *Pose) {
	writer.writeBool(pose != nil)
	if pose != nil {
		writer.writePose(*pose)
	}
}
