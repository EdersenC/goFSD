import {Box, Card, CardContent, Chip, LinearProgress, Stack, Typography} from "@mui/material";
import type {ControlState, RuntimeTelemetry} from "../types";
import {metersPerSecondToMph} from "../stop-sign-plan";

export function TelemetryPanel({control}: {control?: ControlState}) {
    const telemetry = control?.telemetry;
    const runtime = control?.runtime;
    const linked = Boolean(runtime?.fivemConnected);
    const synchronized = linked && runtime?.appliedSafetyEpoch === control?.safetyEpoch;
    const phase = telemetry?.stopSignPhase ?? runtime?.stopSignBatch?.phase;
    const stopConfirmation = stopConfirmationPercent(telemetry);
    return (
        <Card component="section" aria-labelledby="telemetry-title">
            <CardContent>
                <Stack direction="row" sx={{alignItems: "center", justifyContent: "space-between", gap: 1}}>
                    <Typography id="telemetry-title" variant="h2">Live telemetry</Typography>
                    <Chip
                        size="small"
                        color={synchronized ? "success" : linked ? "warning" : "default"}
                        label={synchronized ? "FiveM linked" : linked ? "Restart FSD" : "FiveM offline"}
                    />
                </Stack>
                <Box sx={{display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 1, mt: 2}}>
                    <Metric label="Phase" value={phase || "idle"} accent />
                    <Metric label="Speed" value={formatSpeed(telemetry?.currentSpeed)} />
                    <Metric label="Throttle" value={format(telemetry?.throttleApplied, "")} />
                    <Metric label="Brake" value={format(telemetry?.brakeApplied ?? telemetry?.brakePressureAvg, "")} />
                    <Metric label="Sign" value={format(telemetry?.stopSignDistanceM, "m")} />
                    <Metric label="Stop line" value={format(telemetry?.stopLineDistanceM, "m")} />
                    <Metric label="Lateral" value={format(telemetry?.stopSignLateralErrorM, "m")} />
                    <Metric label="Heading" value={format(telemetry?.stopSignHeadingErrorDeg, "°")} />
                </Box>
                <Typography variant="overline" color="text.secondary" sx={{display: "block", mt: 2}}>Geometry</Typography>
                <Stack sx={{gap: .5}}>
                    <PoseRow label="Sign" pose={telemetry?.stopSignPose} />
                    <PoseRow label="Stop line" pose={telemetry?.stopLinePose} />
                    <PoseRow label="Target ego center" pose={telemetry?.stopSignEgoStopPose} />
                    <PoseRow label="Live ego center" pose={telemetry && telemetry.positionX !== undefined && telemetry.positionY !== undefined && telemetry.positionZ !== undefined ? {
                        x: telemetry.positionX,
                        y: telemetry.positionY,
                        z: telemetry.positionZ,
                        heading: telemetry.currentYaw,
                    } : undefined} />
                </Stack>
                <Box sx={{mt: 2}}>
                    <Stack direction="row" sx={{justifyContent: "space-between", mb: .7}}>
                        <Typography variant="caption" color="text.secondary">Stop confirmation</Typography>
                        <Typography variant="caption" sx={{fontFamily: "ui-monospace, monospace"}}>
                            {Math.round(telemetry?.stopSignConfirmationElapsedMs ?? 0)} / {Math.round(telemetry?.stopSignConfirmationTargetMs ?? 0)} ms
                        </Typography>
                    </Stack>
                    <LinearProgress variant="determinate" value={stopConfirmation} color={telemetry?.stopSignStopped ? "success" : "primary"} />
                </Box>
                <Stack direction="row" sx={{gap: 1, mt: 2, flexWrap: "wrap"}}>
                    <Chip size="small" variant="outlined" label={`Attempt ${telemetry?.stopSignAttemptIndex ?? runtime?.stopSignBatch?.attemptIndex ?? 0}/${telemetry?.stopSignAttemptCount ?? runtime?.stopSignBatch?.attemptCount ?? 0}`} />
                    <Chip size="small" variant="outlined" label={telemetry?.vehicleExists ? "vehicle ready" : "no vehicle"} />
                    <Chip size="small" variant="outlined" label={telemetry?.stopSignTargetConfigured ? "sign calibrated" : "sign not calibrated"} />
                </Stack>
            </CardContent>
        </Card>
    );
}

function PoseRow({label, pose}: {label: string, pose?: {x: number, y: number, z: number, heading: number}}) {
    const value = pose
        ? `${pose.x.toFixed(2)}, ${pose.y.toFixed(2)}, ${pose.z.toFixed(2)} @ ${pose.heading.toFixed(1)}°`
        : "—";
    return (
        <Stack direction="row" sx={{justifyContent: "space-between", gap: 1, fontFamily: "ui-monospace, monospace"}}>
            <Typography variant="caption" color="text.secondary">{label}</Typography>
            <Typography variant="caption" sx={{textAlign: "right"}}>{value}</Typography>
        </Stack>
    );
}

function Metric({label, value, accent = false}: {label: string, value: string, accent?: boolean}) {
    return (
        <Box sx={{bgcolor: "rgba(255,255,255,.025)", border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.15}}>
            <Typography variant="caption" color="text.secondary">{label}</Typography>
            <Typography sx={{fontWeight: 760, fontFamily: "ui-monospace, monospace", color: accent ? "primary.main" : undefined, overflow: "hidden", textOverflow: "ellipsis"}}>
                {value}
            </Typography>
        </Box>
    );
}

function stopConfirmationPercent(telemetry?: RuntimeTelemetry): number {
    const target = telemetry?.stopSignConfirmationTargetMs ?? 0;
    return target > 0 ? Math.min(100, Math.max(0, ((telemetry?.stopSignConfirmationElapsedMs ?? 0) / target) * 100)) : 0;
}

function format(value: number | undefined, suffix: string): string {
    return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(2)}${suffix}` : "—";
}

function formatSpeed(value: number | undefined): string {
    return typeof value === "number" && Number.isFinite(value)
        ? `${metersPerSecondToMph(value).toFixed(1)} mph · ${value.toFixed(2)} m/s`
        : "—";
}
