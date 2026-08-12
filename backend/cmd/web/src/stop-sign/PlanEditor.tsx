import CheckRounded from "@mui/icons-material/CheckRounded";
import DeleteOutlineRounded from "@mui/icons-material/DeleteOutlineRounded";
import MyLocationRounded from "@mui/icons-material/MyLocationRounded";
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    Chip,
    IconButton,
    Slider,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import ExpandMoreRounded from "@mui/icons-material/ExpandMoreRounded";
import {useMemo} from "react";
import {
    automaticTargetSpeedRange,
    coupledVariantStartDistanceM,
    currentSceneStep,
    createStopSignSeed,
    formatSpeed,
    MAXIMUM_MOTION_VARIANCE_PCT,
    MAXIMUM_TARGET_SPEED_MPS,
    MINIMUM_AUTO_TARGET_SPEED_MPS,
    MINIMUM_VARIANT_COUNT,
    metersPerSecondToMph,
    poseSummary,
    requiredRollingStartDistanceM,
    ScenePoseField,
    stopSignPlanStats,
    validateStopSignEntry,
} from "../stop-sign-plan";
import type {StopSignPlan, StopSignPlanEntry} from "../types";

type Props = {
    plan: StopSignPlan
    activeEntryIndex: number
    captureReady: boolean
    captureBusy: boolean
    onChange: (plan: StopSignPlan) => void
    onSelectEntry: (index: number) => void
    onCapture: (field: ScenePoseField) => void
    onRemove: (index: number) => void
};

const points: Array<{field: ScenePoseField, number: string, title: string, detail: string}> = [
    {field: "startPose", number: "1", title: "Start", detail: "Park where every replay should begin."},
    {field: "egoStopPose", number: "2", title: "Stop", detail: "Park at the exact vehicle-center stopping point."},
    {field: "exitPose", number: "3", title: "End", detail: "Park beyond the sign where the continuous run should finish."},
];

export function PlanEditor({
    plan,
    activeEntryIndex,
    captureReady,
    captureBusy,
    onChange,
    onSelectEntry,
    onCapture,
    onRemove,
}: Props) {
    const entry = plan.entries[activeEntryIndex];
    const stats = useMemo(() => stopSignPlanStats(plan), [plan]);
    const errors = entry ? validateStopSignEntry(entry) : ["Teleport to a catalog stop sign first."];
    const step = currentSceneStep(entry);
    const speedRange = entry ? automaticTargetSpeedRange() : null;
    const capturedStartDistanceM = entry?.startPose && entry.egoStopPose
        ? Math.hypot(entry.startPose.x - entry.egoStopPose.x, entry.startPose.y - entry.egoStopPose.y)
        : null;
    const startRange = entry && speedRange && capturedStartDistanceM !== null ? {
        minimumM: coupledVariantStartDistanceM(capturedStartDistanceM, entry.targetSpeedMps, speedRange.minimumMps),
        maximumM: coupledVariantStartDistanceM(capturedStartDistanceM, entry.targetSpeedMps, speedRange.maximumMps),
    } : null;

    const updateEntry = (next: StopSignPlanEntry) => {
        const entries = [...plan.entries];
        entries[activeEntryIndex] = next;
        onChange({...plan, entries});
    };

    return (
        <Card component="section" aria-labelledby="scene-builder-title">
            <CardContent>
                <Stack direction={{xs: "column", md: "row"}} sx={{justifyContent: "space-between", gap: 1.5}}>
                    <Box>
                        <Typography variant="overline" color="primary.main">Scene builder</Typography>
                        <Typography id="scene-builder-title" variant="h2">Capture Start → Stop → End</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{mt: .5}}>
                            Move the setup car to each point and capture it. The three positions save automatically under this GTA sign ID.
                        </Typography>
                    </Box>
                    <Stack direction="row" sx={{gap: 2}}>
                        <Metric value={stats.signCount} label="scenes" />
                        <Metric value={entry?.autoVariations?.count ?? 0} label="variants here" />
                    </Stack>
                </Stack>

                {plan.entries.length > 0 && (
                    <Stack direction="row" sx={{gap: .75, overflowX: "auto", py: 1.5}}>
                        {plan.entries.map((scene, index) => (
                            <Chip
                                key={`${scene.id}-${index}`}
                                clickable
                                color={index === activeEntryIndex ? "primary" : "default"}
                                variant={index === activeEntryIndex ? "filled" : "outlined"}
                                label={`${scene.catalogId ?? scene.id}${currentSceneStep(scene) === 3 ? " ✓" : ""}`}
                                onClick={() => onSelectEntry(index)}
                            />
                        ))}
                    </Stack>
                )}

                {!entry ? (
                    <Alert severity="info" sx={{mt: 1}}>Teleport to preview a catalog sign, then choose Use this stop sign to create its reusable setup.</Alert>
                ) : (
                    <>
                        <Stack direction="row" sx={{justifyContent: "space-between", alignItems: "center", mb: 1}}>
                            <Box>
                                <Typography sx={{fontWeight: 850}}>{entry.catalogId}</Typography>
                                <Typography variant="caption" color="text.secondary">
                                    {entry.catalogPosition ? `${entry.catalogPosition.x.toFixed(1)}, ${entry.catalogPosition.y.toFixed(1)}, ${entry.catalogPosition.z.toFixed(1)}` : "Catalog position unavailable"}
                                </Typography>
                            </Box>
                            <IconButton aria-label={`Delete ${entry.id}`} onClick={() => onRemove(activeEntryIndex)}>
                                <DeleteOutlineRounded />
                            </IconButton>
                        </Stack>

                        <Box sx={{display: "grid", gridTemplateColumns: {xs: "1fr", md: "repeat(3, 1fr)"}, gap: 1}}>
                            {points.map((point, index) => {
                                const pose = entry[point.field];
                                const active = step === index;
                                return (
                                    <Box
                                        key={point.field}
                                        data-scene-point={point.title.toLowerCase()}
                                        sx={{
                                            border: "1px solid",
                                            borderColor: pose ? "success.main" : active ? "primary.main" : "divider",
                                            borderRadius: 1.5,
                                            p: 1.5,
                                            bgcolor: active ? "rgba(233,255,91,.045)" : "transparent",
                                        }}
                                    >
                                        <Stack direction="row" sx={{justifyContent: "space-between", alignItems: "center"}}>
                                            <Typography variant="overline" color={active ? "primary.main" : "text.secondary"}>
                                                {point.number} · {point.title}
                                            </Typography>
                                            {pose && <CheckRounded color="success" fontSize="small" />}
                                        </Stack>
                                        <Typography variant="caption" color="text.secondary" sx={{display: "block", minHeight: 36}}>
                                            {point.detail}
                                        </Typography>
                                        <Typography sx={{fontFamily: "ui-monospace, monospace", fontSize: ".76rem", my: 1}}>
                                            {poseSummary(pose)}
                                        </Typography>
                                        <Button
                                            variant={active ? "contained" : "outlined"}
                                            size="small"
                                            startIcon={<MyLocationRounded />}
                                            disabled={!captureReady || captureBusy || (index > step && !pose)}
                                            onClick={() => onCapture(point.field)}
                                            fullWidth
                                        >
                                            {pose ? "Recapture" : `Capture ${point.title}`}
                                        </Button>
                                    </Box>
                                );
                            })}
                        </Box>

                        {errors.length > 0 && <Alert severity="warning" sx={{mt: 1.5}}>{errors.slice(0, 2).join(" ")}</Alert>}

                        <Box sx={{mt: 2, border: "1px solid", borderColor: "divider", borderRadius: 1.5, p: 1.5}}>
                            <Stack direction={{xs: "column", sm: "row"}} sx={{justifyContent: "space-between", gap: 2}}>
                                <Box>
                                    <Typography sx={{fontWeight: 850}}>Automatic seeded variants</Typography>
                                    <Typography variant="body2" color="text.secondary">
                                        Target speed spans 10–15 m/s (22.4–33.6 mph) across the seeded runs. Faster runs start farther back while preserving the captured cruise buffer. Weather, clock time, color, and route conditions vary too; Stop stays exact.
                                    </Typography>
                                </Box>
                                <Stack direction="row" sx={{gap: .75, flexWrap: "wrap", justifyContent: "flex-end"}}>
                                    <Chip color="secondary" label={`${entry.autoVariations?.count ?? 0} continuous runs · 3 labeled stages each`} />
                                    {speedRange && <Chip
                                        color="primary"
                                        variant="outlined"
                                        label={`${metersPerSecondToMph(speedRange.minimumMps).toFixed(1)}–${metersPerSecondToMph(speedRange.maximumMps).toFixed(1)} mph targets`}
                                    />}
                                    {startRange && <Chip
                                        variant="outlined"
                                        label={`${startRange.minimumM.toFixed(0)}–${startRange.maximumM.toFixed(0)} m generated Starts`}
                                    />}
                                </Stack>
                            </Stack>
                            <Box sx={{display: "grid", gridTemplateColumns: {xs: "1fr", sm: "1fr 1fr"}, gap: 2, alignItems: "center", mt: 1.5}}>
                                <Box>
                                    <Stack direction="row" sx={{justifyContent: "space-between"}}>
                                        <Typography variant="caption">Variant count</Typography>
                                        <Chip size="small" label={entry.autoVariations?.count ?? 50} />
                                    </Stack>
                                    <Slider
                                        aria-label="Variant count"
                                        min={MINIMUM_VARIANT_COUNT}
                                        max={100}
                                        step={1}
                                        value={entry.autoVariations?.count ?? 50}
                                        onChange={(_event, value) => updateEntry({
                                        ...entry,
                                        autoVariations: {
                                            count: Number(value),
                                            motionVariancePct: entry.autoVariations?.motionVariancePct ?? 20,
                                        },
                                    })}
                                        valueLabelDisplay="auto"
                                    />
                                </Box>
                                <Box>
                                    <Stack direction="row" sx={{justifyContent: "space-between"}}>
                                        <Typography variant="caption">Motion variance</Typography>
                                        <Typography variant="caption" sx={{fontFamily: "ui-monospace, monospace"}}>{entry.autoVariations?.motionVariancePct ?? 20}%</Typography>
                                    </Stack>
                                    <Slider
                                        min={0}
                                        max={MAXIMUM_MOTION_VARIANCE_PCT}
                                        step={1}
                                        value={entry.autoVariations?.motionVariancePct ?? 20}
                                        onChange={(_event, value) => updateEntry({
                                            ...entry,
                                            autoVariations: {count: entry.autoVariations?.count ?? 50, motionVariancePct: Number(value)},
                                        })}
                                        valueLabelDisplay="auto"
                                    />
                                </Box>
                            </Box>
                        </Box>

                        <Accordion disableGutters sx={{mt: 1}}>
                            <AccordionSummary expandIcon={<ExpandMoreRounded />}>
                                <Typography variant="body2" sx={{fontWeight: 750}}>Advanced scene defaults</Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                                <Box sx={{display: "grid", gridTemplateColumns: {xs: "1fr", sm: "1fr 1fr"}, gap: 1}}>
                                    <Stack direction="row" sx={{gap: 1}}>
                                        <TextField label="Library seed" value={plan.seed} onChange={(event) => onChange({...plan, seed: event.target.value})} fullWidth />
                                        <Button variant="outlined" onClick={() => onChange({...plan, seed: createStopSignSeed()})}>New seed</Button>
                                    </Stack>
                                    <Box sx={{px: .5}}>
                                        <Typography variant="caption">Automatic target-speed range</Typography>
                                        <Stack direction="row" sx={{gap: .75, mt: .75, flexWrap: "wrap"}}>
                                            <Chip size="small" color="primary" label={`${metersPerSecondToMph(MINIMUM_AUTO_TARGET_SPEED_MPS).toFixed(1)} mph minimum`} />
                                            <Chip size="small" color="primary" variant="outlined" label={`${metersPerSecondToMph(MAXIMUM_TARGET_SPEED_MPS).toFixed(1)} mph maximum`} />
                                        </Stack>
                                    </Box>
                                </Box>
                                <Typography variant="caption" color="text.secondary" sx={{display: "block", mt: 1}}>
                                    The first run uses the minimum, {formatSpeed(entry.targetSpeedMps)}; the second guarantees the maximum, {formatSpeed(MAXIMUM_TARGET_SPEED_MPS)}; remaining runs sample between them. Start is recalculated from each exact speed. The minimum needs about {requiredRollingStartDistanceM(entry.targetSpeedMps).toFixed(0)} m plus your captured cruise buffer. Launch frames stay raw; training begins after one stable cruise second.
                                </Typography>
                            </AccordionDetails>
                        </Accordion>
                    </>
                )}
            </CardContent>
        </Card>
    );
}

function Metric({value, label}: {value: number, label: string}) {
    return (
        <Box sx={{textAlign: "right"}}>
            <Typography sx={{fontWeight: 850, fontSize: "1.15rem", lineHeight: 1}}>{value}</Typography>
            <Typography variant="caption" color="text.secondary">{label}</Typography>
        </Box>
    );
}
