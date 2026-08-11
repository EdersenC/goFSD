import AddRounded from "@mui/icons-material/AddRounded";
import DeleteOutlineRounded from "@mui/icons-material/DeleteOutlineRounded";
import ExpandMoreRounded from "@mui/icons-material/ExpandMoreRounded";
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
    Divider,
    IconButton,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import {useMemo} from "react";
import {createStopSignEntry, stopSignPlanStats, validateStopSignPlan} from "../stop-sign-plan";
import type {Pose, StopSignPlan, StopSignPlanEntry, StopSignVariation} from "../types";

type Props = {
    plan: StopSignPlan
    onChange: (plan: StopSignPlan) => void
    onCalibrate: () => void
    onClearCalibration: () => void
    onApplyCalibration: (entryIndex: number) => void
    calibrationReady: boolean
    calibrationBusy: boolean
};

export function PlanEditor({
    plan,
    onChange,
    onCalibrate,
    onClearCalibration,
    onApplyCalibration,
    calibrationReady,
    calibrationBusy,
}: Props) {
    const stats = useMemo(() => stopSignPlanStats(plan), [plan]);
    const errors = useMemo(() => validateStopSignPlan(plan), [plan]);

    const updateEntry = (index: number, entry: StopSignPlanEntry) => {
        const entries = [...plan.entries];
        entries[index] = entry;
        onChange({...plan, entries});
    };

    return (
        <Card component="section" aria-labelledby="plan-title">
            <CardContent>
                <Stack direction={{xs: "column", md: "row"}} sx={{justifyContent: "space-between", alignItems: {md: "flex-start"}, gap: 2}}>
                    <Box>
                        <Typography variant="overline" color="primary.main">Collection</Typography>
                        <Typography id="plan-title" variant="h2">Stop-sign plan</Typography>
                    </Box>
                    <Stack direction="row" sx={{gap: 2}}>
                        <Counter value={stats.signCount} label="signs" />
                        <Counter value={stats.variationCount} label="variations" />
                        <Counter value={stats.attemptCount} label="attempts" />
                    </Stack>
                </Stack>

                <Box sx={{display: "grid", gridTemplateColumns: {xs: "1fr", sm: "1fr 1fr"}, gap: 1.2, mt: 2}}>
                    <TextField label="Plan id" value={plan.id} onChange={(event) => onChange({...plan, id: event.target.value})} />
                    <TextField label="Seed" value={plan.seed} onChange={(event) => onChange({...plan, seed: event.target.value})} />
                </Box>

                {errors.length > 0 && <Alert severity="warning" sx={{mt: 1.5}}>{errors.slice(0, 2).join(" ")}</Alert>}

                <Stack direction={{xs: "column", sm: "row"}} sx={{gap: 1, my: 2}}>
                    <Button
                        variant="outlined"
                        color="secondary"
                        startIcon={<MyLocationRounded />}
                        disabled={calibrationBusy}
                        onClick={onCalibrate}
                    >
                        Calibrate current sign
                    </Button>
                    <Button
                        variant="text"
                        color="inherit"
                        disabled={!calibrationReady || calibrationBusy}
                        onClick={onClearCalibration}
                    >
                        Clear live calibration
                    </Button>
                    <Typography variant="caption" color="text.secondary" sx={{alignSelf: "center"}}>
                        Captures the setup-car pose and lane heading. Stop/start poses are derived from distances.
                    </Typography>
                </Stack>

                <Stack sx={{gap: 1}}>
                    {plan.entries.map((entry, index) => (
                        <StopSignEntryEditor
                            key={`${entry.id}-${index}`}
                            entry={entry}
                            index={index}
                            onChange={(next) => updateEntry(index, next)}
                            onRemove={() => onChange({...plan, entries: plan.entries.filter((_, candidate) => candidate !== index)})}
                            onApplyCalibration={() => onApplyCalibration(index)}
                            calibrationReady={calibrationReady}
                            removable={plan.entries.length > 1}
                        />
                    ))}
                </Stack>

                <Button
                    startIcon={<AddRounded />}
                    color="inherit"
                    sx={{mt: 1.5}}
                    onClick={() => onChange({...plan, entries: [...plan.entries, createStopSignEntry(plan.entries.length + 1)]})}
                >
                    Add stop sign
                </Button>
            </CardContent>
        </Card>
    );
}

function StopSignEntryEditor({entry, index, onChange, onRemove, onApplyCalibration, calibrationReady, removable}: {
    entry: StopSignPlanEntry
    index: number
    onChange: (entry: StopSignPlanEntry) => void
    onRemove: () => void
    onApplyCalibration: () => void
    calibrationReady: boolean
    removable: boolean
}) {
    const updatePose = (field: keyof Pose, value: number) => onChange({...entry, signPose: {...entry.signPose, [field]: value}});
    const updateVariation = (variationIndex: number, variation: StopSignVariation) => {
        const variations = [...entry.variations];
        variations[variationIndex] = variation;
        onChange({...entry, variations});
    };
    return (
        <Accordion disableGutters defaultExpanded={index === 0}>
            <AccordionSummary expandIcon={<ExpandMoreRounded />}>
                <Stack direction="row" sx={{alignItems: "center", justifyContent: "space-between", width: "100%", pr: 1, gap: 1}}>
                    <Box>
                        <Typography sx={{fontWeight: 800}}>{entry.id || `sign-${index + 1}`}</Typography>
                        <Typography variant="caption" color="text.secondary">
                            {entry.startDistanceM} m start · {entry.stopDistanceM} m stop · {entry.targetSpeedMps} m/s
                        </Typography>
                    </Box>
                    <IconButton
                        size="small"
                        aria-label={`Remove ${entry.id}`}
                        disabled={!removable}
                        onClick={(event) => { event.stopPropagation(); onRemove(); }}
                    >
                        <DeleteOutlineRounded fontSize="small" />
                    </IconButton>
                </Stack>
            </AccordionSummary>
            <AccordionDetails>
                <Stack direction={{xs: "column", sm: "row"}} sx={{gap: 1, mb: 1.5}}>
                    <TextField label="Sign id" value={entry.id} onChange={(event) => onChange({...entry, id: event.target.value})} fullWidth />
                    <Button variant="outlined" disabled={!calibrationReady} onClick={onApplyCalibration}>Use live pose</Button>
                </Stack>
                <Typography variant="overline" color="text.secondary">Sign pose</Typography>
                <Box sx={fieldGrid(4)}>
                    <NumberField label="X" value={entry.signPose.x} onChange={(value) => updatePose("x", value)} />
                    <NumberField label="Y" value={entry.signPose.y} onChange={(value) => updatePose("y", value)} />
                    <NumberField label="Z" value={entry.signPose.z} onChange={(value) => updatePose("z", value)} />
                    <NumberField label="Heading" value={entry.signPose.heading} onChange={(value) => updatePose("heading", value)} />
                </Box>
                <Typography variant="overline" color="text.secondary" sx={{display: "block", mt: 2}}>Base run</Typography>
                <Box sx={fieldGrid(4)}>
                    <NumberField label="Stop distance (m)" value={entry.stopDistanceM} onChange={(value) => onChange({...entry, stopDistanceM: value})} />
                    <NumberField label="Ego-center offset (m)" value={entry.egoCenterOffsetM} onChange={(value) => onChange({...entry, egoCenterOffsetM: value})} />
                    <NumberField label="Start distance (m)" value={entry.startDistanceM} onChange={(value) => onChange({...entry, startDistanceM: value})} />
                    <NumberField label="Target speed (m/s)" value={entry.targetSpeedMps} onChange={(value) => onChange({...entry, targetSpeedMps: value})} />
                    <NumberField label="Dwell (ms)" value={entry.dwellMs} onChange={(value) => onChange({...entry, dwellMs: value})} />
                    <NumberField label="Attempts" value={entry.attemptCount} onChange={(value) => onChange({...entry, attemptCount: value})} />
                    <TextField label="Weather" value={entry.weather} onChange={(event) => onChange({...entry, weather: event.target.value})} />
                    <NumberField label="Hour" value={entry.time.hour} onChange={(value) => onChange({...entry, time: {...entry.time, hour: value}})} />
                    <NumberField label="Minute" value={entry.time.minute} onChange={(value) => onChange({...entry, time: {...entry.time, minute: value}})} />
                    <TextField label="Vehicle model" value={entry.vehicle.model ?? ""} onChange={(event) => onChange({...entry, vehicle: {...entry.vehicle, model: event.target.value}})} />
                    <ColorFields entry={entry} onChange={onChange} />
                </Box>

                <Divider sx={{my: 2}} />
                <Stack direction="row" sx={{alignItems: "center", justifyContent: "space-between"}}>
                    <Box>
                        <Typography sx={{fontWeight: 800}}>Variants</Typography>
                        <Typography variant="caption" color="text.secondary">Blank fields inherit the base run.</Typography>
                    </Box>
                    <Button
                        size="small"
                        startIcon={<AddRounded />}
                        onClick={() => onChange({...entry, variations: [...entry.variations, {id: `variation-${entry.variations.length + 1}`}]})}
                    >
                        Add variation
                    </Button>
                </Stack>
                <Stack sx={{gap: 1, mt: 1}}>
                    {entry.variations.map((variation, variationIndex) => (
                        <VariantEditor
                            key={`${variation.id}-${variationIndex}`}
                            variant={variation}
                            onChange={(next) => updateVariation(variationIndex, next)}
                            onRemove={() => onChange({...entry, variations: entry.variations.filter((_, candidate) => candidate !== variationIndex)})}
                        />
                    ))}
                </Stack>
            </AccordionDetails>
        </Accordion>
    );
}

function VariantEditor({variant, onChange, onRemove}: {variant: StopSignVariation, onChange: (variant: StopSignVariation) => void, onRemove: () => void}) {
    return (
        <Box sx={{border: "1px solid", borderColor: "divider", borderRadius: 1, p: 1.25}}>
            <Stack direction="row" sx={{gap: 1, alignItems: "center"}}>
                <TextField label="Variant id" value={variant.id} onChange={(event) => onChange({...variant, id: event.target.value})} fullWidth />
                <IconButton aria-label={`Remove ${variant.id}`} onClick={onRemove}><DeleteOutlineRounded /></IconButton>
            </Stack>
            <Box sx={{...fieldGrid(4), mt: 1}}>
                <OptionalNumberField label="Stop distance (m)" value={variant.stopDistanceM} onChange={(value) => onChange({...variant, stopDistanceM: value})} />
                <OptionalNumberField label="Start distance (m)" value={variant.startDistanceM} onChange={(value) => onChange({...variant, startDistanceM: value})} />
                <OptionalNumberField label="Ego-center offset (m)" value={variant.egoCenterOffsetM} onChange={(value) => onChange({...variant, egoCenterOffsetM: value})} />
                <OptionalNumberField label="Target speed (m/s)" value={variant.targetSpeedMps} onChange={(value) => onChange({...variant, targetSpeedMps: value})} />
                <OptionalNumberField label="Dwell (ms)" value={variant.dwellMs} onChange={(value) => onChange({...variant, dwellMs: value})} />
                <OptionalNumberField label="Attempts" value={variant.attemptCount} onChange={(value) => onChange({...variant, attemptCount: value})} />
                <TextField label="Weather override" value={variant.weather ?? ""} onChange={(event) => onChange({...variant, weather: event.target.value || undefined})} />
                <TextField label="Vehicle override" value={variant.vehicle?.model ?? ""} onChange={(event) => onChange({...variant, vehicle: {...variant.vehicle, model: event.target.value || undefined}})} />
                <OptionalNumberField label="Hour" value={variant.time?.hour} onChange={(value) => onChange({...variant, time: value === undefined && variant.time?.minute === undefined ? undefined : {hour: value ?? 12, minute: variant.time?.minute ?? 0}})} />
                <OptionalNumberField label="Minute" value={variant.time?.minute} onChange={(value) => onChange({...variant, time: value === undefined && variant.time?.hour === undefined ? undefined : {hour: variant.time?.hour ?? 12, minute: value ?? 0}})} />
                <OptionalVariantColor label="Vehicle R" channel="r" variant={variant} onChange={onChange} />
                <OptionalVariantColor label="Vehicle G" channel="g" variant={variant} onChange={onChange} />
                <OptionalVariantColor label="Vehicle B" channel="b" variant={variant} onChange={onChange} />
            </Box>
        </Box>
    );
}

function OptionalVariantColor({label, channel, variant, onChange}: {
    label: string
    channel: "r" | "g" | "b"
    variant: StopSignVariation
    onChange: (variant: StopSignVariation) => void
}) {
    const color = variant.vehicle?.color;
    return (
        <OptionalNumberField
            label={label}
            value={color?.[channel]}
            onChange={(value) => {
                if (value === undefined && !color) {
                    return;
                }
                const nextColor = value === undefined
                    ? undefined
                    : {...(color ?? {r: 0, g: 0, b: 0}), [channel]: value};
                onChange({...variant, vehicle: {...variant.vehicle, color: nextColor}});
            }}
        />
    );
}

function ColorFields({entry, onChange}: {entry: StopSignPlanEntry, onChange: (entry: StopSignPlanEntry) => void}) {
    const color = entry.vehicle.color ?? {r: 0, g: 0, b: 0};
    const update = (channel: "r" | "g" | "b", value: number) => onChange({
        ...entry,
        vehicle: {...entry.vehicle, color: {...color, [channel]: value}},
    });
    return (
        <>
            <NumberField label="Vehicle R" value={color.r} onChange={(value) => update("r", value)} />
            <NumberField label="Vehicle G" value={color.g} onChange={(value) => update("g", value)} />
            <NumberField label="Vehicle B" value={color.b} onChange={(value) => update("b", value)} />
        </>
    );
}

function NumberField({label, value, onChange}: {label: string, value: number, onChange: (value: number) => void}) {
    return <TextField label={label} type="number" value={value} onChange={(event) => onChange(Number(event.target.value))} />;
}

function OptionalNumberField({label, value, onChange}: {label: string, value?: number, onChange: (value?: number) => void}) {
    return <TextField label={label} type="number" value={value ?? ""} onChange={(event) => onChange(event.target.value === "" ? undefined : Number(event.target.value))} />;
}

function Counter({value, label}: {value: number, label: string}) {
    return (
        <Box sx={{textAlign: "right"}}>
            <Typography sx={{fontWeight: 850, fontSize: "1.15rem", lineHeight: 1}}>{value}</Typography>
            <Typography variant="caption" color="text.secondary">{label}</Typography>
        </Box>
    );
}

function fieldGrid(columns: number) {
    return {
        display: "grid",
        gridTemplateColumns: {xs: "1fr", sm: "repeat(2, minmax(0, 1fr))", lg: `repeat(${columns}, minmax(0, 1fr))`},
        gap: 1,
    };
}
