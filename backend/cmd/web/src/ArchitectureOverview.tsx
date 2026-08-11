import ArrowForwardRounded from "@mui/icons-material/ArrowForwardRounded";
import {Box, Button, Card, CardContent, Chip, Container, Divider, Stack, Typography} from "@mui/material";
import type {ReactNode} from "react";

const phases = [
    ["accelerate", "Launch from the saved start pose"],
    ["cruise_approach", "Hold the planned approach speed"],
    ["decelerate", "Reduce the speed profile before the line"],
    ["stop_hold", "Stay below 0.1 m/s for the full dwell"],
    ["release", "V0 controller releases after the scripted dwell"],
] as const;

export function ArchitectureOverview() {
    return (
        <Box sx={{minHeight: "100vh", bgcolor: "#080b0e", py: 3}}>
            <Container maxWidth="lg">
                <Stack direction={{xs: "column", sm: "row"}} sx={{justifyContent: "space-between", gap: 2, mb: 3}}>
                    <Box>
                        <Typography variant="overline" color="secondary.main">System contract</Typography>
                        <Typography variant="h1">Stop Sign Lab architecture</Typography>
                        <Typography color="text.secondary" sx={{mt: 1, maxWidth: 780}}>
                            The learned model describes desired future motion. A deterministic controller translates that plan into safe game inputs.
                        </Typography>
                    </Box>
                    <Button href="/" variant="outlined" sx={{alignSelf: "flex-start"}}>Open workbench</Button>
                </Stack>

                <Section title="Model → controller → game">
                    <Box sx={{display: "grid", gridTemplateColumns: {xs: "1fr", md: "1fr auto 1fr auto 1fr"}, gap: 1.5, alignItems: "stretch"}}>
                        <FlowCard eyebrow="Input" title="Causal RGB clip" detail="Five RGB frames over one second. Current speed may be added explicitly; sign geometry never enters the model." />
                        <FlowArrow />
                        <FlowCard eyebrow="Learned" title="Temporal motion plan" detail="Future speed at 0.25, 0.5, 1, 2, 3, and 5 seconds plus stop intent. No raw controller voltages." />
                        <FlowArrow />
                        <FlowCard eyebrow="Deterministic" title="Feedback controller" detail="Tracks speed, latches the stop, rate-limits output, and guarantees throttle and brake are mutually exclusive before FiveM." />
                    </Box>
                    <Typography variant="caption" color="warning.main" sx={{display: "block", mt: 2}}>
                        V0 release is scripted after the configured dwell. It is not claimed as learned behavior.
                    </Typography>
                </Section>

                <Box sx={{display: "grid", gridTemplateColumns: {xs: "1fr", md: "1fr 1fr"}, gap: 2, mt: 2}}>
                    <Section title="Scene geometry">
                        <ContractRow name="signPose" detail="Calibrated lane reference and travel heading" />
                        <ContractRow name="stopLinePose" detail="Sign pose minus stopDistanceM" />
                        <ContractRow name="egoStopPose" detail="Stop line minus egoCenterOffsetM" />
                        <ContractRow name="startPose" detail="Ego stop pose minus startDistanceM" />
                        <ContractRow name="exitPose" detail="Sign pose plus exitDistanceM" />
                        <Divider sx={{my: 1.5}} />
                        <Typography variant="body2" color="text.secondary">
                            Signed line distance is positive before the line and negative after it. The scoring oracle uses front-bumper distance; the RGB policy does not receive it.
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{mt: 1}}>
                            Catalog prop coordinates are navigation-only. The origin placeholder is rejected until a live lane pose is applied.
                        </Typography>
                    </Section>

                    <Section title="Temporal labels">
                        <Stack sx={{gap: 1}}>
                            {phases.map(([name, detail]) => (
                                <Stack key={name} direction="row" sx={{gap: 1, alignItems: "center"}}>
                                    <Chip size="small" color="secondary" variant="outlined" label={name} sx={{minWidth: 134, fontFamily: "ui-monospace, monospace"}} />
                                    <Typography variant="body2" color="text.secondary">{detail}</Typography>
                                </Stack>
                            ))}
                        </Stack>
                    </Section>
                </Box>

                <Box sx={{display: "grid", gridTemplateColumns: {xs: "1fr", md: "1fr 1fr"}, gap: 2, mt: 2}}>
                    <Section title="Dataset boundary">
                        <Typography variant="body2" color="text.secondary">
                            Store complete clips with synchronized RGB, speed, acceleration, physical brake pressure, expert throttle/brake diagnostics, phase, stop/go labels, goal revision, variant seed, and outcome. Failed attempts stay inspectable but are excluded from expert training by default.
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{mt: 1.5}}>
                            Build a phase-balanced training index and split by stop location or scenario family—never by individual frame.
                        </Typography>
                    </Section>
                    <Section title="Runtime ownership">
                        <ContractRow name="Catalog" detail="463 physical prop locations for navigation only" />
                        <ContractRow name="FiveM" detail="Scene setup, expert clips, physical telemetry" />
                        <ContractRow name="Go backend" detail="Capture, batch queue, processing, guarded controller" />
                        <ContractRow name="Python" detail="Temporal dataset, training, checkpoint server" />
                        <ContractRow name="Workbench" detail="Collect, Data, Train, Evaluate, global Hold" />
                    </Section>
                </Box>
            </Container>
        </Box>
    );
}

function Section({title, children}: {title: string, children: ReactNode}) {
    return (
        <Card component="section">
            <CardContent>
                <Typography variant="h2" sx={{mb: 2}}>{title}</Typography>
                {children}
            </CardContent>
        </Card>
    );
}

function FlowCard({eyebrow, title, detail}: {eyebrow: string, title: string, detail: string}) {
    return (
        <Box sx={{border: "1px solid", borderColor: "divider", borderRadius: 1.5, p: 2, bgcolor: "rgba(255,255,255,.02)"}}>
            <Typography variant="overline" color="secondary.main">{eyebrow}</Typography>
            <Typography variant="h3">{title}</Typography>
            <Typography variant="body2" color="text.secondary" sx={{mt: 1}}>{detail}</Typography>
        </Box>
    );
}

function FlowArrow() {
    return <ArrowForwardRounded color="secondary" sx={{alignSelf: "center", justifySelf: "center", transform: {xs: "rotate(90deg)", md: "none"}}} />;
}

function ContractRow({name, detail}: {name: string, detail: string}) {
    return (
        <Stack direction="row" sx={{gap: 1.5, justifyContent: "space-between", py: .65}}>
            <Typography variant="body2" sx={{fontFamily: "ui-monospace, monospace", color: "primary.main"}}>{name}</Typography>
            <Typography variant="body2" color="text.secondary" sx={{textAlign: "right"}}>{detail}</Typography>
        </Stack>
    );
}
