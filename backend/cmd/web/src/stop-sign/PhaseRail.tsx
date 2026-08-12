import {Box, Stack, Typography} from "@mui/material";

export const STOP_SIGN_PHASES = [
    {id: "approach", label: "Approach", signal: "flash · launch → cruise"},
    {id: "brake_stop", label: "Brake → Stop", signal: "flash · decelerate → 0 m/s"},
    {id: "release", label: "Release → End", signal: "flash · scripted release"},
] as const;

export function resolvePhaseIndex(phase: string | undefined): number {
    const normalized = phase?.trim().toLowerCase() ?? "";
    if (/go|depart|resume|release|complete|success/.test(normalized)) {
        return 2;
    }
    if (/stop|dwell|hold/.test(normalized)) {
        return 1;
    }
    if (/approach|cruise|decel|brak|slow/.test(normalized)) {
        return /decel|brak|slow/.test(normalized) ? 1 : 0;
    }
    return 0;
}

export function PhaseRail({phase}: {phase?: string}) {
    const active = resolvePhaseIndex(phase);
    return (
        <Box component="section" aria-label="Behavior phases">
            <Stack direction={{xs: "column", md: "row"}} sx={{gap: 1}}>
                {STOP_SIGN_PHASES.map((item, index) => {
                    const isActive = index === active;
                    const isComplete = index < active;
                    return (
                        <Box
                            key={item.id}
                            data-phase={item.id}
                            data-active={isActive || undefined}
                            sx={{
                                flex: 1,
                                minWidth: 0,
                                border: "1px solid",
                                borderColor: isActive ? "primary.main" : "divider",
                                bgcolor: isActive ? "rgba(233,255,91,.08)" : "background.paper",
                                borderRadius: 1.5,
                                px: 1.6,
                                py: 1.3,
                                position: "relative",
                                overflow: "hidden",
                                "&::before": {
                                    content: '""',
                                    position: "absolute",
                                    inset: "0 auto 0 0",
                                    width: 3,
                                    bgcolor: isActive ? "primary.main" : isComplete ? "success.main" : "transparent",
                                },
                            }}
                        >
                            <Typography variant="overline" color={isActive ? "primary.main" : "text.secondary"}>
                                {String(index + 1).padStart(2, "0")}
                            </Typography>
                            <Typography sx={{fontWeight: 800, lineHeight: 1.2}}>{item.label}</Typography>
                            <Typography variant="caption" color="text.secondary">{item.signal}</Typography>
                        </Box>
                    );
                })}
            </Stack>
        </Box>
    );
}
