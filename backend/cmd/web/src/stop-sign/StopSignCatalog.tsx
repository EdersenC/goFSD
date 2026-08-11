import MapRounded from "@mui/icons-material/MapRounded";
import {
    Alert,
    Box,
    Button,
    Card,
    CardContent,
    Chip,
    List,
    ListItemButton,
    ListItemText,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import {useEffect, useMemo, useState} from "react";
import {fetchStopSignCatalog} from "../api";
import type {StopSignCatalogLocation} from "./catalog";

const resultLimit = 12;

export function StopSignCatalog({connected, busy, onSetWaypoint}: {
    connected: boolean
    busy: boolean
    onSetWaypoint: (location: StopSignCatalogLocation) => void
}) {
    const [locations, setLocations] = useState<StopSignCatalogLocation[]>([]);
    const [error, setError] = useState("");
    const [query, setQuery] = useState("");
    const [selectedId, setSelectedId] = useState("");

    useEffect(() => {
        const controller = new AbortController();
        void fetchStopSignCatalog(controller.signal).then((next) => {
            setLocations(next);
            setSelectedId((current) => current || next[0]?.id || "");
        }).catch((reason: unknown) => {
            if (!controller.signal.aborted) {
                setError(reason instanceof Error ? reason.message : "Failed to load stop-sign catalog");
            }
        });
        return () => controller.abort();
    }, []);

    const matches = useMemo(() => {
        const normalized = query.trim().toLowerCase();
        if (!normalized) {
            return locations.slice(0, resultLimit);
        }
        return locations.filter((location) => (
            location.id.toLowerCase().includes(normalized)
            || location.kind.toLowerCase().includes(normalized)
            || location.model.toLowerCase().includes(normalized)
            || location.sourceYmap.toLowerCase().includes(normalized)
        )).slice(0, resultLimit);
    }, [locations, query]);
    const selected = locations.find((location) => location.id === selectedId);

    return (
        <Card component="section" aria-labelledby="catalog-title">
            <CardContent>
                <Stack direction="row" sx={{justifyContent: "space-between", alignItems: "flex-start", gap: 1}}>
                    <Box>
                        <Typography variant="overline" color="secondary.main">Map registry</Typography>
                        <Typography id="catalog-title" variant="h2">Stop-sign catalog</Typography>
                    </Box>
                    <Chip size="small" label={locations.length > 0 ? `${locations.length} props` : "loading"} />
                </Stack>
                {error && <Alert severity="error" sx={{mt: 1.5}}>{error}</Alert>}
                <TextField
                    label="Search id, type, or map"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    size="small"
                    fullWidth
                    sx={{mt: 1.5}}
                />
                <List dense disablePadding sx={{maxHeight: 220, overflow: "auto", my: 1, border: "1px solid", borderColor: "divider", borderRadius: 1}}>
                    {matches.map((location) => (
                        <ListItemButton key={location.id} selected={location.id === selectedId} onClick={() => setSelectedId(location.id)}>
                            <ListItemText
                                primary={location.id}
                                secondary={`${location.kind.split("_").join(" ")} · ${location.x.toFixed(1)}, ${location.y.toFixed(1)}, ${location.z.toFixed(1)}${location.tilted ? " · tilted" : ""}`}
                            />
                        </ListItemButton>
                    ))}
                    {locations.length > 0 && matches.length === 0 && (
                        <Typography variant="body2" color="text.secondary" sx={{p: 1.5}}>No matching signs.</Typography>
                    )}
                </List>
                <Stack direction={{xs: "column", sm: "row"}} sx={{gap: 1, alignItems: {sm: "center"}}}>
                    <Button
                        variant="outlined"
                        startIcon={<MapRounded />}
                        disabled={!connected || !selected || busy}
                        onClick={() => selected && onSetWaypoint(selected)}
                    >
                        Set GTA waypoint
                    </Button>
                    <Typography variant="caption" color="text.secondary">
                        Physical prop only. Use <code>/tpwaypoint</code>, align the setup car in-lane, then calibrate the live pose.
                    </Typography>
                </Stack>
            </CardContent>
        </Card>
    );
}
