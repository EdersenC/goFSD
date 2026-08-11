import NearMeRounded from "@mui/icons-material/NearMeRounded";
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

export function StopSignCatalog({connected, busyId, activeCatalogId, onTeleport}: {
    connected: boolean
    busyId?: string
    activeCatalogId?: string
    onTeleport: (location: StopSignCatalogLocation) => void
}) {
    const [locations, setLocations] = useState<StopSignCatalogLocation[]>([]);
    const [error, setError] = useState("");
    const [query, setQuery] = useState("");

    useEffect(() => {
        const controller = new AbortController();
        void fetchStopSignCatalog(controller.signal).then(setLocations).catch((reason: unknown) => {
            if (!controller.signal.aborted) {
                setError(reason instanceof Error ? reason.message : "Failed to load stop-sign catalog");
            }
        });
        return () => controller.abort();
    }, []);

    const matches = useMemo(() => {
        const normalized = query.trim().toLowerCase();
        if (!normalized) return locations;
        return locations.filter((location) => (
            location.id.toLowerCase().includes(normalized)
            || location.kind.toLowerCase().includes(normalized)
            || location.model.toLowerCase().includes(normalized)
            || location.sourceYmap.toLowerCase().includes(normalized)
            || `${location.x.toFixed(0)},${location.y.toFixed(0)}`.includes(normalized)
        ));
    }, [locations, query]);

    return (
        <Card component="section" aria-labelledby="catalog-title">
            <CardContent>
                <Stack direction="row" sx={{justifyContent: "space-between", alignItems: "flex-start", gap: 1}}>
                    <Box>
                        <Typography variant="overline" color="secondary.main">Step 1</Typography>
                        <Typography id="catalog-title" variant="h2">Choose a stop sign</Typography>
                        <Typography variant="body2" color="text.secondary" sx={{mt: .5}}>
                            Every verified main-map stop sign is listed. Teleport places the managed setup car on its nearest drivable lane.
                        </Typography>
                    </Box>
                    <Chip size="small" label={locations.length > 0 ? `${locations.length} total` : "loading"} />
                </Stack>
                {error && <Alert severity="error" sx={{mt: 1.5}}>{error}</Alert>}
                <TextField
                    label="Search all signs"
                    placeholder="ID, coordinates, type, or map"
                    value={query}
                    onChange={(event) => setQuery(event.target.value)}
                    fullWidth
                    sx={{mt: 1.5}}
                />
                <Typography variant="caption" color="text.secondary" sx={{display: "block", mt: 1}}>
                    Showing {matches.length.toLocaleString()} of {locations.length.toLocaleString()}
                </Typography>
                <List
                    dense
                    disablePadding
                    sx={{maxHeight: 430, overflow: "auto", mt: .5, border: "1px solid", borderColor: "divider", borderRadius: 1}}
                >
                    {matches.map((location) => (
                        <ListItemButton
                            key={location.id}
                            selected={location.id === activeCatalogId}
                            data-catalog-id={location.id}
                            sx={{display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: 1, borderBottom: "1px solid", borderColor: "divider"}}
                            onClick={() => onTeleport(location)}
                        >
                            <ListItemText
                                primary={location.id}
                                secondary={`${location.x.toFixed(1)}, ${location.y.toFixed(1)}, ${location.z.toFixed(1)} · ${location.kind.replace(/_/g, " ")}${location.tilted ? " · tilted" : ""}`}
                            />
                            <Button
                                size="small"
                                variant={location.id === activeCatalogId ? "contained" : "outlined"}
                                startIcon={<NearMeRounded />}
                                disabled={!connected || Boolean(busyId)}
                                onClick={(event) => { event.stopPropagation(); onTeleport(location); }}
                            >
                                {busyId === location.id ? "Teleporting…" : "Teleport"}
                            </Button>
                        </ListItemButton>
                    ))}
                    {locations.length > 0 && matches.length === 0 && (
                        <Typography variant="body2" color="text.secondary" sx={{p: 1.5}}>No matching signs.</Typography>
                    )}
                </List>
            </CardContent>
        </Card>
    );
}
