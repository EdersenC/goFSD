import BookmarkAddedRounded from "@mui/icons-material/BookmarkAddedRounded";
import {
    Alert,
    Box,
    Button,
    Chip,
    List,
    ListItemButton,
    Stack,
    Typography,
} from "@mui/material";
import {isStopSignSceneCalibrated, poseSummary} from "../stop-sign-plan";
import type {Pose, StopSignPlanEntry} from "../types";

type Props = {
    scenes: readonly StopSignPlanEntry[]
    activeCatalogId?: string
    busyCatalogId?: string
    onOpen: (catalogId: string) => void
};

export function SavedStopSignScenes({scenes, activeCatalogId, busyCatalogId, onOpen}: Props) {
    const calibrated = scenes.filter(isStopSignSceneCalibrated);
    const draftCount = scenes.length - calibrated.length;

    return (
        <Box component="section" aria-labelledby="saved-stop-signs-title" sx={{mt: 1.5}}>
            <Stack direction="row" sx={{alignItems: "center", justifyContent: "space-between", gap: 1}}>
                <Box>
                    <Typography id="saved-stop-signs-title" sx={{fontWeight: 850}}>Saved signs</Typography>
                    <Typography variant="caption" color="text.secondary">
                        Calibrated positions are linked to the GTA sign ID and saved automatically.
                    </Typography>
                </Box>
                <Chip size="small" color={calibrated.length > 0 ? "success" : "default"} label={`${calibrated.length} ready`} />
            </Stack>

            {calibrated.length === 0 ? (
                <Alert severity="info" sx={{mt: 1}}>Capture Start, Stop, and End once to save a reusable sign setup.</Alert>
            ) : (
                <List dense disablePadding sx={{mt: 1, maxHeight: 250, overflow: "auto", border: "1px solid", borderColor: "divider", borderRadius: 1}}>
                    {calibrated.map((scene) => {
                        const catalogId = scene.catalogId;
                        return (
                            <ListItemButton
                                key={catalogId}
                                selected={catalogId === activeCatalogId}
                                data-saved-catalog-id={catalogId}
                                onClick={() => onOpen(catalogId)}
                                disabled={Boolean(busyCatalogId)}
                                sx={{display: "grid", gridTemplateColumns: "minmax(0, 1fr) auto", gap: 1.5, borderBottom: "1px solid", borderColor: "divider"}}
                            >
                                <Box sx={{minWidth: 0}}>
                                    <Stack direction="row" sx={{alignItems: "center", gap: .75}}>
                                        <BookmarkAddedRounded color="success" fontSize="small" />
                                        <Typography sx={{fontWeight: 850}}>{catalogId}</Typography>
                                    </Stack>
                                    <Box sx={{display: "grid", gridTemplateColumns: {xs: "1fr", sm: "repeat(3, minmax(0, 1fr))"}, gap: .75, mt: .5}}>
                                        <SavedPose label="Start" pose={scene.startPose} />
                                        <SavedPose label="Stop" pose={scene.egoStopPose} />
                                        <SavedPose label="End" pose={scene.exitPose} />
                                    </Box>
                                </Box>
                                <Button size="small" variant="outlined" disabled={Boolean(busyCatalogId)} onClick={(event) => { event.stopPropagation(); onOpen(catalogId); }}>
                                    {busyCatalogId === catalogId ? "Going…" : "Open + go"}
                                </Button>
                            </ListItemButton>
                        );
                    })}
                </List>
            )}
            {draftCount > 0 && (
                <Typography variant="caption" color="text.secondary" sx={{display: "block", mt: .75}}>
                    {draftCount} unfinished {draftCount === 1 ? "setup remains" : "setups remain"} in the Scene builder.
                </Typography>
            )}
        </Box>
    );
}

function SavedPose({label, pose}: {label: string, pose: Pose}) {
    return (
        <Typography variant="caption" color="text.secondary" sx={{fontFamily: "ui-monospace, monospace", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap"}}>
            {label} {poseSummary(pose)}
        </Typography>
    );
}
