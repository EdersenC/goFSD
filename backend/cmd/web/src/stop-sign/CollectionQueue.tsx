import ArrowDownwardRounded from "@mui/icons-material/ArrowDownwardRounded";
import ArrowUpwardRounded from "@mui/icons-material/ArrowUpwardRounded";
import {
    Box,
    Button,
    Checkbox,
    Chip,
    IconButton,
    Stack,
    Typography,
} from "@mui/material";
import {isStopSignSceneCalibrated} from "../stop-sign-plan";
import type {StopSignPlanEntry} from "../types";

type Props = {
    scenes: readonly StopSignPlanEntry[]
    selectedEntryIds: readonly string[]
    activeEntryId?: string
    disabled: boolean
    onToggle: (entryId: string) => void
    onSelectAll: () => void
    onSelectCurrent: () => void
    onClear: () => void
    onMove: (entryId: string, direction: -1 | 1) => void
};

export function CollectionQueue({
    scenes,
    selectedEntryIds,
    activeEntryId,
    disabled,
    onToggle,
    onSelectAll,
    onSelectCurrent,
    onClear,
    onMove,
}: Props) {
    const readyScenes = scenes.filter(isStopSignSceneCalibrated);
    const scenesById = new Map(readyScenes.map((scene) => [scene.id, scene]));
    const selectedScenes = selectedEntryIds.flatMap((entryId) => {
        const scene = scenesById.get(entryId);
        return scene ? [scene] : [];
    });
    const selectedIds = new Set(selectedEntryIds);
    const unselectedScenes = readyScenes.filter((scene) => !selectedIds.has(scene.id));

    return (
        <Box component="section" aria-labelledby="collection-queue-title" sx={{border: "1px solid", borderColor: "divider", borderRadius: 1.5, p: 1.25}}>
            <Stack direction="row" sx={{justifyContent: "space-between", alignItems: "center", gap: 1}}>
                <Box>
                    <Typography id="collection-queue-title" sx={{fontWeight: 850}}>Collection queue</Typography>
                    <Typography variant="caption" color="text.secondary">
                        Runs each selected sign in this order, then advances automatically.
                    </Typography>
                </Box>
                <Chip size="small" color={selectedScenes.length > 0 ? "secondary" : "default"} label={`${selectedScenes.length}/${readyScenes.length} signs`} />
            </Stack>

            <Stack direction="row" sx={{gap: .75, flexWrap: "wrap", mt: 1}}>
                <Button size="small" variant="outlined" disabled={disabled || readyScenes.length === 0} onClick={onSelectAll}>All ready</Button>
                <Button size="small" variant="outlined" disabled={disabled || !activeEntryId} onClick={onSelectCurrent}>Current only</Button>
                <Button size="small" color="inherit" disabled={disabled || selectedScenes.length === 0} onClick={onClear}>Clear</Button>
            </Stack>

            {readyScenes.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{mt: 1}}>Save Start, Stop, and End for at least one sign.</Typography>
            ) : (
                <Stack sx={{gap: .5, mt: 1}}>
                    {[...selectedScenes, ...unselectedScenes].map((scene) => {
                        const selectedIndex = selectedEntryIds.indexOf(scene.id);
                        const selected = selectedIndex >= 0;
                        return (
                            <Box
                                key={scene.id}
                                data-collection-entry-id={scene.id}
                                data-collection-order={selected ? selectedIndex + 1 : undefined}
                                sx={{
                                    display: "grid",
                                    gridTemplateColumns: "auto auto minmax(0, 1fr) auto",
                                    gap: .5,
                                    alignItems: "center",
                                    borderRadius: 1,
                                    bgcolor: selected ? "rgba(91,196,255,.065)" : "transparent",
                                    pr: .5,
                                }}
                            >
                                <Checkbox
                                    size="small"
                                    checked={selected}
                                    disabled={disabled}
                                    slotProps={{input: {"aria-label": `${selected ? "Remove" : "Add"} ${scene.catalogId ?? scene.id} ${selected ? "from" : "to"} collection queue`}}}
                                    onChange={() => onToggle(scene.id)}
                                />
                                <Chip size="small" variant={selected ? "filled" : "outlined"} label={selected ? `#${selectedIndex + 1}` : "—"} />
                                <Box sx={{minWidth: 0}}>
                                    <Typography noWrap sx={{fontWeight: 750, fontSize: ".86rem"}}>{scene.catalogId ?? scene.id}</Typography>
                                    <Typography variant="caption" color="text.secondary">
                                        {scene.autoVariations?.count ?? Math.max(1, scene.variations.length)} seeded runs
                                    </Typography>
                                </Box>
                                <Stack direction="row">
                                    <IconButton
                                        size="small"
                                        aria-label={`Move ${scene.catalogId ?? scene.id} earlier`}
                                        disabled={disabled || !selected || selectedIndex === 0}
                                        onClick={() => onMove(scene.id, -1)}
                                    >
                                        <ArrowUpwardRounded fontSize="small" />
                                    </IconButton>
                                    <IconButton
                                        size="small"
                                        aria-label={`Move ${scene.catalogId ?? scene.id} later`}
                                        disabled={disabled || !selected || selectedIndex === selectedScenes.length - 1}
                                        onClick={() => onMove(scene.id, 1)}
                                    >
                                        <ArrowDownwardRounded fontSize="small" />
                                    </IconButton>
                                </Stack>
                            </Box>
                        );
                    })}
                </Stack>
            )}
        </Box>
    );
}
