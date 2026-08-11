#!/usr/bin/env bash
set -uo pipefail

parking_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parking_backend_url="${PARKING_BACKEND_URL:-http://127.0.0.1:8080}"
parking_model_url="${PARKING_MODEL_URL:-http://127.0.0.1:8090}"

printf 'Stop Sign Lab resume\n\n'
parking_doctor_status=0
"${parking_project_root}/scripts/doctor.sh" || parking_doctor_status=$?

parking_repo_dirty_count=0
printf '\nRepository\n'
if command -v git >/dev/null 2>&1 \
    && git -C "${parking_project_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    parking_branch="$(git -C "${parking_project_root}" branch --show-current 2>/dev/null)"
    parking_commit="$(git -C "${parking_project_root}" log -1 --format='%h %s' 2>/dev/null)"
    while IFS= read -r -d '' parking_change; do
        parking_repo_dirty_count=$((parking_repo_dirty_count + 1))
    done < <(git -C "${parking_project_root}" status --porcelain=v1 -z --untracked-files=normal 2>/dev/null)
    printf 'branch=%s changes=%d\n' "${parking_branch:-detached}" "${parking_repo_dirty_count}"
    printf 'head=%s\n' "${parking_commit:-unknown}"
else
    printf 'skip  repository status is unavailable\n'
fi

parking_backend_up=false
parking_model_up=false
printf '\nServices\n'
if command -v curl >/dev/null 2>&1; then
    if curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
        "${parking_backend_url}/healthz" >/dev/null 2>&1; then
        parking_backend_up=true
        printf 'ok    backend: %s\n' "${parking_backend_url}"
    else
        printf 'down  backend: %s\n' "${parking_backend_url}"
    fi
    if curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
        "${parking_model_url}/healthz" >/dev/null 2>&1; then
        parking_model_up=true
        printf 'ok    model server: %s\n' "${parking_model_url}"
    else
        printf 'down  model server: %s\n' "${parking_model_url}"
    fi
else
    printf 'skip  curl is unavailable; service probes were not run\n'
fi

resolve_linux_data_root() {
    local raw_root="${FSD_DATA_ROOT:-}"
    local candidate=""
    raw_root="${raw_root#\"}"
    raw_root="${raw_root%\"}"
    raw_root="${raw_root#\'}"
    raw_root="${raw_root%\'}"

    if [[ -n "${raw_root}" && "${raw_root}" == /* ]]; then
        candidate="${raw_root}"
    elif [[ "${raw_root}" =~ ^([[:alpha:]]):[\\/](.*)$ ]]; then
        local drive
        local rest
        drive="$(printf '%s' "${BASH_REMATCH[1]}" | tr '[:upper:]' '[:lower:]')"
        rest="$(printf '%s' "${BASH_REMATCH[2]}" | tr '\\' '/')"
        while [[ "${rest}" == /* ]]; do
            rest="${rest#/}"
        done
        candidate="/mnt/${drive}/${rest}"
    elif [[ -z "${raw_root}" && -d /mnt/s/fsd_fivem_data ]]; then
        candidate="/mnt/s/fsd_fivem_data"
    fi

    if [[ -n "${candidate}" && -d "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi
    return 1
}

parking_data_root=""
parking_run_count=0
parking_trip_count=0
parking_dataset_count=0
parking_goal_trip_count=0
parking_goal_dataset_count=0
parking_goal_ready_count=0
parking_goal_missing_dataset_count=0
parking_goal_unverified_dataset_count=0
parking_legacy_trip_count=0
parking_training_trip_count=0
parking_training_sample_count=0
parking_training_run_count=0
parking_training_eligibility_error_count=0
parking_training_ready=0
parking_training_ready_label="no"
parking_current_fingerprint=""
parking_readiness_resolved=false
if parking_data_root="$(resolve_linux_data_root)"; then
    parking_runs_root="${parking_data_root}/runs"
    parking_readiness_json=""
    if [[ "${parking_backend_up}" == true ]]; then
        parking_readiness_json="$(curl --fail --silent --show-error --connect-timeout 1 --max-time 5 \
            "${parking_backend_url}/processing/readiness" 2>/dev/null || true)"
    fi
    if [[ -z "${parking_readiness_json}" ]]; then
        parking_readiness_json="$("${parking_project_root}/scripts/dev-backend.sh" \
            processing-status -stop-sign-only 2>/dev/null || true)"
    fi
    if [[ -n "${parking_readiness_json}" ]] && command -v node >/dev/null 2>&1; then
        parking_readiness_row="$(printf '%s' "${parking_readiness_json}" | node -e '
let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
    try {
        const value = JSON.parse(raw);
        const countKeys = [
            "runCount",
            "totalTripCount",
            "totalDatasetCount",
            "selectedTripCount",
            "selectedDatasetCount",
            "currentTripCount",
            "missingDatasetCount",
            "unreadyDatasetCount",
            "excludedTripCount",
            "trainingEligibleTripCount",
            "trainingSampleCount",
            "trainingEligibilityErrorCount",
        ];
        if (value.scope !== "stop-sign-temporal-v1") return;
        if (!/^sha256:[0-9a-f]{64}$/.test(value.configFingerprint)) return;
        if (!countKeys.every((key) => Number.isInteger(value[key]) && value[key] >= 0)) return;
        if (!Array.isArray(value.trainingEligibleRunIds) || !value.trainingEligibleRunIds.every((id) => typeof id === "string" && id.trim())) return;
        if (typeof value.trainingReady !== "boolean") return;
        process.stdout.write([
            ...countKeys.map((key) => String(value[key])),
            String(value.trainingEligibleRunIds.length),
            value.trainingReady ? "1" : "0",
            value.configFingerprint,
        ].join("\t"));
    } catch {}
});
' 2>/dev/null || true)"
        if [[ -n "${parking_readiness_row}" ]]; then
            IFS=$'\t' read -r \
                parking_run_count \
                parking_trip_count \
                parking_dataset_count \
                parking_goal_trip_count \
                parking_goal_dataset_count \
                parking_goal_ready_count \
                parking_goal_missing_dataset_count \
                parking_goal_unverified_dataset_count \
                parking_legacy_trip_count \
                parking_training_trip_count \
                parking_training_sample_count \
                parking_training_eligibility_error_count \
                parking_training_run_count \
                parking_training_ready \
                parking_current_fingerprint <<< "${parking_readiness_row}"
            parking_readiness_resolved=true
            if [[ "${parking_training_ready}" -eq 1 ]]; then
                parking_training_ready_label="yes"
            fi
        fi
    fi
    printf '\nData\n'
    if [[ "${parking_readiness_resolved}" == true ]]; then
        printf 'root=%s runs=%d trips=%d datasets=%d stop_sign_trips=%d stop_sign_current=%d legacy_or_other=%d contract=resolved\n' \
            "${parking_data_root}" \
            "${parking_run_count}" \
            "${parking_trip_count}" \
            "${parking_dataset_count}" \
            "${parking_goal_trip_count}" \
            "${parking_goal_ready_count}" \
            "${parking_legacy_trip_count}"
        printf 'training_eligible_trips=%d training_samples=%d independent_runs=%d training_ready=%s eligibility_errors=%d\n' \
            "${parking_training_trip_count}" \
            "${parking_training_sample_count}" \
            "${parking_training_run_count}" \
            "${parking_training_ready_label}" \
            "${parking_training_eligibility_error_count}"
    else
        printf 'root=%s\n' "${parking_data_root}"
        printf 'skip  authoritative processing readiness is unavailable\n'
    fi
else
    printf '\nData\n'
    printf 'skip  no Linux-readable FSD_DATA_ROOT was resolved\n'
fi

parking_action_index=1
print_next_action() {
    printf '  %d. %s\n' "${parking_action_index}" "$1"
    parking_action_index=$((parking_action_index + 1))
}

printf '\nNext actions\n'
if ((parking_doctor_status != 0)); then
    print_next_action "Fix the failed doctor checks above, then run: npm run resume"
fi
if ((parking_repo_dirty_count > 0)); then
    print_next_action "Review ${parking_repo_dirty_count} worktree change(s) before switching context: git status --short"
fi
if [[ "${parking_backend_up}" != true ]]; then
    print_next_action "Start Stop Sign Lab (services plus browser): npm start"
fi
if [[ "${parking_readiness_resolved}" == true && parking_goal_trip_count -eq 0 ]]; then
    print_next_action "Open ${parking_backend_url} and collect the first fresh stop-sign attempts"
elif [[ "${parking_readiness_resolved}" == true && parking_goal_missing_dataset_count -gt 0 ]]; then
    print_next_action "Backfill ${parking_goal_missing_dataset_count} stop-sign trip(s) without datasets: npm run data:backfill"
elif [[ "${parking_readiness_resolved}" == true && parking_goal_unverified_dataset_count -gt 0 ]]; then
    print_next_action "Bring ${parking_goal_unverified_dataset_count} stale stop-sign dataset(s) onto the current contract: npm run data:backfill"
elif [[ "${parking_readiness_resolved}" == true && parking_training_eligibility_error_count -gt 0 ]]; then
    print_next_action "Repair ${parking_training_eligibility_error_count} trip metadata file(s) before training"
elif [[ "${parking_readiness_resolved}" == true && parking_training_ready -ne 1 ]]; then
    print_next_action "Collect a successful stop-sign attempt in another run; training currently has ${parking_training_sample_count} examples from ${parking_training_run_count} independent run(s)"
elif [[ -n "${parking_data_root}" && "${parking_readiness_resolved}" != true ]]; then
    print_next_action "Rerun npm run resume after the backend readiness check is available"
fi
if [[ "${parking_model_up}" != true && "${parking_backend_up}" == true ]]; then
    print_next_action "When training or evaluating, start missing services with: npm start"
fi
if [[ "${parking_backend_up}" == true ]]; then
    print_next_action "Open the Stop Sign workspace: ${parking_backend_url}"
fi
printf 'Architecture: %s/architecture\n' "${parking_backend_url}"

exit "${parking_doctor_status}"
