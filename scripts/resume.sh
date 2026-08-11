#!/usr/bin/env bash
set -uo pipefail

stop_sign_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
stop_sign_backend_url="${STOP_SIGN_BACKEND_URL:-http://127.0.0.1:8080}"
stop_sign_model_url="${STOP_SIGN_MODEL_URL:-http://127.0.0.1:8090}"

printf 'Stop Sign Lab resume\n\n'
stop_sign_doctor_status=0
"${stop_sign_project_root}/scripts/doctor.sh" || stop_sign_doctor_status=$?

stop_sign_repo_dirty_count=0
printf '\nRepository\n'
if command -v git >/dev/null 2>&1 \
    && git -C "${stop_sign_project_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    stop_sign_branch="$(git -C "${stop_sign_project_root}" branch --show-current 2>/dev/null)"
    stop_sign_commit="$(git -C "${stop_sign_project_root}" log -1 --format='%h %s' 2>/dev/null)"
    while IFS= read -r -d '' stop_sign_change; do
        stop_sign_repo_dirty_count=$((stop_sign_repo_dirty_count + 1))
    done < <(git -C "${stop_sign_project_root}" status --porcelain=v1 -z --untracked-files=normal 2>/dev/null)
    printf 'branch=%s changes=%d\n' "${stop_sign_branch:-detached}" "${stop_sign_repo_dirty_count}"
    printf 'head=%s\n' "${stop_sign_commit:-unknown}"
else
    printf 'skip  repository status is unavailable\n'
fi

stop_sign_backend_up=false
stop_sign_model_up=false
printf '\nServices\n'
if command -v curl >/dev/null 2>&1; then
    if curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
        "${stop_sign_backend_url}/healthz" >/dev/null 2>&1; then
        stop_sign_backend_up=true
        printf 'ok    backend: %s\n' "${stop_sign_backend_url}"
    else
        printf 'down  backend: %s\n' "${stop_sign_backend_url}"
    fi
    if curl --fail --silent --show-error --connect-timeout 1 --max-time 2 \
        "${stop_sign_model_url}/healthz" >/dev/null 2>&1; then
        stop_sign_model_up=true
        printf 'ok    model server: %s\n' "${stop_sign_model_url}"
    else
        printf 'down  model server: %s\n' "${stop_sign_model_url}"
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

stop_sign_data_root=""
stop_sign_run_count=0
stop_sign_trip_count=0
stop_sign_dataset_count=0
stop_sign_goal_trip_count=0
stop_sign_goal_dataset_count=0
stop_sign_goal_ready_count=0
stop_sign_goal_missing_dataset_count=0
stop_sign_goal_unverified_dataset_count=0
stop_sign_legacy_trip_count=0
stop_sign_training_trip_count=0
stop_sign_training_sample_count=0
stop_sign_training_run_count=0
stop_sign_training_eligibility_error_count=0
stop_sign_training_ready=0
stop_sign_training_ready_label="no"
stop_sign_current_fingerprint=""
stop_sign_readiness_resolved=false
if stop_sign_data_root="$(resolve_linux_data_root)"; then
    stop_sign_runs_root="${stop_sign_data_root}/runs"
    stop_sign_readiness_json=""
    if [[ "${stop_sign_backend_up}" == true ]]; then
        stop_sign_readiness_json="$(curl --fail --silent --show-error --connect-timeout 1 --max-time 5 \
            "${stop_sign_backend_url}/processing/readiness" 2>/dev/null || true)"
    fi
    if [[ -z "${stop_sign_readiness_json}" ]]; then
        stop_sign_readiness_json="$("${stop_sign_project_root}/scripts/dev-backend.sh" \
            processing-status -stop-sign-only 2>/dev/null || true)"
    fi
    if [[ -n "${stop_sign_readiness_json}" ]] && command -v node >/dev/null 2>&1; then
        stop_sign_readiness_row="$(printf '%s' "${stop_sign_readiness_json}" | node -e '
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
        if (value.scope !== "stop-sign-continuous-v2") return;
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
        if [[ -n "${stop_sign_readiness_row}" ]]; then
            IFS=$'\t' read -r \
                stop_sign_run_count \
                stop_sign_trip_count \
                stop_sign_dataset_count \
                stop_sign_goal_trip_count \
                stop_sign_goal_dataset_count \
                stop_sign_goal_ready_count \
                stop_sign_goal_missing_dataset_count \
                stop_sign_goal_unverified_dataset_count \
                stop_sign_legacy_trip_count \
                stop_sign_training_trip_count \
                stop_sign_training_sample_count \
                stop_sign_training_eligibility_error_count \
                stop_sign_training_run_count \
                stop_sign_training_ready \
                stop_sign_current_fingerprint <<< "${stop_sign_readiness_row}"
            stop_sign_readiness_resolved=true
            if [[ "${stop_sign_training_ready}" -eq 1 ]]; then
                stop_sign_training_ready_label="yes"
            fi
        fi
    fi
    printf '\nData\n'
    if [[ "${stop_sign_readiness_resolved}" == true ]]; then
        printf 'root=%s runs=%d trips=%d datasets=%d stop_sign_trips=%d stop_sign_current=%d legacy_or_other=%d contract=resolved\n' \
            "${stop_sign_data_root}" \
            "${stop_sign_run_count}" \
            "${stop_sign_trip_count}" \
            "${stop_sign_dataset_count}" \
            "${stop_sign_goal_trip_count}" \
            "${stop_sign_goal_ready_count}" \
            "${stop_sign_legacy_trip_count}"
        printf 'training_eligible_trips=%d training_samples=%d independent_runs=%d training_ready=%s eligibility_errors=%d\n' \
            "${stop_sign_training_trip_count}" \
            "${stop_sign_training_sample_count}" \
            "${stop_sign_training_run_count}" \
            "${stop_sign_training_ready_label}" \
            "${stop_sign_training_eligibility_error_count}"
    else
        printf 'root=%s\n' "${stop_sign_data_root}"
        printf 'skip  authoritative processing readiness is unavailable\n'
    fi
else
    printf '\nData\n'
    printf 'skip  no Linux-readable FSD_DATA_ROOT was resolved\n'
fi

stop_sign_action_index=1
print_next_action() {
    printf '  %d. %s\n' "${stop_sign_action_index}" "$1"
    stop_sign_action_index=$((stop_sign_action_index + 1))
}

printf '\nNext actions\n'
if ((stop_sign_doctor_status != 0)); then
    print_next_action "Fix the failed doctor checks above, then run: npm run resume"
fi
if ((stop_sign_repo_dirty_count > 0)); then
    print_next_action "Review ${stop_sign_repo_dirty_count} worktree change(s) before switching context: git status --short"
fi
if [[ "${stop_sign_backend_up}" != true ]]; then
    print_next_action "Start Stop Sign Lab (services plus browser): npm start"
fi
if [[ "${stop_sign_readiness_resolved}" == true && stop_sign_goal_trip_count -eq 0 ]]; then
    print_next_action "Open ${stop_sign_backend_url} and collect the first fresh stop-sign attempts"
elif [[ "${stop_sign_readiness_resolved}" == true && stop_sign_goal_missing_dataset_count -gt 0 ]]; then
    print_next_action "Backfill ${stop_sign_goal_missing_dataset_count} stop-sign trip(s) without datasets: npm run data:backfill"
elif [[ "${stop_sign_readiness_resolved}" == true && stop_sign_goal_unverified_dataset_count -gt 0 ]]; then
    print_next_action "Bring ${stop_sign_goal_unverified_dataset_count} stale stop-sign dataset(s) onto the current contract: npm run data:backfill"
elif [[ "${stop_sign_readiness_resolved}" == true && stop_sign_training_eligibility_error_count -gt 0 ]]; then
    print_next_action "Repair ${stop_sign_training_eligibility_error_count} trip metadata file(s) before training"
elif [[ "${stop_sign_readiness_resolved}" == true && stop_sign_training_ready -ne 1 ]]; then
    print_next_action "Collect a successful stop-sign attempt in another run; training currently has ${stop_sign_training_sample_count} examples from ${stop_sign_training_run_count} independent run(s)"
elif [[ -n "${stop_sign_data_root}" && "${stop_sign_readiness_resolved}" != true ]]; then
    print_next_action "Rerun npm run resume after the backend readiness check is available"
fi
if [[ "${stop_sign_model_up}" != true && "${stop_sign_backend_up}" == true ]]; then
    print_next_action "When training or evaluating, start missing services with: npm start"
fi
if [[ "${stop_sign_backend_up}" == true ]]; then
    print_next_action "Open the Stop Sign workspace: ${stop_sign_backend_url}"
fi
printf 'Architecture: %s/architecture\n' "${stop_sign_backend_url}"

exit "${stop_sign_doctor_status}"
