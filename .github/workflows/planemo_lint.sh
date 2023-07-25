#!/usr/bin/env bash
set -exo pipefail

echo -n "$REPOSITORIES" > repository_list.txt
mapfile -t REPO_ARRAY < repository_list.txt
for DIR in "${REPO_ARRAY[@]}"; do
    planemo shed_lint --tools --ensure_metadata --report_level "$REPORT_LEVEL" --fail_level "$FAIL_LEVEL" --recursive "$DIR" "${ADDITIONAL_PLANEMO_OPTIONS[@]}" | tee -a lint_report.txt
done

# Check if each changed tool is in the list of changed repositories
echo -n "$TOOLS" > tool_list.txt
mapfile -t TOOL_ARRAY < tool_list.txt
for TOOL in "${TOOL_ARRAY[@]}"; do
# Check if any changed repo dir is a substring of $TOOL
if ! echo "$TOOL" | grep -qf repository_list.txt; then
    echo "Tool $TOOL not in changed repositories list: .shed.yml file missing" >&2
    exit 1
fi
done