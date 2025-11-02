#!/bin/bash

# SSH / SCP configuration
export REMOTE_HOST=${REMOTE_HOST:-192.168.68.113}
export USERNAME=${USERNAME:-ningchenspark}
export REMOTE_BASE_DIR=${REMOTE_BASE_DIR:-workspace/Stormwater-Management-Model}
# Optional: password for SSH/SCP (used via sshpass if provided)
# Set REMOTE_PASSWORD in your environment to enable non-interactive password authentication
REMOTE_PASSWORD="${REMOTE_PASSWORD:-}"

# Compose ssh/scp commands, using sshpass if REMOTE_PASSWORD is set and sshpass is available
if [[ -n "$REMOTE_PASSWORD" ]] && command -v sshpass >/dev/null 2>&1; then
    SCP_BIN=(sshpass -p "$REMOTE_PASSWORD" scp)
    SSH_BIN=(sshpass -p "$REMOTE_PASSWORD" ssh)
else
    if [[ -n "$REMOTE_PASSWORD" ]] && ! command -v sshpass >/dev/null 2>&1; then
        echo "Warning: REMOTE_PASSWORD is set but sshpass is not installed. Falling back to interactive scp/ssh." >&2
    fi
    SCP_BIN=(scp)
    SSH_BIN=(ssh)
fi

# Common SSH options to prefer password auth and avoid key/pkcs11 prompts
SSH_OPTS=(-o PreferredAuthentications=password -o PubkeyAuthentication=no -o PKCS11Provider=none -o StrictHostKeyChecking=accept-new)

# Get list of added or modified files from git
CHANGED_FILES=$(git status --porcelain | awk '{if ($1 == "A" || $1 == "M" || $1 == "AM" || $1 == "MM") print $2}')

# Check if there are any changed files
if [ -z "$CHANGED_FILES" ]; then
    echo "No added or modified files to transfer."
    exit 0
fi

echo "Transferring the following files:"
echo "$CHANGED_FILES"
echo "Remote base directory: ~/$REMOTE_BASE_DIR"

# SCP each file to the remote host
for file in $CHANGED_FILES; do
    if [ -f "$file" ]; then
        echo "Transferring $file..."
        # Determine remote directory path under REMOTE_BASE_DIR
        rel_dir="$(dirname "$file")"
        if [ "$rel_dir" = "." ]; then
            remote_dir="~/$REMOTE_BASE_DIR"
        else
            remote_dir="~/$REMOTE_BASE_DIR/$rel_dir"
        fi
        # Ensure remote directory exists on remote host
        "${SSH_BIN[@]}" "${SSH_OPTS[@]}" "$USERNAME@$REMOTE_HOST" "mkdir -p \"$remote_dir\"" || { echo "Failed to create remote directory $remote_dir"; continue; }
        # Transfer file to the target remote directory
        "${SCP_BIN[@]}" "${SSH_OPTS[@]}" "$file" "$USERNAME@$REMOTE_HOST:$remote_dir/" || echo "Failed to transfer $file"
    else
        echo "File $file does not exist locally, skipping."
    fi
done

echo "Transfer complete."
