#!/usr/bin/env bash
# Install audio2text globally from this checkout (uv tool), so `audio2text` works in any folder.
# Re-run after pulling changes: --reinstall rebuilds the tool from the current sources.
set -euo pipefail

cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
    UV=uv
elif [ -x "$HOME/.local/bin/uv" ]; then
    UV="$HOME/.local/bin/uv"
else
    echo "uv не найден, ставлю..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
fi

"$UV" tool install --force --reinstall .
"$UV" tool update-shell >/dev/null 2>&1 || true

echo "Готово: $("$UV" tool dir --bin)/audio2text"
echo "Если команда audio2text не находится, откройте терминал заново."
