#!/data/data/com.termux/files/usr/bin/bash
# ARIZE phone edition installer for Termux (Android).
#
# Run it in Termux with:
#   curl -fsSL https://raw.githubusercontent.com/SnyperWolfe/ARIZE/main/mobile/install-termux.sh | bash
#
# Safe to run again: it updates ARIZE and keeps your settings and memories.
set -e

MODEL="${ARIZE_MODEL:-qwen3:1.7b}"
REPO="https://github.com/SnyperWolfe/ARIZE.git"
DIR="$HOME/ARIZE"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

say "Installing the programs ARIZE needs (this takes a few minutes)"
yes | pkg update -y
pkg install -y python python-pillow git ollama termux-api

say "Asking for access to your phone's files (tap Allow if Android asks)"
if [ ! -d "$HOME/storage/shared" ]; then
  termux-setup-storage || true
  sleep 3
fi

say "Downloading ARIZE"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone --depth 1 "$REPO" "$DIR"
fi

say "Installing ARIZE"
pip install --upgrade "$DIR"

say "Creating the home-screen shortcuts"
mkdir -p "$HOME/.shortcuts"
cat > "$HOME/.shortcuts/ARIZE" <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
arize mobile
EOF
cat > "$HOME/.shortcuts/Stop ARIZE" <<'EOF'
#!/data/data/com.termux/files/usr/bin/bash
pkill -f "arize mobile"
pkill -f "ollama serve"
termux-wake-unlock
echo "ARIZE stopped."
sleep 1
EOF
chmod +x "$HOME/.shortcuts/ARIZE" "$HOME/.shortcuts/Stop ARIZE"
mkdir -p "$HOME/.shortcuts/icons"
cp "$DIR/src/arize_assistant/web/icon-192.png" "$HOME/.shortcuts/icons/ARIZE.png" 2>/dev/null || true

say "Downloading the AI model $MODEL (about 1.4 GB: use Wi-Fi)"
if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  ollama serve >"$HOME/ollama-install.log" 2>&1 &
  sleep 5
fi
ollama pull "$MODEL"

say "All done!"
cat <<EOF

  Add ARIZE to your home screen:
    1. Long-press an empty spot on your home screen and tap "Widgets".
    2. Find "Termux:Widget", drag it onto your home screen.
    3. Tap "ARIZE" in the widget to start it.

  Or start it now by typing:  arize mobile

EOF
