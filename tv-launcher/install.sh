#!/bin/sh
# Install the launcher on a rooted LG TV (Homebrew Channel with SSH enabled)
# and make it open automatically at boot.
#   usage: ./install.sh <tv-ip>
# The Homebrew Channel's default SSH login is root / alpine; change it.
set -e
TV="$1"
[ -n "$TV" ] || { echo "usage: $0 <tv-ip>"; exit 1; }
cd "$(dirname "$0")"
IPK=$(ls -1t dist/*.ipk 2>/dev/null | head -n1)
[ -n "$IPK" ] || { ./build.sh; IPK=$(ls -1t dist/*.ipk | head -n1); }

scp "$IPK" "root@$TV:/tmp/tvlauncher.ipk"
scp root/autostart-launcher "root@$TV:/var/lib/webosbrew/init.d/autostart-launcher"
ssh "root@$TV" '
  chmod +x /var/lib/webosbrew/init.d/autostart-launcher
  luna-send -w 30000 -i luna://com.webos.appInstallService/dev/install \
    "{\"id\":\"com.local.tvlauncher\",\"ipkUrl\":\"/tmp/tvlauncher.ipk\",\"subscribe\":true}" \
    | grep -m1 -E "installed|failed|error" || true
  rm -f /tmp/tvlauncher.ipk
  luna-send -n 1 luna://com.webos.applicationManager/launch "{\"id\":\"com.local.tvlauncher\"}"
'
echo "Done. The launcher should now be on screen."
