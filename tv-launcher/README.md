# TV Launcher

An ad-free home screen for an LG OLED55C1PUB (webOS 6). It shows big tiles for
Netflix, YouTube, YouTube TV, Prime Video, Disney+, HBO Max, Apple TV, Peacock
and Paramount+, and works with the arrow keys or the Magic Remote pointer.

This folder is separate from the MLB modelling code in the rest of the repo.

## Layout

| Path | What it is |
| --- | --- |
| `app/` | The webOS app itself (HTML/CSS/JS, no dependencies) |
| `app/apps.js` | The tile list: edit it to add, remove or reorder apps |
| `build.sh` | Packages `app/` into `dist/*.ipk` (needs Node; fetches LG's CLI with `npx`) |
| `install.sh` | Copies the package to a rooted TV over SSH, installs it, and sets it to open at boot |
| `root/autostart-launcher` | Homebrew Channel boot script that opens the launcher after the TV starts |

## How app IDs are found

Each tile lists known webOS app IDs and a title pattern. At start-up the
launcher asks the TV which apps are installed and matches each tile by ID, then
by title, so an app that changed its ID after a rebrand still opens. Tiles for
apps that aren't installed are dimmed and marked "Not installed".

## Install (rooted TV)

1. Root the TV and install the Homebrew Channel (see rootmy.tv / webosbrew.org
   for a method that supports your firmware). Enable SSH in the Homebrew Channel
   settings, and change the default root password (`alpine`).
2. From a computer on the same network: `./install.sh <tv-ip>`

## Install (no root, Developer Mode)

Package with `./build.sh`, then install with LG's CLI:
`npx -p @webos-tools/cli ares-install -d <device> dist/*.ipk`. Developer Mode
expires after about 50 hours unless renewed, and there is no autostart.

## Limits

- The Home button still opens LG's own home bar; the launcher is opened at boot
  and from the app list.
- Tested in a desktop browser with a simulated TV service, not yet on the TV.
