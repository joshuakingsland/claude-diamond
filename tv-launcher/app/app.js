(function () {
  "use strict";

  var APP_MANAGER = "luna://com.webos.applicationManager";
  var KEY = { LEFT: 37, UP: 38, RIGHT: 39, DOWN: 40, ENTER: 13, BACK: 461 };
  var STORE_KEY = "tvlauncher.lastFocus";

  var grid = document.getElementById("grid");
  var statusEl = document.getElementById("status");
  var tiles = [];
  var focusIndex = 0;
  var installed = null; // array of {id,title} once known, null if unknown

  // --- Luna service calls (no external library needed) --------------------
  function luna(uri, params, cb) {
    if (typeof window.PalmServiceBridge === "undefined") {
      cb({ returnValue: false, errorText: "Not running on webOS" });
      return;
    }
    var bridge = new window.PalmServiceBridge();
    var done = false;
    bridge.onservicecallback = function (msg) {
      if (done) return;
      done = true;
      var res;
      try { res = JSON.parse(msg); } catch (e) { res = { returnValue: false, errorText: String(msg) }; }
      cb(res);
    };
    bridge.call(uri, JSON.stringify(params || {}));
  }

  // --- Resolve each tile to an installed app ---------------------------------
  function resolve(app) {
    if (!installed) return null;
    for (var i = 0; i < app.ids.length; i++) {
      for (var j = 0; j < installed.length; j++) {
        if (installed[j].id === app.ids[i]) return installed[j].id;
      }
    }
    for (var k = 0; k < installed.length; k++) {
      if (app.match.test(installed[k].title || "")) return installed[k].id;
    }
    return null;
  }

  function applyInstalled() {
    tiles.forEach(function (t) {
      t.resolved = resolve(t.app);
      t.el.classList.toggle("missing", installed !== null && !t.resolved);
    });
  }

  function loadInstalled() {
    luna(APP_MANAGER + "/listApps", {}, function (res) {
      if (res.returnValue && res.apps) {
        installed = res.apps.map(function (a) { return { id: a.id, title: a.title }; });
        applyInstalled();
      }
      // If listing is not permitted, tiles stay enabled and launch falls back
      // to trying each known ID in turn.
    });
  }

  // --- Launching ------------------------------------------------------------
  function launch(tile) {
    var candidates = tile.resolved ? [tile.resolved] : tile.app.ids.slice();
    tile.el.classList.add("launching");
    setStatus("Opening " + tile.app.name + "…");
    (function next() {
      var id = candidates.shift();
      if (!id) {
        tile.el.classList.remove("launching");
        setStatus(tile.app.name + " isn't installed. Install it from the LG Content Store.");
        return;
      }
      luna(APP_MANAGER + "/launch", { id: id }, function (res) {
        if (res.returnValue) {
          tile.el.classList.remove("launching");
          setStatus("");
        } else {
          next();
        }
      });
    })();
  }

  function setStatus(text) { statusEl.textContent = text; }

  // --- Focus & remote navigation -------------------------------------------
  function setFocus(i) {
    if (!tiles[i]) return;
    tiles[focusIndex].el.classList.remove("focused");
    focusIndex = i;
    tiles[i].el.classList.add("focused");
    try { localStorage.setItem(STORE_KEY, String(i)); } catch (e) { /* ignore */ }
  }

  function center(el) {
    var r = el.getBoundingClientRect();
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
  }

  // Move to the nearest tile in the pressed direction.
  function move(dx, dy) {
    var from = center(tiles[focusIndex].el);
    var best = -1, bestScore = Infinity;
    tiles.forEach(function (t, i) {
      if (i === focusIndex) return;
      var c = center(t.el);
      var vx = c.x - from.x, vy = c.y - from.y;
      var along = vx * dx + vy * dy;
      if (along <= 1) return;
      var across = Math.abs(vx * dy) + Math.abs(vy * dx);
      var score = along + across * 2;
      if (score < bestScore) { bestScore = score; best = i; }
    });
    if (best >= 0) setFocus(best);
  }

  document.addEventListener("keydown", function (e) {
    switch (e.keyCode) {
      case KEY.LEFT: move(-1, 0); break;
      case KEY.RIGHT: move(1, 0); break;
      case KEY.UP: move(0, -1); break;
      case KEY.DOWN: move(0, 1); break;
      case KEY.ENTER: launch(tiles[focusIndex]); break;
      case KEY.BACK: break; // stay on the launcher
      default: return;
    }
    e.preventDefault();
  });

  // Magic Remote pointer support.
  function bindPointer(tile, i) {
    tile.el.addEventListener("mouseenter", function () { setFocus(i); });
    tile.el.addEventListener("click", function () { setFocus(i); launch(tile); });
  }

  // Refresh installed list whenever the launcher comes back to the front.
  document.addEventListener("webOSRelaunch", loadInstalled);
  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) { loadInstalled(); setStatus(""); }
  });

  // --- Clock ----------------------------------------------------------------
  function tick() {
    var d = new Date();
    var h = d.getHours();
    var m = d.getMinutes();
    document.getElementById("clock").textContent =
      ((h % 12) || 12) + ":" + (m < 10 ? "0" : "") + m;
    document.getElementById("greeting").textContent =
      h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
  }

  // --- Build ----------------------------------------------------------------
  window.LAUNCHER_APPS.forEach(function (app, i) {
    var el = document.createElement("div");
    el.className = "tile";
    el.style.setProperty("--tile", app.color);
    el.textContent = app.name;
    grid.appendChild(el);
    var tile = { app: app, el: el, resolved: null };
    tiles.push(tile);
    bindPointer(tile, i);
  });

  var saved = 0;
  try { saved = parseInt(localStorage.getItem(STORE_KEY), 10) || 0; } catch (e) { /* ignore */ }
  tiles[0].el.classList.add("focused");
  setFocus(Math.min(saved, tiles.length - 1));

  tick();
  setInterval(tick, 10000);
  loadInstalled();
})();
