// Tiles shown on the launcher, in order.
// `ids` are known webOS app IDs, tried in order. `match` is a fallback: if the
// TV reports its installed apps, a tile resolves to whichever app's title
// matches, so an app that changed its ID (e.g. after a rebrand) still works.
window.LAUNCHER_APPS = [
  { name: "Netflix",     color: "#e50914", ids: ["netflix"],                                   match: /netflix/i },
  { name: "YouTube",     color: "#ff0033", ids: ["youtube.leanback.v4"],                       match: /^youtube$/i },
  { name: "YouTube TV",  color: "#c4001f", ids: ["youtube.leanback.ytv.v1"],                   match: /youtube\s*tv/i },
  { name: "Prime Video", color: "#1a98ff", ids: ["amazon", "com.amazon.amazonvideo.webos"],    match: /prime\s*video|amazon/i },
  { name: "Disney+",     color: "#113ccf", ids: ["com.disney.disneyplus-prod"],                match: /disney/i },
  { name: "HBO Max",     color: "#5a2be2", ids: ["com.hbo.hbomax", "com.wbd.hbomax", "com.wbd.stream"], match: /hbo|^max$/i },
  { name: "Apple TV",    color: "#3a3a3c", ids: ["com.apple.appletv"],                         match: /apple\s*tv/i },
  { name: "Peacock",     color: "#0a7d55", ids: ["com.peacock.tv", "com.peacocktv.peacocktvapp"], match: /peacock/i },
  { name: "Paramount+",  color: "#0064ff", ids: ["com.cbs-ent.paramountplus", "com.cbsinteractive.paramountplus", "com.cbs.cbsallaccess"], match: /paramount/i }
];
