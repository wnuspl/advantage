// Shared top bar for the Advantage pages (served at /nav.js). Each page defines the colour
// tokens (--panel, --line, --text, --muted, --accent, --accent-soft) and includes this script
// right after <body>. The bar marks the current page and keeps each page's own URL hash.
// Main pages sit on the left; the "Performance" menu on the right holds how the model and
// the flagged bets have done (scorecard, paper P&L).
(function () {
  "use strict";
  const PAGES = [
    { href: "/live", match: ["/", "/live", "/live.html", "/index.html"], label: "Live markets" },
    { href: "/ratings", match: ["/ratings", "/ratings.html"], label: "Ratings" },
  ];
  const PERFORMANCE = [
    { href: "/scorecard", match: ["/scorecard", "/scorecard.html"], label: "Scorecard", sub: "Model vs market log loss" },
    { href: "/pnl", match: ["/pnl", "/pnl.html"], label: "P&L", sub: "Paper bets and sell rules" },
  ];
  const here = location.pathname;
  const css = `
  .topnav { position: sticky; top: 0; z-index: 20; background: var(--panel); border-bottom: 1px solid var(--line); }
  .topnav .in { max-width: 1400px; margin: 0 auto; padding: 0 16px; display: flex; align-items: center; gap: 4px; height: 44px; }
  .topnav .brand { font-weight: 700; letter-spacing: .01em; margin-right: 14px; color: var(--text); text-decoration: none; }
  .topnav .brand span { color: var(--accent); }
  .topnav a.tab, .topnav button.tab { color: var(--muted); text-decoration: none; padding: 6px 10px; border-radius: 6px; font: inherit; font-size: 14px;
    white-space: nowrap; background: none; border: 0; cursor: pointer; }
  .topnav .tab:hover { color: var(--text); background: var(--hover, rgba(127,127,127,.08)); }
  .topnav .tab[aria-current="page"] { color: var(--accent); background: var(--accent-soft); font-weight: 600; }
  .topnav .note { margin-left: auto; margin-right: 6px; color: var(--muted); font-size: 12px; white-space: nowrap; }
  .topnav .menu { position: relative; }
  .topnav .menu button.tab::after { content: ""; display: inline-block; margin-left: 6px; vertical-align: 2px;
    border: 4px solid transparent; border-top-color: currentColor; border-bottom: 0; }
  .topnav .drop { position: absolute; right: 0; top: calc(100% + 6px); min-width: 230px; background: var(--panel);
    border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 6px 20px rgba(0,0,0,.14); padding: 4px; }
  .topnav .drop[hidden] { display: none; }
  .topnav .drop a { display: block; padding: 7px 10px; border-radius: 6px; color: var(--text); text-decoration: none; font-size: 14px; }
  .topnav .drop a small { display: block; color: var(--muted); font-size: 12px; }
  .topnav .drop a:hover, .topnav .drop a:focus-visible { background: var(--hover, rgba(127,127,127,.08)); outline: none; }
  .topnav .drop a[aria-current="page"] { color: var(--accent); background: var(--accent-soft); font-weight: 600; }
  @media (max-width: 560px) { .topnav .note { display: none; } .topnav .brand { margin-right: 6px; }
    .topnav .menu { margin-left: auto; } .topnav .in { gap: 0; }
    .topnav a.tab, .topnav button.tab { padding: 6px 7px; font-size: 13px; } }`;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);
  const nav = document.createElement("nav");
  nav.className = "topnav";
  nav.setAttribute("aria-label", "Pages");
  const inner = document.createElement("div");
  inner.className = "in";
  const brand = document.createElement("a");
  brand.className = "brand";
  brand.href = "/live";
  brand.innerHTML = "Advant<span>age</span>";
  inner.appendChild(brand);
  const link = (p, cls) => {
    const a = document.createElement("a");
    if (cls) a.className = cls;
    a.href = p.href;
    a.textContent = p.label;
    if (p.sub) { const s = document.createElement("small"); s.textContent = p.sub; a.appendChild(s); }
    if (p.match.includes(here)) a.setAttribute("aria-current", "page");
    return a;
  };
  for (const p of PAGES) inner.appendChild(link(p, "tab"));
  const note = document.createElement("span");
  note.className = "note";
  note.textContent = "Read-only · model run 790";
  inner.appendChild(note);

  // Performance menu (right side)
  const menu = document.createElement("div");
  menu.className = "menu";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "tab";
  btn.textContent = "Performance";
  btn.setAttribute("aria-haspopup", "true");
  btn.setAttribute("aria-expanded", "false");
  const drop = document.createElement("div");
  drop.className = "drop";
  drop.id = "perf-menu";
  drop.hidden = true;
  btn.setAttribute("aria-controls", drop.id);
  for (const p of PERFORMANCE) drop.appendChild(link(p));
  if (PERFORMANCE.some((p) => p.match.includes(here))) btn.setAttribute("aria-current", "page");
  const open = (on) => { drop.hidden = !on; btn.setAttribute("aria-expanded", String(on)); };
  btn.addEventListener("click", (e) => { e.stopPropagation(); open(drop.hidden); if (!drop.hidden) drop.querySelector("a").focus(); });
  document.addEventListener("click", (e) => { if (!menu.contains(e.target)) open(false); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !drop.hidden) { open(false); btn.focus(); }
  });
  drop.addEventListener("keydown", (e) => {
    const items = [...drop.querySelectorAll("a")], i = items.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length].focus(); }
    if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
  });
  menu.addEventListener("focusout", (e) => { if (!menu.contains(e.relatedTarget)) open(false); });
  menu.append(btn, drop);
  inner.appendChild(menu);
  nav.appendChild(inner);
  document.body.prepend(nav);
})();
