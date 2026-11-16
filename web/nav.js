// Shared top bar for the Advantage pages (served at /nav.js). Each page defines the colour
// tokens (--panel, --line, --text, --muted, --accent, --accent-soft) and includes this script
// right after <body>. The bar marks the current page and keeps each page's own URL hash.
(function () {
  "use strict";
  const PAGES = [
    { href: "/live", match: ["/", "/live", "/live.html", "/index.html"], label: "Live markets" },
    { href: "/ratings", match: ["/ratings", "/ratings.html"], label: "Ratings" },
  ];
  const here = location.pathname;
  const css = `
  .topnav { position: sticky; top: 0; z-index: 20; background: var(--panel); border-bottom: 1px solid var(--line); }
  .topnav .in { max-width: 1400px; margin: 0 auto; padding: 0 16px; display: flex; align-items: center; gap: 4px; height: 44px; }
  .topnav .brand { font-weight: 700; letter-spacing: .01em; margin-right: 14px; color: var(--text); text-decoration: none; }
  .topnav .brand span { color: var(--accent); }
  .topnav a.tab { color: var(--muted); text-decoration: none; padding: 6px 10px; border-radius: 6px; font-size: 14px; }
  .topnav a.tab:hover { color: var(--text); background: var(--hover, rgba(127,127,127,.08)); }
  .topnav a.tab[aria-current="page"] { color: var(--accent); background: var(--accent-soft); font-weight: 600; }
  .topnav .note { margin-left: auto; color: var(--muted); font-size: 12px; white-space: nowrap; }
  .topnav a.tab { white-space: nowrap; }
  @media (max-width: 560px) { .topnav .note { display: none; } .topnav .brand { margin-right: 6px; } }`;
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
  for (const p of PAGES) {
    const a = document.createElement("a");
    a.className = "tab";
    a.href = p.href;
    a.textContent = p.label;
    if (p.match.includes(here)) a.setAttribute("aria-current", "page");
    inner.appendChild(a);
  }
  const note = document.createElement("span");
  note.className = "note";
  note.textContent = "Read-only · model run 790";
  inner.appendChild(note);
  nav.appendChild(inner);
  document.body.prepend(nav);
})();
