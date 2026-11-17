"""Alerts when the decision layer fires (track B4, docs/PLAN.md), pushed with ntfy.

    detector = ChangeDetector(AlertConfig())
    alerter = Alerter(detector, NtfyNotifier(topic))     # notifier None: detect + print only
    alerts = alerter.process(as_of, rows, decisions)     # once per watch poll

Three parts, each testable alone:

ChangeDetector  per match, compares this poll's B3a decision with what it last saw.
                A decision is *actionable* when it is a bet with stake >= min_stake and
                expected profit >= min_ev. Changes:
                  new   not actionable -> actionable
                  side  actionable on the other player
                  grew  stake >= grow_ratio x the stake at the last alert
                  gone  was alerted as a bet, no longer actionable (with the reason). Not
                        a sell signal: the message tells holders to hold, and to sell only
                        on a 'side' alert (the model flags the other player)
                Unthrottled: the watch prints every change.
Alerter         turns changes into messages and applies the limits: a per-match cooldown
                after a sent alert ('gone' bypasses it, so an edge that vanishes right after
                its alert is still reported), and a global cap per rolling hour. Every
                change comes back as an Alert with status sent / suppressed / failed /
                printed (no notifier), so live.db can record what was pushed and why not.
NtfyNotifier    POST https://ntfy.sh/ with a JSON body (UTF-8 safe for accented names);
                optional access token. The topic name is the only secret on ntfy.sh:
                anyone who knows it can read the stream, so it lives in .env (NTFY_TOPIC).

Messages carry the tags that say how far to trust a bet (stale days, loose name, surface
from history), the same ones B3b splits by. Detector state is in memory: a restarted
watch re-alerts current bets (within the hourly cap). Read-only project: an alert means
"the policy would bet this", never an order.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Protocol

from .decide import Decision
from .live import MarketSnapshot

ROOT = Path(__file__).resolve().parents[3]
NTFY_SERVER = "https://ntfy.sh"


@dataclass(frozen=True)
class AlertConfig:
    min_stake: float = 5.0            # dollars: smaller bets are not worth a push
    min_ev: float = 0.5               # dollars, under the sizing probability
    grow_ratio: float = 1.5           # re-alert when the stake grows this much since the last alert
    cooldown_s: float = 900.0         # per match, after a sent alert
    max_per_hour: int = 12            # all matches, rolling hour
    alert_gone: bool = True           # push when an alerted bet disappears
    priority_ev: float = 10.0         # EV at or above this -> ntfy priority 4 (high)


@dataclass
class Change:
    event: str
    kind: str                         # new / side / grew / gone
    snapshot: MarketSnapshot
    decision: Decision
    prev_cost: float | None = None    # stake at the last alert (grew/gone/side)


@dataclass
class Alert:
    change: Change
    title: str
    message: str
    priority: int = 3
    tags: list[str] = field(default_factory=list)
    status: str = "printed"           # sent / suppressed / failed / printed
    detail: str = ""                  # why suppressed or failed

    @property
    def event(self) -> str:
        return self.change.event

    @property
    def kind(self) -> str:
        return self.change.kind


def actionable(d: Decision, cfg: AlertConfig) -> bool:
    return d.bet and d.sizing.cost >= cfg.min_stake and d.sizing.ev >= cfg.min_ev


class ChangeDetector:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self.alerted: dict[str, tuple[str, float]] = {}   # event -> (side, stake) at the last alerted bet

    def update(self, rows: list[MarketSnapshot], decisions: list[Decision]) -> list[Change]:
        snap = {s.event.event: s for s in rows}
        out = []
        for d in decisions:
            s = snap.get(d.event)
            if s is None:
                continue
            last = self.alerted.get(d.event)
            if actionable(d, self.cfg):
                if last is None:
                    out.append(Change(d.event, "new", s, d))
                elif last[0] != d.side:
                    out.append(Change(d.event, "side", s, d, last[1]))
                elif d.sizing.cost >= self.cfg.grow_ratio * last[1]:
                    out.append(Change(d.event, "grew", s, d, last[1]))
            elif last is not None:
                out.append(Change(d.event, "gone", s, d, last[1]))
        # matches that left the open list: their alerted bets are gone too
        for ev in set(self.alerted) - set(snap):
            self.alerted.pop(ev)
        return out

    def mark(self, ch: Change) -> None:
        """Called for every change, sent or not: the detector tracks what it reported, so a
        suppressed 'new' is not re-reported every poll."""
        if ch.kind == "gone":
            self.alerted.pop(ch.event, None)
        else:
            self.alerted[ch.event] = (ch.decision.side, ch.decision.sizing.cost)


def _tags(s: MarketSnapshot) -> list[str]:
    """Plain-text trust tags for the message."""
    out = []
    det = s.prediction.detail if s.prediction else {}
    for side, key in (("A", "last_a"), ("B", "last_b")):
        last = det.get(key)
        if last and s.day:
            days = (datetime.fromisoformat(s.day) - datetime.fromisoformat(last)).days
            if days > 60:
                out.append(f"{side} last rated {days}d ago")
    for side, p in (("A", s.event.a), ("B", s.event.b)):
        if p.resolved and p.resolved != "exact":
            out.append(f"{side} name {p.resolved}")
    if s.surface_from.startswith("history"):
        out.append("surface from history")
    elif s.surface_from in ("fallback", "unknown"):
        out.append(f"surface {s.surface_from}")
    if s.prediction and s.prediction.stale:
        out.append("STALE")
    if s.status == "live":
        out.append("IN PLAY")
    return out


def format_alert(ch: Change, cfg: AlertConfig) -> Alert:
    s, d = ch.snapshot, ch.decision
    e = s.event
    when = "LIVE " + (s.live.raw if s.live else "") if s.status == "live" else \
        (e.start or "")[:16].replace("T", " ") + " UTC"
    head = f"{e.a.name} v {e.b.name} · {s.tournament or e.title} · {when}"
    mkt = "-" if s.p_market is None else f"{s.p_market:.3f}"
    pm = "-" if s.p_model is None else f"{s.p_model:.3f}"
    if ch.kind == "gone":
        # decide() knows nothing of positions: 'gone' means no NEW buy at today's ask. Selling
        # (= buying the other side) pays spread + fee again; it is worth it only when the
        # model flags the other side, which comes as a 'side' alert, not this one.
        title = f"GONE {e.title}: {d.reason or 'no bet'} (holders: hold)"
        body = [head, f"was ${ch.prev_cost:.0f}; model {pm} vs market {mkt} (P {e.a.name})",
                "Not a sell signal: no new buy at this price. If you hold it, hold; "
                "sell only on a 'side flipped' alert."]
        prio, tags = 3, ["x"]
    else:
        sz = d.sizing
        verb = {"new": "BET", "side": "BET (side flipped)", "grew": f"BET up from ${ch.prev_cost or 0:.0f}"}[ch.kind]
        title = f"{verb} {d.name} {sz.contracts} @{sz.avg_price:.2f} (${sz.cost:.0f})"
        routes = sorted({f.route.split()[0] + " " + f.route.split()[1].rsplit('-', 1)[-1] for f in sz.fills})
        body = [head,
                f"model {pm} vs market {mkt} (P {e.a.name}); p used {d.p_side:.3f} on {d.name}",
                f"EV ${sz.ev:+.2f}, ask {d.ask:.2f}, fills via {', '.join(routes)}"]
        prio = 4 if sz.ev >= cfg.priority_ev else 3
        tags = ["tennis"]
    t = _tags(s)
    if t:
        body.append("check: " + "; ".join(t))
    return Alert(ch, title, "\n".join(body), prio, tags)


class Notifier(Protocol):
    def send(self, alert: Alert) -> None: ...


class NtfyNotifier:
    """ntfy publish (JSON). `post` is injectable for tests: post(url, body_bytes, headers)."""

    def __init__(self, topic: str, server: str = NTFY_SERVER, token: str | None = None,
                 post: Callable[[str, bytes, dict], None] | None = None, timeout: float = 10.0):
        if not topic:
            raise ValueError("ntfy topic required")
        self.topic, self.server, self.token, self.timeout = topic, server.rstrip("/"), token, timeout
        self.post = post or self._post

    def _post(self, url: str, body: bytes, headers: dict) -> None:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            r.read()

    def send(self, alert: Alert) -> None:
        body = {"topic": self.topic, "title": alert.title, "message": alert.message,
                "priority": alert.priority, "tags": alert.tags}
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self.post(self.server + "/", json.dumps(body, ensure_ascii=False).encode("utf-8"), headers)


class Alerter:
    def __init__(self, detector: ChangeDetector, notifier: Notifier | None = None):
        self.detector, self.notifier, self.cfg = detector, notifier, detector.cfg
        self.sent_at: list[datetime] = []                 # rolling hour of sent pushes
        self.last_sent: dict[str, tuple[datetime, str]] = {}   # event -> (time, kind) of its last push

    def process(self, as_of: str, rows: list[MarketSnapshot], decisions: list[Decision]) -> list[Alert]:
        now = datetime.fromisoformat(as_of)
        self.sent_at = [t for t in self.sent_at if t > now - timedelta(hours=1)]
        out = []
        changes = self.detector.update(rows, decisions)
        order = {"gone": 0, "side": 1, "new": 2, "grew": 3}      # most urgent first under the cap
        for ch in sorted(changes, key=lambda c: (order[c.kind], -c.decision.sizing.ev)):
            a = format_alert(ch, self.cfg)
            self.detector.mark(ch)
            out.append(a)
            if self.notifier is None:
                continue
            why = self._limit(ch, now)
            if why:
                a.status, a.detail = "suppressed", why
                continue
            try:
                self.notifier.send(a)
            except Exception as e:                          # a push failing must not stop the watch
                a.status, a.detail = "failed", f"{type(e).__name__}: {e}"[:500]
                continue
            a.status = "sent"
            self.sent_at.append(now)
            self.last_sent[ch.event] = (now, ch.kind)
        return out

    def _limit(self, ch: Change, now: datetime) -> str:
        last = self.last_sent.get(ch.event)
        if ch.kind == "gone":
            if not self.cfg.alert_gone:
                return "gone alerts off"
            if last is None or last[1] == "gone":
                return "its bet was never pushed"
        elif last is not None and (now - last[0]).total_seconds() < self.cfg.cooldown_s:
            return f"cooldown ({self.cfg.cooldown_s:g}s per match)"
        if len(self.sent_at) >= self.cfg.max_per_hour:
            return f"hourly cap ({self.cfg.max_per_hour})"
        return ""


def env_value(name: str, path: Path = ROOT / ".env") -> str | None:
    """From the environment, else a KEY=value line in .env (gitignored)."""
    if os.environ.get(name):
        return os.environ[name]
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip().removeprefix("export ").strip() == name:
            v = v.strip()
            return v[1:-1] if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"" else v
    return None
