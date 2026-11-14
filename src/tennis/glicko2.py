"""Glicko-2 with an overall rating and one rating per surface, walk-forward, per match.

Each player (and each player-surface) has a rating r, a deviation RD (uncertainty) and a
volatility sigma (how erratic their results are). Internally Glicko-2 works on the scale
mu = (r - 1500) / 173.7178, phi = RD / 173.7178.

Differences from textbook Glicko-2, chosen for match-by-match tennis data:
- Every match is its own rating period. Instead of a fixed period grid, RD grows with the
  time since the player's last rated match:
      phi_pre = min(sqrt(phi^2 + sigma^2 * days / period_days), rd_max / 173.7178)
  so a player returning from injury is uncertain again. Matches in the same week (Sackmann
  dates matches by their event's start) add no inflation.
- The update uses phi_pre directly (no extra sigma'^2 term), since time inflation already
  covers the period's growth:
      g(phi) = 1 / sqrt(1 + 3 phi^2 / pi^2)
      E = 1 / (1 + exp(-g(phi_opp) (mu - mu_opp)))
      v = 1 / (g^2 E (1 - E)),  Delta = v g (s - E)
      sigma' = volatility update (Illinois algorithm, constraint tau); tau = 0 keeps sigma fixed
      phi' = 1 / sqrt(1 / phi_pre^2 + 1 / v)
      mu'  = mu + weight * phi'^2 g (s - E)
  Retirements update with weight = retired_weight (0 skips them); walkovers and
  unfinished matches never update.
- Predictions use both players' uncertainty:
      P(a beats b) = 1 / (1 + exp(-g(sqrt(phi_a^2 + phi_b^2)) (mu_a - mu_b)))
  With surfaces, mu = (1 - blend) mu_overall + blend mu_surface and
  phi^2 = (1 - blend) phi_overall^2 + blend phi_surface^2 (prediction only: blend never
  changes the history, exactly as in elo.py).
- Surface states inflate with time since the player's last match ON THAT SURFACE, so a
  grass rating is uncertain again at the start of each grass season.

Pure logic: no database access. Reuses elo.Matches / elo.load_matches / elo.until.
"""
from __future__ import annotations

import math
from array import array
from dataclasses import asdict, dataclass, replace
from datetime import date

SCALE = 173.7178
_PI2 = math.pi ** 2
NAN = float("nan")


@dataclass(frozen=True)
class Glicko2Params:
    tau: float = 0.5              # volatility constraint; 0 = fixed volatility (faster)
    rd_init: float = 350.0        # RD of a new player (and of a new surface rating)
    vol_init: float = 0.06        # starting volatility
    period_days: float = 30.0     # days per 'rating period' for RD inflation
    rd_max: float = 350.0         # RD never inflates past this
    init: float = 1500.0          # starting rating
    retired_weight: float = 1.0   # update weight for retirements; 0 = ignore them
    surface_init: float = 0.0     # first surface rating: 0 = init, 1 = current overall rating
    blend: float = 0.0            # prediction weight on the surface state (0 = overall only)

    def as_dict(self) -> dict:
        return asdict(self)

    def update_key(self) -> "Glicko2Params":
        return replace(self, blend=0.0)


@dataclass
class Glicko2Result:
    p_winner: array        # pre-match P(actual winner wins) at the run's blend
    r_winner: array        # pre-match overall ratings (r scale)
    r_loser: array
    d_winner: array        # pre-match overall RD, inflated to the match date
    d_loser: array
    s_winner: array        # pre-match surface ratings (NaN when the edition has no surface)
    s_loser: array
    sd_winner: array       # pre-match surface RD
    sd_loser: array
    n_winner: array        # pre-match overall rated-match counts
    n_loser: array
    state: "Glicko2State | None" = None


@dataclass
class Glicko2State:
    """Ratings AFTER the last processed match (r/RD scale), plus what is needed to
    inflate RD to a later date."""
    params: Glicko2Params
    rating: dict           # player_id -> r
    rd: dict               # player_id -> RD right after their last rated match
    vol: dict
    count: dict
    last: dict             # player_id -> date ordinal of their last rated match
    srating: dict          # (player_id, surface) -> r
    srd: dict
    svol: dict
    scount: dict
    slast: dict

    def _inflate(self, rd, vol, last, day):
        p = self.params
        el = max(day - last, 0)
        return min(math.sqrt(rd * rd + (SCALE * vol) ** 2 * el / p.period_days), p.rd_max)

    def rd_at(self, pid: int, on: str) -> float:
        """Overall RD of a player on an ISO date (grown by inactivity since their last match)."""
        return self._inflate(self.rd[pid], self.vol[pid], self.last[pid], date.fromisoformat(on).toordinal())

    def srd_at(self, pid: int, surface: str, on: str) -> float | None:
        key = (pid, surface)
        if key not in self.srd:
            return None
        return self._inflate(self.srd[key], self.svol[key], self.slast[key], date.fromisoformat(on).toordinal())


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / _PI2)


def _prob(mu_a, phi_a, mu_b, phi_b) -> float:
    return 1.0 / (1.0 + math.exp(-_g(math.sqrt(phi_a * phi_a + phi_b * phi_b)) * (mu_a - mu_b)))


def _blend_prob(rw, dw, rl, dl, sw, sdw, sl, sdl, blend) -> float:
    """P(winner wins) from r/RD values, blending in the surface state when present."""
    mw, pw, ml, pl = (rw - 1500.0) / SCALE, dw / SCALE, (rl - 1500.0) / SCALE, dl / SCALE
    if blend and sw == sw:                      # sw == sw is False for NaN (no surface)
        b = blend
        mw = (1 - b) * mw + b * (sw - 1500.0) / SCALE
        ml = (1 - b) * ml + b * (sl - 1500.0) / SCALE
        pw = math.sqrt((1 - b) * pw * pw + b * (sdw / SCALE) ** 2)
        pl = math.sqrt((1 - b) * pl * pl + b * (sdl / SCALE) ** 2)
    return _prob(mw, pw, ml, pl)


def _volatility(phi: float, sigma: float, delta: float, v: float, tau: float) -> float:
    """Glicko-2 step 5 (Illinois algorithm)."""
    if tau <= 0.0:
        return sigma
    a = math.log(sigma * sigma)
    phi2, d2 = phi * phi, delta * delta
    tau2 = tau * tau

    def f(x):
        ex = math.exp(x)
        return ex * (d2 - phi2 - v - ex) / (2.0 * (phi2 + v + ex) ** 2) - (x - a) / tau2

    A = a
    if d2 > phi2 + v:
        B = math.log(d2 - phi2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        B = a - k * tau
    fA, fB = f(A), f(B)
    while abs(B - A) > 1e-6:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA /= 2.0
        B, fB = C, fC
    return math.exp(A / 2.0)


def _update(mu, phi, sigma, mu_o, phi_o, s, tau, weight):
    g = _g(phi_o)
    e = 1.0 / (1.0 + math.exp(-g * (mu - mu_o)))
    v = 1.0 / (g * g * e * (1.0 - e))
    sigma2 = _volatility(phi, sigma, v * g * (s - e), v, tau)
    phi2 = 1.0 / math.sqrt(1.0 / (phi * phi) + 1.0 / v)
    return mu + weight * phi2 * phi2 * g * (s - e), phi2, sigma2


def predict(res: Glicko2Result, blend: float) -> array:
    """P(actual winner wins) for every match at another blend weight, from stored values."""
    out = array("d", bytes(8 * len(res.r_winner)))
    for i in range(len(out)):
        out[i] = _blend_prob(res.r_winner[i], res.d_winner[i], res.r_loser[i], res.d_loser[i],
                             res.s_winner[i], res.sd_winner[i], res.s_loser[i], res.sd_loser[i], blend)
    return out


def run(m, p: Glicko2Params) -> Glicko2Result:
    n = len(m)
    floats = lambda: array("d", bytes(8 * n))                   # noqa: E731
    ints = lambda: array("l", bytes(array("l").itemsize * n))   # noqa: E731
    out = Glicko2Result(*(floats() for _ in range(9)), ints(), ints())
    mu0, phi0, sig0 = (p.init - 1500.0) / SCALE, p.rd_init / SCALE, p.vol_init
    phimax, per, tau, rw, blend = p.rd_max / SCALE, p.period_days, p.tau, p.retired_weight, p.blend
    from_overall = p.surface_init >= 0.5
    mu, phi, sig, last, cnt = {}, {}, {}, {}, {}
    smu, sphi, ssig, slast, scnt = {}, {}, {}, {}, {}
    days: dict[str, int] = {}

    def pre(dmu, dphi, dsig, dlast, key, day, mu_default):
        if key in dmu:
            ph, s = dphi[key], dsig[key]
            el = day - dlast[key]
            if el > 0:
                ph = min(math.sqrt(ph * ph + s * s * el / per), phimax)
            return dmu[key], ph, s
        return mu_default, phi0, sig0

    for i in range(n):
        start = m.start[i]
        day = days.get(start)
        if day is None:
            day = days[start] = date.fromisoformat(start).toordinal()
        w, l, surf = m.winner[i], m.loser[i], m.surface[i]
        mw, pw, sw_ = pre(mu, phi, sig, last, w, day, mu0)
        ml, pl, sl_ = pre(mu, phi, sig, last, l, day, mu0)
        out.r_winner[i], out.r_loser[i] = 1500.0 + SCALE * mw, 1500.0 + SCALE * ml
        out.d_winner[i], out.d_loser[i] = SCALE * pw, SCALE * pl
        out.n_winner[i], out.n_loser[i] = cnt.get(w, 0), cnt.get(l, 0)
        if surf is not None:
            kw, kl = (w, surf), (l, surf)
            smw, spw, ssw = pre(smu, sphi, ssig, slast, kw, day, mw if from_overall else mu0)
            sml, spl, ssl = pre(smu, sphi, ssig, slast, kl, day, ml if from_overall else mu0)
            out.s_winner[i], out.s_loser[i] = 1500.0 + SCALE * smw, 1500.0 + SCALE * sml
            out.sd_winner[i], out.sd_loser[i] = SCALE * spw, SCALE * spl
            out.p_winner[i] = _blend_prob(out.r_winner[i], out.d_winner[i], out.r_loser[i], out.d_loser[i],
                                          out.s_winner[i], out.sd_winner[i], out.s_loser[i], out.sd_loser[i],
                                          blend)
        else:
            out.s_winner[i] = out.s_loser[i] = out.sd_winner[i] = out.sd_loser[i] = NAN
            # same code path as predict(), so stored and re-predicted values match exactly
            out.p_winner[i] = _blend_prob(out.r_winner[i], out.d_winner[i], out.r_loser[i], out.d_loser[i],
                                          NAN, NAN, NAN, NAN, blend)
        status = m.status[i]
        if status == "walkover" or status == "unfinished":
            continue
        weight = rw if status == "retired" else 1.0
        if weight <= 0.0:
            continue
        mu[w], phi[w], sig[w] = _update(mw, pw, sw_, ml, pl, 1.0, tau, weight)
        mu[l], phi[l], sig[l] = _update(ml, pl, sl_, mw, pw, 0.0, tau, weight)
        last[w] = last[l] = day
        cnt[w], cnt[l] = cnt.get(w, 0) + 1, cnt.get(l, 0) + 1
        if surf is not None:
            smu[kw], sphi[kw], ssig[kw] = _update(smw, spw, ssw, sml, spl, 1.0, tau, weight)
            smu[kl], sphi[kl], ssig[kl] = _update(sml, spl, ssl, smw, spw, 0.0, tau, weight)
            slast[kw] = slast[kl] = day
            scnt[kw], scnt[kl] = scnt.get(kw, 0) + 1, scnt.get(kl, 0) + 1

    out.state = Glicko2State(
        p,
        {k: 1500.0 + SCALE * v for k, v in mu.items()}, {k: SCALE * v for k, v in phi.items()},
        dict(sig), cnt, last,
        {k: 1500.0 + SCALE * v for k, v in smu.items()}, {k: SCALE * v for k, v in sphi.items()},
        dict(ssig), scnt, slast)
    return out
