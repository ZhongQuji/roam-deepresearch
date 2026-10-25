#!/usr/bin/env python3
"""
Design-space simulation: ReAct vs PSV (verify-before-write) vs PSV variants.

This is an *abstract cost/recall model*, not an LLM experiment. Every quantity
(document length, catalog size, per-call overheads, scout yield, planner noise,
latency) is a parameter in `Params`, and the defaults are tuned so that the
baseline ReAct and PSV rows land near the Drafts48 medians reported in the
project log (tokens per opened document, documents opened, gold-read recall,
forced-termination rate, wall time). The purpose is to rank design variants
*before* spending API budget, and to make the structural argument (quadratic
context accumulation vs. bounded per-role contexts) visible. Numbers here must
not be quoted as experimental evidence in the paper.

Designs
-------
react       sequential search/open loop; context accumulates; stops on budget
react2x     same, with twice the token budget (the paper's budget control)
psv         planner -> K parallel scouts -> model-free gate -> writer (+revision)
psv_T       tiered: cheap scout model except for look-alike identity scans
psv_L       lexical pre-screen drops low-overlap named docs before scouting
psv_D       card de-duplication + per-document metadata in the writer input
psv_M       mention-following: one bounded second wave on docs cited by cards
psv_R1      one bounded re-planning round with a compact coverage ledger
psv_S       saturation waves: scout in waves, stop when a wave yields no card
psv_lean    T + L + D + M combined

Usage
-----
python3 sim/psv_cost_sim.py --tasks 600 --seed 7 --out sim/out
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List

import numpy as np


# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
@dataclass
class Params:
    # corpus / task
    N: int = 300                 # catalogue size visible to the planner
    G: int = 8                   # gold documents per task
    H: int = 5                   # planted look-alikes (hard negatives)
    hidden_frac: float = 0.35    # share of gold docs with low catalogue-visible relevance
    mention_p: float = 0.25      # P(a visible gold record cross-references a given hidden gold record)
    doc_len_mu: float = float(np.log(1500.0))
    doc_len_sigma: float = 0.60
    doc_len_min: int = 300
    doc_len_max: int = 8000
    catalog_tok_per_doc: int = 35
    base_ctx: int = 1500         # system prompt + brief

    # ReAct
    react_search_k: int = 10
    react_search_tok: int = 60
    react_step_out: int = 300    # thought + action (reasoning tokens)
    react_final_out: int = 1500
    react_max_steps: int = 40
    react_tok_budget: int = 150_000
    react_ctx_limit: int = 128_000
    react_done_docs: int = 12    # voluntary stop (rarely reached under budget)
    react_submit_p: float = 0.02 # per-step chance of an early voluntary submit once >= 4 docs are open
    react_mention_prob: float = 0.3
    react_score_noise: float = 0.35   # ad-hoc queries: noisy ranking of the catalogue

    # PSV
    K: int = 20
    docs_per_scout: int = 3      # a scout call covers one sub-question and ~3 named documents
    planner_out: int = 600
    planner_noise: float = 0.15
    scout_in_overhead: int = 800
    card_tok: int = 200          # scout output per card (quote + claim + stance)
    cards_gold_lo: int = 2
    cards_gold_hi: int = 3
    p_card_gold: float = 0.92
    p_card_distractor: float = 0.30
    lookalike_extra_out: int = 300   # identity reasoning in the comparator scan
    gate_drop_strong: float = 0.03   # quote-hallucination rate removed by the gate
    gate_drop_cheap: float = 0.08
    p_card_gold_cheap: float = 0.85
    writer_in_overhead: int = 1500
    writer_card_tok: int = 150       # card as seen by the writer (quote + claim + ids)
    writer_doc_meta: int = 40
    writer_out: int = 1800
    revision_out: int = 600
    dup_card_frac: float = 0.18      # share of cards that duplicate another card's quote

    # variants
    prescreen_tau: float = 0.35
    mention_k: int = 5
    replan_k: int = 5
    replan_hidden_p: float = 0.40
    replan_ledger_tok: int = 500
    replan_out: int = 300
    wave_size: int = 8

    # prices, USD per 1M tokens (generic "strong" tier; cheap tier is a factor)
    price_in: float = 1.0
    price_out: float = 4.0
    cheap_factor: float = 0.2

    # latency model (seconds)
    lat_base: float = 2.0
    lat_in_per_tok: float = 1.0 / 20_000
    lat_out_per_tok: float = 0.05
    cheap_speed: float = 0.7
    jitter_sigma: float = 0.30


# ----------------------------------------------------------------------------
# Task generation
# ----------------------------------------------------------------------------
@dataclass
class Task:
    kind: np.ndarray        # 0 distractor, 1 gold visible, 2 gold hidden, 3 look-alike
    length: np.ndarray      # tokens
    surface: np.ndarray     # catalogue-visible relevance
    overlap: np.ndarray     # lexical overlap with the planner's sub-questions
    mentions: Dict[int, List[int]] = field(default_factory=dict)

    @property
    def gold(self) -> np.ndarray:
        return np.flatnonzero((self.kind == 1) | (self.kind == 2))

    @property
    def hidden(self) -> np.ndarray:
        return np.flatnonzero(self.kind == 2)


def make_task(rng: np.random.Generator, P: Params) -> Task:
    n = P.N
    kind = np.zeros(n, dtype=int)
    idx = rng.permutation(n)
    n_hidden = int(round(P.G * P.hidden_frac))
    gold = idx[: P.G]
    hidden, visible = gold[:n_hidden], gold[n_hidden:]
    look = idx[P.G: P.G + P.H]
    kind[visible], kind[hidden], kind[look] = 1, 2, 3

    length = np.clip(rng.lognormal(P.doc_len_mu, P.doc_len_sigma, n),
                     P.doc_len_min, P.doc_len_max).astype(int)

    surface = rng.beta(1, 4, n) * 0.6
    surface[kind == 1] = 0.80
    surface[kind == 2] = 0.25
    surface[kind == 3] = 0.75

    overlap = rng.beta(3, 7, n)
    overlap[kind == 1] = rng.beta(7, 3, (kind == 1).sum())
    overlap[kind == 2] = rng.beta(5, 5, (kind == 2).sum())
    overlap[kind == 3] = rng.beta(6, 4, (kind == 3).sum())

    mentions: Dict[int, List[int]] = {}
    for g in visible:                       # gold records cross-reference hidden gold
        m = [int(h) for h in hidden if rng.random() < P.mention_p]
        if m:
            mentions[int(g)] = m
    for l in look:                          # look-alikes discuss the comparison
        m = [int(g) for g in visible if rng.random() < 0.3]
        if m:
            mentions[int(l)] = m
    return Task(kind, length, surface, overlap, mentions)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def latency(P: Params, tok_in: int, tok_out: int, rng: np.random.Generator,
            speed: float = 1.0) -> float:
    base = P.lat_base + tok_in * P.lat_in_per_tok + tok_out * P.lat_out_per_tok
    return base * speed * float(rng.lognormal(0.0, P.jitter_sigma))


def usd(P: Params, tok_in: int, tok_out: int, cheap: bool = False) -> float:
    f = P.cheap_factor if cheap else 1.0
    return f * (tok_in * P.price_in + tok_out * P.price_out) / 1e6


@dataclass
class Result:
    design: str
    tok_in: int
    tok_out: int
    usd: float
    wall_s: float
    docs_opened: int
    gold_recall: float
    hidden_recall: float
    lookalikes_opened: int
    forced: int
    calls: int
    cards_to_writer: int
    gate_drops: int
    serial_s: float = 0.0       # sum of per-call latencies (what a sequential run would take)

    @property
    def tokens(self) -> int:
        return self.tok_in + self.tok_out


# ----------------------------------------------------------------------------
# ReAct
# ----------------------------------------------------------------------------
def run_react(task: Task, rng: np.random.Generator, P: Params,
              budget_mult: float = 1.0, name: str = "react") -> Result:
    budget = P.react_tok_budget * budget_mult
    tok_in = tok_out = 0
    wall = 0.0
    ctx = P.base_ctx
    opened: List[int] = []
    seen: set = set()
    steps = n_search = 0
    forced = 0

    while True:
        if steps >= P.react_max_steps or len(opened) >= P.react_done_docs:
            break
        if len(opened) >= 4 and rng.random() < P.react_submit_p:
            break                                   # voluntary submit
        step_cost = ctx + P.react_step_out
        if tok_in + tok_out + step_cost + ctx + P.react_final_out > budget or ctx > P.react_ctx_limit:
            forced = 1
            break
        tok_in += ctx
        tok_out += P.react_step_out
        wall += latency(P, ctx, P.react_step_out, rng)
        steps += 1

        # alternate search / open (a real ReAct trace spends roughly half its steps
        # on searching and re-reading results); search again if nothing is left to open
        do_search = (steps % 2 == 1) or not (seen - set(opened))
        if do_search:
            noise = P.react_score_noise / (1.0 + 0.3 * n_search)
            score = task.surface + rng.normal(0.0, noise, P.N)
            score[opened] = -np.inf
            top = np.argsort(-score)[: P.react_search_k]
            seen.update(int(t) for t in top)
            n_search += 1
            ctx += P.react_search_k * P.react_search_tok + P.react_step_out
        else:
            cand = None
            if rng.random() < P.react_mention_prob:
                pool = [m for d in opened for m in task.mentions.get(int(d), []) if m not in opened]
                if pool:
                    cand = int(rng.choice(pool))
            if cand is None:
                avail = [d for d in seen if d not in opened]
                noisy = {d: task.surface[d] + rng.normal(0.0, P.react_score_noise) for d in avail}
                cand = max(noisy, key=noisy.get)
            opened.append(cand)
            ctx += int(task.length[cand]) + P.react_step_out

    # final write always happens (forced or not)
    tok_in += ctx
    tok_out += P.react_final_out
    wall += latency(P, ctx, P.react_final_out, rng)

    gold = set(task.gold.tolist())
    hidden = set(task.hidden.tolist())
    op = set(opened)
    return Result(
        design=name, tok_in=tok_in, tok_out=tok_out, usd=usd(P, tok_in, tok_out),
        wall_s=wall, docs_opened=len(op),
        gold_recall=len(op & gold) / max(1, len(gold)),
        hidden_recall=(len(op & hidden) / len(hidden)) if hidden else float("nan"),
        lookalikes_opened=int(sum(task.kind[d] == 3 for d in op)),
        forced=forced, calls=steps + 1, cards_to_writer=0, gate_drops=0, serial_s=wall,
    )


# ----------------------------------------------------------------------------
# PSV and variants
# ----------------------------------------------------------------------------
def run_psv(task: Task, rng: np.random.Generator, P: Params, name: str = "psv",
            tiered: bool = False, prescreen: bool = False, dedup: bool = False,
            mention: bool = False, replan: bool = False, waves: bool = False) -> Result:
    tok_in = tok_out = 0
    usd_total = 0.0
    wall = 0.0
    serial = 0.0
    calls = 0
    gate_drops = 0
    opened: List[int] = []
    kept_cards: List[int] = []          # doc id per kept card
    cards_by_doc: Dict[int, int] = {}

    # --- planner -----------------------------------------------------------
    cat = P.N * P.catalog_tok_per_doc
    pin, pout = P.base_ctx + cat, P.planner_out
    tok_in += pin; tok_out += pout; usd_total += usd(P, pin, pout); calls += 1
    lp = latency(P, pin, pout, rng); wall += lp; serial += lp
    score = task.surface + rng.normal(0.0, P.planner_noise, P.N)
    named = [int(d) for d in np.argsort(-score)[: P.K]]
    if prescreen:
        named = [d for d in named if task.overlap[d] >= P.prescreen_tau]

    # --- scouting wave ------------------------------------------------------
    def scout_wave(docs: List[int]) -> int:
        """One parallel wave of scout calls; each call covers up to `docs_per_scout`
        documents (one sub-question). Returns the number of cards that passed the gate."""
        nonlocal tok_in, tok_out, usd_total, wall, serial, calls, gate_drops
        todo = [d for d in docs if d not in opened]
        # look-alikes go to the dedicated comparator (identity) scan, which stays on
        # the strong model; ordinary documents are grouped by sub-question
        look = [d for d in todo if task.kind[d] == 3]
        rest = [d for d in todo if task.kind[d] != 3]
        groups = [rest[i: i + P.docs_per_scout] for i in range(0, len(rest), P.docs_per_scout)]
        groups += [look[i: i + P.docs_per_scout] for i in range(0, len(look), P.docs_per_scout)]
        new_cards = 0
        lat_max = 0.0
        for group in groups:
            cheap = tiered and not any(task.kind[d] == 3 for d in group)
            s_in = P.scout_in_overhead
            s_out = 0
            for d in group:
                opened.append(d)
                s_in += int(task.length[d])
                k = task.kind[d]
                if k in (1, 2):
                    p = P.p_card_gold_cheap if cheap else P.p_card_gold
                    n_cards = int(rng.integers(P.cards_gold_lo, P.cards_gold_hi + 1)) if rng.random() < p else 0
                elif k == 3:
                    n_cards = 1
                    s_out += P.lookalike_extra_out
                else:
                    n_cards = 1 if rng.random() < P.p_card_distractor else 0
                s_out += n_cards * P.card_tok
                drop_p = P.gate_drop_cheap if cheap else P.gate_drop_strong
                for _ in range(n_cards):
                    if rng.random() < drop_p:
                        gate_drops += 1
                    else:
                        kept_cards.append(d)
                        cards_by_doc[d] = cards_by_doc.get(d, 0) + 1
                        new_cards += 1
            tok_in += s_in; tok_out += s_out; calls += 1
            usd_total += usd(P, s_in, s_out, cheap=cheap)
            ls = latency(P, s_in, s_out, rng, speed=P.cheap_speed if cheap else 1.0)
            serial += ls
            lat_max = max(lat_max, ls)
        wall += lat_max
        return new_cards

    if waves:
        for i in range(0, len(named), P.wave_size):
            if scout_wave(named[i: i + P.wave_size]) == 0:
                break
    else:
        scout_wave(named)

    # --- bounded second chances --------------------------------------------
    if mention:
        pool: Dict[int, int] = {}
        for d in opened:
            if cards_by_doc.get(d, 0) == 0:
                continue                      # mentions travel on cards
            for m in task.mentions.get(int(d), []):
                if m not in opened:
                    pool[m] = pool.get(m, 0) + 1
        follow = sorted(pool, key=lambda m: -pool[m])[: P.mention_k]
        if follow:
            scout_wave(follow)

    if replan:
        rin, rout = P.base_ctx + cat + P.replan_ledger_tok, P.replan_out
        tok_in += rin; tok_out += rout; usd_total += usd(P, rin, rout); calls += 1
        lr = latency(P, rin, rout, rng); wall += lr; serial += lr
        extra: List[int] = []
        for h in task.hidden:
            if int(h) not in opened and rng.random() < P.replan_hidden_p:
                extra.append(int(h))
        if len(extra) < P.replan_k:
            rest = [int(d) for d in np.argsort(-score) if int(d) not in opened and int(d) not in extra]
            extra += rest[: P.replan_k - len(extra)]
        scout_wave(extra[: P.replan_k])

    # --- writer (+ coverage revision) ----------------------------------------
    n_cards = len(kept_cards)
    if dedup:
        n_cards = int(round(n_cards * (1.0 - P.dup_card_frac)))
        meta = len(cards_by_doc) * P.writer_doc_meta
    else:
        meta = n_cards * P.writer_doc_meta           # metadata repeated per card
    w_in = P.writer_in_overhead + n_cards * P.writer_card_tok + meta
    tok_in += w_in; tok_out += P.writer_out; usd_total += usd(P, w_in, P.writer_out); calls += 1
    lw = latency(P, w_in, P.writer_out, rng); wall += lw; serial += lw
    tok_in += w_in; tok_out += P.revision_out; usd_total += usd(P, w_in, P.revision_out); calls += 1
    lv = latency(P, w_in, P.revision_out, rng); wall += lv; serial += lv

    gold = set(task.gold.tolist())
    hidden = set(task.hidden.tolist())
    op = set(opened)
    return Result(
        design=name, tok_in=tok_in, tok_out=tok_out, usd=usd_total, wall_s=wall,
        docs_opened=len(op), gold_recall=len(op & gold) / max(1, len(gold)),
        hidden_recall=(len(op & hidden) / len(hidden)) if hidden else float("nan"),
        lookalikes_opened=int(sum(task.kind[d] == 3 for d in op)),
        forced=0, calls=calls, cards_to_writer=n_cards, gate_drops=gate_drops, serial_s=serial,
    )


DESIGNS = {
    "react":    lambda t, r, P: run_react(t, r, P, 1.0, "react"),
    "react2x":  lambda t, r, P: run_react(t, r, P, 2.0, "react2x"),
    "psv":      lambda t, r, P: run_psv(t, r, P, "psv"),
    "psv_T":    lambda t, r, P: run_psv(t, r, P, "psv_T", tiered=True),
    "psv_L":    lambda t, r, P: run_psv(t, r, P, "psv_L", prescreen=True),
    "psv_D":    lambda t, r, P: run_psv(t, r, P, "psv_D", dedup=True),
    "psv_M":    lambda t, r, P: run_psv(t, r, P, "psv_M", mention=True),
    "psv_R1":   lambda t, r, P: run_psv(t, r, P, "psv_R1", replan=True),
    "psv_S":    lambda t, r, P: run_psv(t, r, P, "psv_S", waves=True),
    "psv_lean": lambda t, r, P: run_psv(t, r, P, "psv_lean", tiered=True, prescreen=True,
                                        dedup=True, mention=True),
}

LABELS = {
    "react": "ReAct", "react2x": "ReAct 2× budget", "psv": "PSV",
    "psv_T": "PSV-T tiered scouts", "psv_L": "PSV-L lexical pre-screen",
    "psv_D": "PSV-D card dedup", "psv_M": "PSV-M mention-following",
    "psv_R1": "PSV-R1 one re-plan", "psv_S": "PSV-S saturation waves",
    "psv_lean": "PSV-lean (T+L+D+M)",
}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def summarise(rows: List[Result]) -> Dict[str, float]:
    def med(x): return float(np.nanmedian(x))
    def iqr(x):
        q1, q3 = np.nanpercentile(x, [25, 75]); return float(q1), float(q3)
    tokens = np.array([r.tokens for r in rows], dtype=float)
    out = {
        "n": len(rows),
        "tokens_med": med(tokens), "tokens_q1": iqr(tokens)[0], "tokens_q3": iqr(tokens)[1],
        "tok_in_med": med([r.tok_in for r in rows]),
        "tok_out_med": med([r.tok_out for r in rows]),
        "usd_med": med([r.usd for r in rows]),
        "wall_med": med([r.wall_s for r in rows]),
        "wall_q1": iqr([r.wall_s for r in rows])[0], "wall_q3": iqr([r.wall_s for r in rows])[1],
        "docs_med": med([r.docs_opened for r in rows]),
        "tok_per_doc_med": med([r.tokens / max(1, r.docs_opened) for r in rows]),
        "gold_recall_mean": float(np.mean([r.gold_recall for r in rows])),
        "gold_recall_sd": float(np.std([r.gold_recall for r in rows])),
        "hidden_recall_mean": float(np.nanmean([r.hidden_recall for r in rows])),
        "lookalikes_med": med([r.lookalikes_opened for r in rows]),
        "forced_rate": float(np.mean([r.forced for r in rows])),
        "calls_med": med([r.calls for r in rows]),
        "cards_med": med([r.cards_to_writer for r in rows]),
        "gate_drops_mean": float(np.mean([r.gate_drops for r in rows])),
        "serial_med": med([r.serial_s for r in rows]),
        "overlap_med": med([r.serial_s / max(1e-9, r.wall_s) for r in rows]),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="sim/out")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    P = Params()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tasks = [make_task(rng, P) for _ in range(args.tasks)]

    per_design: Dict[str, List[Result]] = {}
    for i, (name, fn) in enumerate(DESIGNS.items()):
        # deterministic per-design stream (Python's str hash is salted per process)
        drng = np.random.default_rng(args.seed + 1000 + i)
        per_design[name] = [fn(t, drng, P) for t in tasks]

    summary = {name: summarise(rows) for name, rows in per_design.items()}

    with open(os.path.join(args.out, "params.json"), "w") as f:
        json.dump(asdict(P), f, indent=2)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "per_task.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "task", "tok_in", "tok_out", "tokens", "usd", "wall_s", "docs_opened",
                    "gold_recall", "hidden_recall", "lookalikes_opened", "forced", "calls",
                    "cards_to_writer", "gate_drops", "serial_s"])
        for name, rows in per_design.items():
            for i, r in enumerate(rows):
                w.writerow([name, i, r.tok_in, r.tok_out, r.tokens, f"{r.usd:.5f}", f"{r.wall_s:.1f}",
                            r.docs_opened, f"{r.gold_recall:.3f}", f"{r.hidden_recall:.3f}",
                            r.lookalikes_opened, r.forced, r.calls, r.cards_to_writer, r.gate_drops, f"{r.serial_s:.1f}"])

    # markdown table
    base = summary["psv"]
    lines = [
        "| design | tokens (med, IQR) | vs PSV | $/task | wall s (med) | serial/wall | docs | tok/doc | gold recall | hidden-gold recall | look-alikes read | forced | calls | gate drops |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in DESIGNS:
        s = summary[name]
        lines.append(
            f"| {LABELS[name]} | {s['tokens_med']/1000:.0f}k ({s['tokens_q1']/1000:.0f}–{s['tokens_q3']/1000:.0f}k) "
            f"| {100*s['tokens_med']/base['tokens_med']:.0f}% | {s['usd_med']:.3f} | {s['wall_med']:.0f} | {s['overlap_med']:.2f} "
            f"| {s['docs_med']:.0f} | {s['tok_per_doc_med']/1000:.1f}k | {s['gold_recall_mean']:.2f}±{s['gold_recall_sd']:.2f} "
            f"| {s['hidden_recall_mean']:.2f} | {s['lookalikes_med']:.0f}/{P.H} | {100*s['forced_rate']:.0f}% "
            f"| {s['calls_med']:.0f} | {s['gate_drops_mean']:.1f} |"
        )
    table = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(table + "\n")
    print(table)

    sweeps = run_sweeps(tasks, P, args.seed)
    with open(os.path.join(args.out, "sweeps.json"), "w") as f:
        json.dump(sweeps, f, indent=2)

    if not args.no_plots:
        make_plots(per_design, summary, P, args.out)
        make_sweep_plots(sweeps, P, args.out)


# ----------------------------------------------------------------------------
# Sensitivity sweeps: the three parameters the conclusions hinge on
# ----------------------------------------------------------------------------
def run_sweeps(tasks: List[Task], P: Params, seed: int) -> Dict[str, list]:
    import copy
    out: Dict[str, list] = {"react_budget": [], "mention_p": [], "prescreen_tau": []}

    for mult in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        rng = np.random.default_rng(seed + 11)
        rows = [run_react(t, rng, P, mult, "react") for t in tasks]
        s = summarise(rows)
        out["react_budget"].append({"mult": mult, **s})

    for mp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        Pm = copy.deepcopy(P); Pm.mention_p = mp
        rng_t = np.random.default_rng(seed)
        tasks_m = [make_task(rng_t, Pm) for _ in range(len(tasks))]
        rng = np.random.default_rng(seed + 12)
        rows = [run_psv(t, rng, Pm, "psv_M", mention=True) for t in tasks_m]
        base = [run_psv(t, np.random.default_rng(seed + 13), Pm, "psv") for t in tasks_m]
        s, b = summarise(rows), summarise(base)
        out["mention_p"].append({"mention_p": mp, **s, "base_gold_recall_mean": b["gold_recall_mean"],
                                 "base_tokens_med": b["tokens_med"]})

    for tau in (0.0, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6):
        Pt = copy.deepcopy(P); Pt.prescreen_tau = tau
        rng = np.random.default_rng(seed + 14)
        rows = [run_psv(t, rng, Pt, "psv_L", prescreen=True) for t in tasks]
        s = summarise(rows)
        out["prescreen_tau"].append({"tau": tau, **s})
    return out


def make_sweep_plots(sweeps: Dict[str, list], P: Params, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_react, c_psv = "#b2182b", "#2166ac"
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))

    ax = axes[0]
    rb = sweeps["react_budget"]
    x = [r["tokens_med"] / 1000 for r in rb]
    ax.plot(x, [r["gold_recall_mean"] for r in rb], "o-", color=c_react, label="gold recall")
    ax.set_xlabel("ReAct tokens per task (k) as the budget grows 0.5×→4×")
    ax.set_ylabel("gold documents read (recall)", color=c_react)
    ax2 = ax.twinx()
    ax2.plot(x, [r["docs_med"] for r in rb], "s--", color="#ef8a62", label="documents opened")
    ax2.set_ylabel("documents opened (median)", color="#ef8a62")
    for r in rb:
        ax.annotate(f"{r['mult']:g}×", (r["tokens_med"] / 1000, r["gold_recall_mean"]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.set_title("(a) ReAct: diminishing returns to budget")
    ax.set_ylim(0, 1.0)

    ax = axes[1]
    mp = sweeps["mention_p"]
    xs = [r["mention_p"] for r in mp]
    ax.plot(xs, [r["gold_recall_mean"] for r in mp], "o-", color=c_psv, label="PSV-M gold recall")
    ax.plot(xs, [r["base_gold_recall_mean"] for r in mp], "s--", color="grey", label="PSV gold recall")
    ax.set_xlabel("P(a read gold record cross-references a hidden gold record)")
    ax.set_ylabel("gold documents read (recall)")
    ax2 = ax.twinx()
    ax2.plot(xs, [100 * r["tokens_med"] / r["base_tokens_med"] for r in mp], "^:", color="#4393c3")
    ax2.set_ylabel("PSV-M tokens as % of PSV", color="#4393c3")
    ax2.set_ylim(100, 130)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.set_title("(b) Mention-following pays only if records cross-reference")

    ax = axes[2]
    pt = sweeps["prescreen_tau"]
    xs = [r["tau"] for r in pt]
    ax.plot(xs, [r["gold_recall_mean"] for r in pt], "o-", color=c_psv, label="gold recall")
    ax.set_xlabel("lexical pre-screen threshold τ")
    ax.set_ylabel("gold documents read (recall)")
    ax2 = ax.twinx()
    ax2.plot(xs, [r["tokens_med"] / 1000 for r in pt], "^:", color="#4393c3", label="tokens")
    ax2.set_ylabel("tokens per task (k)", color="#4393c3")
    ax.set_ylim(0.3, 0.8)
    ax.set_title("(c) Pre-screen: tokens fall before recall does")

    fig.suptitle("Sensitivity of the three variant conclusions (simulation; parameters to be measured on real trajectories)",
                 fontsize=9, y=1.02, color="#444444")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_sim_sensitivity.{ext}"), bbox_inches="tight",
                    dpi=220 if ext == "png" else None)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def make_plots(per_design: Dict[str, List[Result]], summary: Dict[str, Dict[str, float]],
               P: Params, out: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110,
    })
    c_react, c_react2, c_psv, c_var = "#b2182b", "#ef8a62", "#2166ac", "#4393c3"
    colors = {"react": c_react, "react2x": c_react2, "psv": c_psv}
    order = list(DESIGNS)
    markers = {"react": "o", "react2x": "^", "psv": "s", "psv_T": "D", "psv_L": "v", "psv_D": "<",
               "psv_M": ">", "psv_R1": "P", "psv_S": "X", "psv_lean": "*"}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) cumulative tokens vs documents opened: ReAct quadratic, PSV linear
    ax = ax_a
    jit = np.random.default_rng(1)
    for name, c, mk in (("react", c_react, "o"), ("react2x", c_react2, "^"), ("psv", c_psv, "s")):
        rows = per_design[name]
        x = np.array([r.docs_opened for r in rows]) + jit.normal(0, 0.12, len(rows))
        y = np.array([r.tokens for r in rows]) / 1000
        ax.scatter(x, y, s=7, alpha=0.25, color=c, marker=mk, label=f"{LABELS[name]} (simulated tasks)")
    d = np.arange(1, 31)
    L = float(np.exp(P.doc_len_mu + P.doc_len_sigma ** 2 / 2))
    S = 2 * d                                             # one search per opened document
    delta = (P.react_search_k * P.react_search_tok + L + 2 * P.react_step_out) / 2
    react_curve = (P.base_ctx * S + delta * S * (S - 1) / 2 + (P.base_ctx + S * delta)
                   + S * P.react_step_out + P.react_final_out) / 1000
    cards = 1.0 * d                                       # about one gate-passing card per document
    psv_curve = (P.base_ctx + P.N * P.catalog_tok_per_doc + P.planner_out
                 + d * (P.scout_in_overhead / P.docs_per_scout + L + 1.0 * P.card_tok)
                 + 2 * (P.writer_in_overhead + cards * (P.writer_card_tok + P.writer_doc_meta))
                 + P.writer_out + P.revision_out) / 1000
    ax.plot(d, react_curve, color=c_react, lw=1.4, ls="--",
            label="ReAct cost model: whole context re-read at every step (∝ d²)")
    ax.plot(d, psv_curve, color=c_psv, lw=1.4, ls="--",
            label="PSV cost model: each document read once, bounded roles (∝ d)")
    ax.axhline(P.react_tok_budget / 1000, color="grey", lw=0.8, ls=":")
    ax.text(29.5, P.react_tok_budget / 1000 + 8, "ReAct token budget", color="grey", fontsize=7, ha="right")
    ax.set_xlabel("documents opened per task")
    ax.set_ylabel("total tokens per task (k)")
    ax.set_title("(a) Why the budget binds: context accumulation vs. bounded roles")
    ax.set_xlim(0, 30); ax.set_ylim(0, 420)
    ax.legend(loc="upper left", frameon=False)

    # (b) Pareto: gold recall vs tokens, with IQR bars
    ax = ax_b
    for name in order:
        s = summary[name]
        c = colors.get(name, c_var)
        ax.errorbar(s["tokens_med"] / 1000, s["gold_recall_mean"],
                    xerr=[[(s["tokens_med"] - s["tokens_q1"]) / 1000], [(s["tokens_q3"] - s["tokens_med"]) / 1000]],
                    yerr=s["gold_recall_sd"] / np.sqrt(s["n"]) * 1.96,
                    fmt=markers[name], color=c, ms=7 if name != "psv_lean" else 12, capsize=2, lw=0.8,
                    label=LABELS[name], zorder=3)
    ann = {"react": ("ReAct", (10, 4), "left"), "react2x": ("ReAct 2× budget", (-10, 8), "right"),
           "psv": ("PSV  (-T, -D, -S overlap)", (10, -14), "left"), "psv_M": ("PSV-M mention-following", (8, 6), "left"),
           "psv_lean": ("PSV-lean (T+L+D+M)", (-4, -16), "center"), "psv_L": ("PSV-L pre-screen", (6, -16), "left"),
           "psv_R1": ("PSV-R1 one re-plan", (8, -6), "left")}
    for name, (txt, off, ha) in ann.items():
        s = summary[name]
        ax.annotate(txt, (s["tokens_med"] / 1000, s["gold_recall_mean"]),
                    textcoords="offset points", xytext=off, fontsize=7.5, ha=ha,
                    color=colors.get(name, c_var))
    ax.set_xlabel("total tokens per task (k, median; bars = IQR)")
    ax.set_ylabel("gold documents read (recall, mean; bars = 95% CI)")
    ax.set_title("(b) Cost–recall frontier of the design variants")
    ax.set_ylim(0.3, 1.0); ax.set_xlim(40, 310)

    # (c) wall time with serial sum
    ax = ax_c
    y = np.arange(len(order))
    cols = [colors.get(n, c_var) for n in order]
    wall = [summary[n]["wall_med"] for n in order]
    serial = [summary[n]["serial_med"] for n in order]
    ax.barh(y, serial, height=0.6, color="none", edgecolor=cols, lw=0.9, ls="--",
            label="serial sum of call latencies")
    ax.barh(y, wall, height=0.6, color=cols, alpha=0.85, label="wall time (parallel scouts)")
    for yi, w, sr in zip(y, wall, serial):
        ax.text(max(w, sr) + 4, yi, f"{w:.0f} s" + (f"  (×{sr / w:.1f} overlap)" if sr / w > 1.05 else ""),
                va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([LABELS[n] for n in order]); ax.invert_yaxis()
    ax.set_xlabel("seconds per task (median)")
    ax.set_xlim(0, max(serial) * 1.45)
    ax.set_title("(c) Latency: what parallel scouting buys")
    ax.legend(loc="upper right", frameon=False)

    # (d) dollars
    ax = ax_d
    cost = [summary[n]["usd_med"] for n in order]
    ax.barh(y, cost, height=0.6, color=cols, alpha=0.85)
    for yi, c in zip(y, cost):
        ax.text(c + 0.003, yi, f"\\${c:.3f}", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlabel("USD per task (median; strong tier \\$1/M in, \\$4/M out; cheap tier ×0.2)")
    ax.set_xlim(0, max(cost) * 1.25)
    ax.set_title("(d) Price: where tiered scouts help")

    fig.suptitle("Design-space simulation — abstract cost model calibrated to Drafts48 medians; not an LLM experiment",
                 fontsize=9.5, y=0.995, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_sim_frontier.{ext}"), bbox_inches="tight",
                    dpi=220 if ext == "png" else None)
    plt.close(fig)

    # second figure: where the gains come from (docs, hidden gold, cards, calls)
    fig, axes2 = plt.subplots(1, 4, figsize=(12.5, 3.4))
    metrics = [("docs_med", "documents opened (median)"), ("hidden_recall_mean", "hidden-gold recall (mean)"),
               ("cards_med", "verified cards reaching the writer"), ("calls_med", "model calls per task")]
    for ax, (key, title) in zip(axes2, metrics):
        vals = [summary[n][key] for n in order]
        ax.barh(y, vals, color=cols, alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels([LABELS[n] for n in order] if ax is axes2[0] else [])
        ax.invert_yaxis(); ax.set_title(title, fontsize=9)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.2f}" if v < 2 else f" {v:.0f}", va="center", fontsize=7)
        ax.set_xlim(0, max(vals) * 1.28)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_sim_breakdown.{ext}"), bbox_inches="tight",
                    dpi=220 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()
