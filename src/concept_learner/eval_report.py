from collections import Counter, defaultdict
import math


def _bin_calibration(probs, labels, bins=10):
    # Expected Calibration Error (ECE)
    if not probs:
        return 0.0, 0.0
    buckets = [[] for _ in range(bins)]
    for p, y in zip(probs, labels):
        i = min(bins - 1, int(p * bins))
        buckets[i].append((p, y))
    ece, brier = 0.0, 0.0
    n = len(probs)
    for p, y in zip(probs, labels):
        brier += (p - y) ** 2
    brier /= max(1, n)
    for bucket in buckets:
        if not bucket:
            continue
        conf = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        w = len(bucket) / n
        ece += w * abs(conf - acc)
    return ece, brier


def _safe_div(a, b):
    return a / b if b else 0.0


def _prf(tp, fp, fn):
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def canonicalize(q):
    # Optional: pass in parsed meta to render canonical; keep trivial fallback.
    return q.get("canon") or q.get("english") or q.get("text") or ""


def print_eval_report(rows, threshold_used):
    """
    rows: list of dicts with keys:
      rel:str, gold:int(0/1), p_yes:float, pred:int(0/1),
      a:int(optional), b:int(optional), english:str(optional), canon:str(optional),
      gold_rel:str(optional), pred_rel:str(optional) for multi-rel confusion (if available)
    threshold_used: float
    """
    N = len(rows)
    golds = [r["gold"] for r in rows]
    preds = [r["pred"] for r in rows]
    pyes = [r["p_yes"] for r in rows]
    micro_acc = sum(int(g == p) for g, p in zip(golds, preds)) / max(1, N)
    ece_all, brier_all = _bin_calibration(pyes, golds, bins=12)

    # Per-relation stats
    rels = sorted(set(r["rel"] for r in rows))
    per = {}
    f1s = []
    for rel in rels:
        sub = [r for r in rows if r["rel"] == rel]
        y = [r["gold"] for r in sub]
        yhat = [r["pred"] for r in sub]
        py = [r["p_yes"] for r in sub]
        supp = len(sub)
        acc = sum(int(a == b) for a, b in zip(y, yhat)) / max(1, supp)
        tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0)
        prec, rec, f1 = _prf(tp, fp, fn)
        ece, _ = _bin_calibration(py, y, bins=8)
        avg_p = sum(py) / max(1, len(py))
        per[rel] = dict(
            support=supp, acc=acc, prec=prec, rec=rec, f1=f1, avg_p_yes=avg_p, ece=ece
        )
        f1s.append(f1)
    macro_f1 = sum(f1s) / max(1, len(f1s))
    macro_acc = sum(per[r]["acc"] for r in rels) / max(1, len(rels))

    # Confusions across relations if gold_rel/pred_rel present
    conf = defaultdict(Counter)
    if all(("gold_rel" in r and "pred_rel" in r) for r in rows):
        for r in rows:
            conf[r["gold_rel"]][r["pred_rel"]] += 1

    # Difficulty slices (examples)
    def is_close(r):
        if "a" in r and "b" in r and r["rel"] in ("greater", "smaller"):
            return abs(int(r["a"]) - int(r["b"])) <= 1
        return None

    slices = {
        "greater_close": [r for r in rows if r["rel"] == "greater" and is_close(r) is True],
        "greater_far": [r for r in rows if r["rel"] == "greater" and is_close(r) is False],
        "smaller_close": [r for r in rows if r["rel"] == "smaller" and is_close(r) is True],
        "smaller_far": [r for r in rows if r["rel"] == "smaller" and is_close(r) is False],
    }

    # Magnitude buckets (0–9 / 10–19 / 20+)
    def bucket(n):
        if n is None:
            return None
        n = int(n)
        if 0 <= n <= 9:
            return "0-9"
        if 10 <= n <= 19:
            return "10-19"
        return "20+"

    mag = defaultdict(list)
    for r in rows:
        if "a" in r:
            mag["a:" + str(bucket(r.get("a")))].append(r)
        if "b" in r:
            mag["b:" + str(bucket(r.get("b")))].append(r)

    # Parity and edge slices
    parity = defaultdict(list)
    edges = defaultdict(list)
    edge_vals = {0, 9, 10, 99}
    for r in rows:
        a = r.get("a")
        b = r.get("b")
        if a is not None:
            parity[f"a:{int(a)%2}"] .append(r)
            if int(a) in edge_vals:
                edges[f"a:{int(a)}"].append(r)
        if b is not None:
            parity[f"b:{int(b)%2}"] .append(r)
            if int(b) in edge_vals:
                edges[f"b:{int(b)}"].append(r)

    # Hardest mistakes (highest-confidence wrong answers)
    wrong = [r for r in rows if r["gold"] != r["pred"]]
    wrong_sorted = sorted(
        wrong,
        key=lambda r: r["p_yes"] if r["pred"] == 1 else (1 - r["p_yes"]),
        reverse=True,
    )[:12]

    # ---- Print section ----
    print(f"Eval N={N}")
    print(
        f"Overall: micro_acc={micro_acc:.3f}  macro_acc={macro_acc:.3f}  macro_F1={macro_f1:.3f}"
    )
    print(
        f"Calibration: Brier={brier_all:.3f}  ECE={ece_all:.3f}  thr_used={threshold_used:.2f}"
    )
    print("\nPer-relation:")
    print("  rel              supp   acc    P      R      F1     avg_p   ECE")
    for rel in rels:
        s = per[rel]
        print(
            f"  {rel:14s} {s['support']:5d}  {s['acc']:.3f}  {s['prec']:.3f}  {s['rec']:.3f}  {s['f1']:.3f}  {s['avg_p_yes']:.3f}  {s['ece']:.3f}"
        )

    if conf:
        print("\nTop confusions (gold → pred count):")
        for g in rels:
            row = conf[g]
            if not row:
                continue
            common = row.most_common(3)
            pairs = ", ".join([f"{p}:{c}" for p, c in common if p != g])
            if pairs:
                print(f"  {g:14s} → {pairs}")

    print("\nDifficulty slices:")
    for k, v in slices.items():
        if not v:
            continue
        acc = sum(int(r['gold'] == r['pred']) for r in v) / len(v)
        print(f"  {k:16s} n={len(v):3d} acc={acc:.3f}")

    print("\nMagnitude buckets:")
    for k, v in mag.items():
        if not v:
            continue
        acc = sum(int(r['gold'] == r['pred']) for r in v) / len(v)
        print(f"  {k:6s}  n={len(v):3d} acc={acc:.3f}")

    if parity:
        print("\nParity buckets:")
        for k, v in parity.items():
            acc = sum(int(r['gold'] == r['pred']) for r in v) / max(1, len(v))
            print(f"  {k:4s}  n={len(v):3d} acc={acc:.3f}")

    if edges:
        print("\nEdge numbers:")
        for k, v in edges.items():
            acc = sum(int(r['gold'] == r['pred']) for r in v) / max(1, len(v))
            print(f"  {k:6s} n={len(v):3d} acc={acc:.3f}")

    if wrong_sorted:
        print("\nHardest mistakes (high-confidence wrong):")
        for r in wrong_sorted:
            conf = r["p_yes"] if r["pred"] == 1 else (1 - r["p_yes"])
            q = canonicalize(r)
            print(
                f"  rel={r['rel']:12s} gold={r['gold']} pred={r['pred']} conf={conf:.3f} :: {q}"
            )

