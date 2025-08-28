import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Literal, Dict, Any
from concept_learner.reasoning_ops import (
    TypedState,
    OpBase,
    OpFilterMLP,
    OpCount,
    OpCompareGeneric,
    OpNAC,
    OpNALU,
    OpFunction,
    StepSpec,
    FnCandidate,
    OpSubstituteMask,
    OpLet,
    OpLoad,
)


# --- ReasonerV2 ------------------------------------------------------------


class ReasonerV2(nn.Module):
    """
    Multi-step reasoning with a soft mixture-of-operators per step.
    Keeps your STOP head and broadcast.
    """

    def __init__(
        self,
        d_model: int,
        max_steps: int = 4,
        temperature: float = 1.0,
        lambda_sparse: float = 1e-3,
        lambda_halt: float = 1e-3,
        # v2.5 additions
        ops: List[OpBase] | None = None,  # primitives
        use_functions: bool = True,
        num_fn_slots: int = 16,
        fn_max_body_len: int = 4,
        fn_allow_higher_order: bool = True,
        fn_allow_self_call: bool = True,
        exec_mode: Literal["macro", "micro"] = "macro",
        wake_sleep: bool = False,
        # wake/sleep knobs
        trace_topk: int = 1,
        mine_min_support: int = 20,
        mine_max_len: int = 3,
        mdl_gain_threshold: float = 0.0,
        prune_threshold: float = 0.02,
        mdl_alpha: float = 1.0,
        dream_replay_mix: float = 0.5,
    ):
        super().__init__()
        # primitives
        if ops is None:
            # Minimal primitive core: rewriting, arithmetic, compare, and small env.
            prims = [
                OpSubstituteMask(d_model),  # Rewrite mask based on H,z
                OpCount(d_model, learn_phi=False),  # Aggregator from set->scalar
                OpNAC(d_model),  # NAC for add/sub on scalars
                OpNALU(d_model),  # NALU for mul/div/pow on scalars
                OpCompareGeneric(d_model, use_z=True),  # boolean guard
                OpLet(d_model),  # env write (registers)
                OpLoad(d_model),  # env read
            ]
        else:
            prims = list(ops)
        assert len(prims) >= 1, "Need at least one primitive operator"
        self.d = d_model
        self.prims = nn.ModuleList(prims)
        self.max_steps = max_steps
        self.temperature = float(temperature)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_halt = float(lambda_halt)

        # optional function library
        self.use_functions = bool(use_functions)
        self.num_fn_slots = int(num_fn_slots) if use_functions else 0
        self.exec_mode = exec_mode
        if self.use_functions:
            self.op_function = OpFunction(
                d_model=d_model,
                base_ops=self.prims,  # pass primitives only
                num_slots=num_fn_slots,
                max_body_len=fn_max_body_len,
                allow_higher_order=fn_allow_higher_order,
                allow_self_call=fn_allow_self_call,
            )
            # Install a small standard library of higher-order-like functions
            self._install_std_lib()
        else:
            self.op_function = None

        # token featurizer for initial state
        self.phi = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        # gating + stop + broadcast
        total_actions = len(self.prims) + (self.num_fn_slots if self.use_functions else 0)
        self.to_action = nn.Linear(2 * d_model, total_actions)  # logits over actions (prims + fn)
        self.to_stop = nn.Linear(2 * d_model, 1)
        self.broadcast = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        # optional state stabilizer (delta & gate, like your v1)
        self.to_delta = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.to_gate = nn.Linear(2 * d_model, d_model)

        # small head to read an initial scalar value from the question tokens
        # val0 = read_number_from_tokens(H, mask)
        self.read_number = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )

        # --- Wake/Sleep bookkeeping ---
        self.wake_sleep = bool(wake_sleep)
        self.trace_topk = int(trace_topk)
        self.mine_min_support = int(mine_min_support)
        self.mine_max_len = int(mine_max_len)
        self.mdl_gain_threshold = float(mdl_gain_threshold)
        self.prune_threshold = float(prune_threshold)
        self.mdl_alpha = float(mdl_alpha)
        self.dream_replay_mix = float(dream_replay_mix)
        # slot usage EMA
        if self.use_functions:
            self.register_buffer(
                "_slot_usage_ema",
                torch.zeros(self.num_fn_slots),
                persistent=False,
            )
        else:
            self._slot_usage_ema = None

    def forward(
        self, H: torch.Tensor, z: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns:
          H_reasoned:  (B,T,d)
          s_final:     (B,d)
          stop_logits: (B, max_steps)
          action_logits: (B, max_steps, num_ops)
        """
        B, T, d = H.shape
        m = mask.float().unsqueeze(-1)

        # initial typed state (DeepSets pool)
        H_phi = self.phi(H)
        s = (H_phi * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        # initialize typed scalar streams
        mask_float = mask.float().clamp(0.0, 1.0)  # start with the input padding mask
        # read initial number (scalar) from the question tokens
        h_pool = (H_phi * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        val0 = self.read_number(h_pool)  # (B,1)
        val = val0
        boolean = H.new_zeros(B, 1)  # no boolean decision yet
        # small register file (env)
        env = H.new_zeros(B, 4)

        stop_logits_all = []
        action_logits_all = []
        # track per-example STOP
        done = torch.zeros(B, dtype=torch.bool, device=H.device)
        # cumulative halting probability (for variable compute)
        cum_halt = torch.zeros(B, 1, device=H.device)
        # sparsity and over-halt penalties accumulators
        pen_sparse = []
        pen_halt = []

        # trace per example
        traces: List[List[Tuple[str, int]]] = [[] for _ in range(B)] if self.wake_sleep else None
        # micro-exec stacks (per-example), only used if exec_mode=="micro"
        stacks: List[list] = [[] for _ in range(B)] if (self.use_functions and self.exec_mode == "micro") else None
        micro_noop_flag = None

        for step in range(self.max_steps):
            x = torch.cat([s, z], dim=-1)  # (B,2d)

            # ----- action logits over ops -----
            logits_action = self.to_action(x)  # (B, total_actions)
            # mask empty function slots if enabled
            if self.use_functions and self.op_function is not None and self.num_fn_slots > 0:
                with torch.no_grad():
                    non_empty = torch.tensor(
                        [1.0 if len(slt.steps) > 0 else 0.0 for slt in self.op_function.slots],
                        device=H.device,
                    )
                # Build mask vector: [zeros for prims, -inf for empty fn slots]
                fn_mask = (non_empty > 0).float()
                # logits add: (0 for prims, -1e9 for empty fns)
                pad_prim = torch.zeros(B, len(self.prims), device=H.device)
                pad_fn = (fn_mask.view(1, -1).expand(B, -1) - 1).clamp(max=0) * 1e9
                pad = torch.cat([pad_prim, pad_fn], dim=-1)
                logits_action = logits_action + pad

            probs_action = torch.softmax(logits_action, dim=-1)
            # sparsity penalty: encourage peaky distribution (L1-like)
            pen_sparse.append((probs_action.sum(dim=-1) - probs_action.max(dim=-1).values).mean())
            # Discrete operator selection per step (base pick)
            if self.training:
                pick_base = F.gumbel_softmax(
                    logits_action, tau=self.temperature, hard=True, dim=-1
                )  # (B, total_actions) one-hot
            else:
                idx = logits_action.argmax(-1)  # (B,)
                pick_base = F.one_hot(idx, num_classes=logits_action.size(-1)).float()

            # ----- run all typed-state ops -----
            ts = TypedState(H=H, mask=mask_float, val=val, boolean=boolean, env=env)

            cand_masks, cand_vals, cand_bools, cand_envs = [], [], [], []
            # primitives
            for op in self.prims:
                m_i, v_i, b_i = op(ts, z)  # m_i:(B,T), v_i:(B,1), b_i:(B,1)
                cand_masks.append(m_i)
                cand_vals.append(v_i)
                cand_bools.append(b_i)
                # optional environment update side-channel
                if hasattr(op, "env_update") and callable(getattr(op, "env_update")):
                    e_i = op.env_update(ts, z)
                    if e_i is None:
                        e_i = env
                else:
                    e_i = env
                cand_envs.append(e_i)
            # functions as macro actions
            if self.use_functions and self.op_function is not None and self.num_fn_slots > 0:
                for k in range(self.num_fn_slots):
                    m_i, v_i, b_i, e_i = self.op_function.run_slot_env(k, ts, z)
                    cand_masks.append(m_i)
                    cand_vals.append(v_i)
                    cand_bools.append(b_i)
                    cand_envs.append(e_i)

            M = torch.stack(cand_masks, dim=-1)  # (B, T, num_ops)
            V = torch.stack(cand_vals, dim=-1)  # (B, 1, num_ops)
            Bv = torch.stack(cand_bools, dim=-1)  # (B, 1, num_ops)
            E = torch.stack(cand_envs, dim=-1)  # (B, K, num_ops)

            # ----- select operator outputs -----
            pick = pick_base
            if self.use_functions and self.exec_mode == "micro" and stacks is not None and self.op_function is not None and self.num_fn_slots > 0:
                pick = pick_base.clone()
                micro_noop = torch.zeros(B, dtype=torch.bool, device=H.device)
                for bi in range(B):
                    st = stacks[bi]
                    if len(st) == 0:
                        # If base pick chose a function, enter micro and do no-op this step
                        sel = int(pick_base[bi].argmax().item())
                        if sel >= len(self.prims):
                            sid = sel - len(self.prims)
                            stacks[bi].append({"sid": sid, "pc": 0, "depth": 1})
                            pick[bi].zero_()
                            micro_noop[bi] = True
                        # else: leave pick as-is
                        continue
                    # Have an active frame; follow its StepSpec
                    frame = st[-1]
                    sid = frame["sid"]
                    pc = frame["pc"]
                    steps_list = self.op_function.slots[sid].steps
                    if pc >= len(steps_list):
                        st.pop()
                        pick[bi].zero_(); micro_noop[bi] = True
                        continue
                    spec = steps_list[pc]
                    if spec.kind == "return":
                        st.pop(); pick[bi].zero_(); micro_noop[bi] = True
                        continue
                    if spec.kind == "function":
                        callee = int(spec.idx) if spec.idx is not None else -1
                        if callee >= 0 and (callee != sid or self.op_function.allow_self_call) and self.op_function.allow_higher_order:
                            st.append({"sid": callee, "pc": 0, "depth": frame.get("depth", 1) + 1})
                        pick[bi].zero_(); micro_noop[bi] = True
                        continue
                    if spec.kind == "primitive":
                        prim_idx = int(spec.idx) if spec.idx is not None else -1
                        if prim_idx < 0 or prim_idx >= len(self.prims):
                            pick[bi].zero_(); micro_noop[bi] = True
                            continue
                        one = torch.zeros_like(pick[bi])
                        one[prim_idx] = 1.0
                        pick[bi] = one
                        frame["pc"] = pc + 1
                micro_noop_flag = micro_noop

            new_mask = torch.einsum("btn,bn->bt", M, pick)  # (B, T)
            new_val = torch.einsum("bin,bn->bi", V, pick)  # (B, 1)
            new_bool = torch.einsum("bin,bn->bi", Bv, pick)  # (B, 1)
            new_env = torch.einsum("bkn,bn->bk", E, pick)  # (B, K)

            # freeze items that are already done
            if done.any():
                mask_float = torch.where(done.unsqueeze(-1), mask_float, new_mask)
                val = torch.where(done.unsqueeze(-1), val, new_val)
                boolean = torch.where(done.unsqueeze(-1), boolean, new_bool)
                env = torch.where(done.unsqueeze(-1), env, new_env)
            else:
                mask_float = new_mask
                val = new_val
                boolean = new_bool
                env = new_env

            # If micro step did no-op for some examples, restore previous state
            if micro_noop_flag is not None and micro_noop_flag.any():
                prev_mask, prev_val, prev_bool, prev_env = ts.mask, ts.val, ts.boolean, ts.env
                keep = micro_noop_flag
                mask_float = torch.where(keep.unsqueeze(-1), prev_mask, mask_float)
                val = torch.where(keep.unsqueeze(-1), prev_val, val)
                boolean = torch.where(keep.unsqueeze(-1), prev_bool, boolean)
                env = torch.where(keep.unsqueeze(-1), prev_env, env)
                micro_noop_flag = None

            # logging usage and tracing
            if self.wake_sleep and self.use_functions and self.op_function is not None and self.num_fn_slots > 0:
                # usage ema update: per-slot probability mass picked (or hard pick in eval)
                with torch.no_grad():
                    fn_pick = pick[:, len(self.prims) :]
                    mass = fn_pick.mean(dim=0)  # (num_fn,)
                    if self._slot_usage_ema is not None and mass.numel() == self._slot_usage_ema.numel():
                        self._slot_usage_ema.mul_(0.95).add_(0.05 * mass)
                if traces is not None:
                    # append greedy action id per example
                    with torch.no_grad():
                        # Use base pick for logging to reflect controller's choice
                        ids = pick_base.argmax(dim=-1).tolist()
                        for bi, a in enumerate(ids):
                            if a < len(self.prims):
                                traces[bi].append(("prim", int(a)))
                            else:
                                traces[bi].append(("func", int(a - len(self.prims))))

            # ----- recompute pooled state s from updated mask -----
            H_phi = self.phi(H)  # (B, T, d)
            denom = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            s_new = (H_phi * mask_float.unsqueeze(-1)).sum(dim=1) / denom  # (B, d)
            if done.any():
                s = torch.where(done.unsqueeze(-1), s, s_new)
            else:
                s = s_new

            # In micro mode: optional RETURN head for slots with policy 'return_head'
            if self.use_functions and self.exec_mode == "micro" and stacks is not None and self.op_function is not None and self.num_fn_slots > 0:
                with torch.no_grad():
                    for bi in range(B):
                        st = stacks[bi]
                        if len(st) == 0:
                            continue
                        top = st[-1]
                        sid = top["sid"]
                        slot = self.op_function.slots[sid]
                        if slot.ret_policy == "return_head":
                            rlog = self.op_function.return_logit(s[bi : bi + 1], z[bi : bi + 1])
                            if torch.sigmoid(rlog).item() > 0.5:
                                st.pop()

            # STOP head (per-example)
            stop_logit = self.to_stop(torch.cat([s, z], dim=-1))  # (B,1)
            stop_logits_all.append(stop_logit)
            action_logits_all.append(logits_action.unsqueeze(1))  # (B,1,total_actions)

            p_stop = torch.sigmoid(stop_logit)  # (B,1)
            cum_halt = cum_halt + p_stop
            # over-halt penalty when cumulative exceeds 1.0
            pen_halt.append((cum_halt - 1.0).clamp_min(0.0).mean())
            done = done | (cum_halt.squeeze(-1) >= 1.0)

        stop_logits = torch.cat(stop_logits_all, dim=1)  # (B, max_steps)
        action_logits = torch.cat(action_logits_all, dim=1)  # (B, max_steps, total_actions)

        # broadcast final state to tokens
        s_rep = s.unsqueeze(1).expand(B, T, d)
        H_reasoned = H + self.broadcast(torch.cat([H, s_rep], dim=-1))

        # expose final scalar value stream for downstream decoders
        self._last_val = val
        # expose auxiliary losses for trainer
        self._aux_loss = {
            "sparse": (torch.stack(pen_sparse).mean() if len(pen_sparse) > 0 else torch.tensor(0.0, device=H.device)),
            "halt_over": (torch.stack(pen_halt).mean() if len(pen_halt) > 0 else torch.tensor(0.0, device=H.device)),
        }

        # expose traces for wake/sleep
        if self.wake_sleep:
            self._last_traces = traces
        else:
            self._last_traces = None

        # telemetry summary
        telem = {
            "num_prims": len(self.prims),
            "num_fn_slots": self.num_fn_slots,
        }
        if self.use_functions and self.op_function is not None and self.num_fn_slots > 0:
            nonempty = sum(1 for s_ in self.op_function.slots if len(s_.steps) > 0)
            telem["fn_nonempty"] = int(nonempty)
            try:
                flat_lens = [len(self.op_function._flat_bodies[i]) for i in range(self.num_fn_slots)]
                telem["avg_flat_len"] = float(sum(flat_lens) / max(1, len(flat_lens)))
            except Exception:
                telem["avg_flat_len"] = 0.0
            if self._slot_usage_ema is not None:
                telem["slot_usage_ema"] = self._slot_usage_ema.detach().cpu()
        if self.use_functions and self.exec_mode == "micro" and stacks is not None:
            depths = [len(s_) for s_ in stacks]
            telem["stack_depths"] = depths
            telem["max_stack_depth"] = max(depths) if len(depths) > 0 else 0
        self._telemetry = telem

        return H_reasoned, s, stop_logits, action_logits

    # --- Standard Library Installation ---
    def _install_std_lib(self):
        if self.op_function is None:
            return
        # map operator names to primitive indices
        name_to_idx = {op.name: i for i, op in enumerate(self.prims)}
        get = lambda n: name_to_idx.get(n, None)
        idx_filter = get("substitute_mask")
        idx_nac = get("nac")
        idx_nalu = get("nalu")
        idx_cmp = get("cmp_generic")
        idx_let = get("let")
        idx_load = get("load")

        # 1) Filter wrapper (implemented via substitute_mask primitive)
        steps_filter = []
        if idx_filter is not None:
            steps_filter.append(StepSpec(kind="primitive", idx=idx_filter))
        steps_filter.append(StepSpec(kind="return"))
        ids = self.op_function.install([FnCandidate(steps=steps_filter)])
        sid_filter = ids[0] if ids else 0

        # 2) No-op wrapper (for else branch)
        steps_noop = [StepSpec(kind="return")]
        sid_noop = self.op_function.install([FnCandidate(steps=steps_noop)])[0]

        # 3) If(compare) then Filter else No-op
        steps_if = []
        if idx_cmp is not None:
            steps_if.append(StepSpec(kind="primitive", idx=idx_cmp))
        steps_if.append(StepSpec(kind="cond", idx=sid_filter, idx2=sid_noop))
        steps_if.append(StepSpec(kind="return"))
        self.op_function.install([FnCandidate(steps=steps_if)])

        # 4) Fold step: load -> NAC -> let -> return
        steps_fold = []
        if idx_load is not None:
            steps_fold.append(StepSpec(kind="primitive", idx=idx_load))
        if idx_nac is not None:
            steps_fold.append(StepSpec(kind="primitive", idx=idx_nac))
        if idx_let is not None:
            steps_fold.append(StepSpec(kind="primitive", idx=idx_let))
        steps_fold.append(StepSpec(kind="return"))
        self.op_function.install([FnCandidate(steps=steps_fold)])

        # 5) Map-like transform: NALU -> LET (write transformed val) -> return
        steps_map = []
        if idx_nalu is not None:
            steps_map.append(StepSpec(kind="primitive", idx=idx_nalu))
        if idx_let is not None:
            steps_map.append(StepSpec(kind="primitive", idx=idx_let))
        steps_map.append(StepSpec(kind="return"))
        self.op_function.install([FnCandidate(steps=steps_map)])

    # --- Sleep-time utilities -------------------------------------------------
    @staticmethod
    def mine_candidates(
        traces: List[Dict[str, Any]] | List[List[Tuple[str, int]]],
        min_support: int = 20,
        max_len: int = 3,
        topM: int = 32,
        mdl_alpha: float = 1.0,
        mdl_gain_threshold: float = 0.0,
        existing_patterns: List[Tuple[int, ...]] | None = None,
    ) -> List[FnCandidate]:
        from collections import Counter

        # Normalize traces structure
        seqs: List[List[int]] = []
        for tr in traces or []:
            if isinstance(tr, dict) and "actions" in tr:
                seq = [a[1] for a in tr["actions"] if a[0] == "prim"]
            else:
                seq = [a[1] for a in tr if a[0] == "prim"]
            if len(seq) > 1:
                seqs.append(seq)
        C = Counter()
        for seq in seqs:
            n = len(seq)
            for L in range(2, max_len + 1):
                for i in range(0, n - L + 1):
                    C[tuple(seq[i : i + L])] += 1
        pats = [(pat, cnt) for pat, cnt in C.items() if cnt >= min_support]
        # MDL gain: gain = c*(L-1) - (L + alpha)
        scored = []
        exist = set(existing_patterns or [])
        for pat, cnt in pats:
            L = len(pat)
            gain = cnt * (L - 1) - (L + mdl_alpha)
            if gain > mdl_gain_threshold and tuple(pat) not in exist and L >= 2:
                scored.append((gain, pat, cnt))
        scored.sort(key=lambda x: (x[0], x[2], len(x[1])), reverse=True)
        cands: List[FnCandidate] = []
        for (gain, pat, cnt) in scored[:topM]:
            steps = [StepSpec(kind="primitive", idx=j) for j in pat]
            cands.append(FnCandidate(steps=steps, ret_policy="implicit"))
        return cands

    def sleep_abstraction(self, mdl_gain_threshold: float | None = None) -> List[int]:
        if not self.use_functions or self.op_function is None:
            return []
        traces = []
        if getattr(self, "_last_traces", None) is not None:
            for t in self._last_traces:
                traces.append({"actions": t})
        existing = self._existing_primitive_patterns()
        cands = self.mine_candidates(
            traces,
            min_support=self.mine_min_support,
            max_len=self.mine_max_len,
            mdl_alpha=self.mdl_alpha,
            mdl_gain_threshold=(self.mdl_gain_threshold if mdl_gain_threshold is None else mdl_gain_threshold),
            existing_patterns=existing,
        )
        installed = self.op_function.install(cands)
        # prune based on usage ema
        if self._slot_usage_ema is not None:
            usage = self._slot_usage_ema.detach().cpu().tolist()
            self.op_function.prune(usage, self.prune_threshold)
        return installed

    def _existing_primitive_patterns(self) -> List[Tuple[int, ...]]:
        pats: List[Tuple[int, ...]] = []
        if not self.use_functions or self.op_function is None:
            return pats
        for slot in self.op_function.slots:
            seq: List[int] = []
            ok = True
            for st in slot.steps:
                if st.kind == "primitive" and st.idx is not None:
                    seq.append(int(st.idx))
                elif st.kind == "return":
                    break
                else:
                    ok = False
                    break
            if ok and len(seq) >= 2:
                pats.append(tuple(seq))
        return pats
