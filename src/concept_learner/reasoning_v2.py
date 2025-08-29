import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Literal, Dict, Any
from concept_learner.reasoning_ops import (
    TypedState,
    OpBase,
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
        # action selection knobs
        epsilon: float = 0.03,
        topk_actions: int = 4,
        action_entropy_weight: float = 1e-3,
        stop_bias: float = -0.5,
        # function install filters
        mine_min_body_len: int = 2,
        mine_min_effect: float = 1e-3,
        mine_cosine_max: float = 0.90,
        mine_overlap_max: float = 0.70,
        # post-install bias
        fn_bias_init: float = 1.0,
        fn_bias_decay: float = 0.995,
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
        # new knobs
        self.epsilon = float(epsilon)
        self.topk_actions = int(topk_actions)
        self.action_entropy_weight = float(action_entropy_weight)
        self.stop_bias = float(stop_bias)

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
        # install filters
        self.mine_min_body_len = int(mine_min_body_len)
        self.mine_min_effect = float(mine_min_effect)
        self.mine_cosine_max = float(mine_cosine_max)
        self.mine_overlap_max = float(mine_overlap_max)
        # slot usage EMA
        if self.use_functions:
            self.register_buffer(
                "_slot_usage_ema",
                torch.zeros(self.num_fn_slots),
                persistent=False,
            )
        else:
            self._slot_usage_ema = None
        # post-install function action logit bias (decays over time)
        if self.use_functions:
            self.register_buffer(
                "_fn_post_bias",
                torch.zeros(self.num_fn_slots),
                persistent=False,
            )
        else:
            self._fn_post_bias = None  # type: ignore
        self.fn_bias_init = float(fn_bias_init)
        self.fn_bias_decay = float(fn_bias_decay)

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
        ent_list = []

        # trace per example
        traces: List[List[Tuple[str, int]]] = [[] for _ in range(B)] if self.wake_sleep else None
        # micro-exec stacks (per-example), only used if exec_mode=="micro"
        stacks: List[list] = [[] for _ in range(B)] if (self.use_functions and self.exec_mode == "micro") else None
        max_depth_seen = [0 for _ in range(B)] if stacks is not None else None
        # per-batch function call histogram (base controller selections)
        fn_call_hist = torch.zeros(self.num_fn_slots, device=H.device) if (self.use_functions and self.num_fn_slots > 0) else None
        micro_noop_flag = None

        # track average primitive effect from this forward pass (for mining filters)
        effects_accum = None  # (num_prims,) tensor on device

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
            # add post-install function bias (encourage fresh functions)
            if self.use_functions and self._fn_post_bias is not None and self.num_fn_slots > 0:
                bias = self._fn_post_bias.view(1, -1).expand(B, -1)
                pad_prim = torch.zeros(B, len(self.prims), device=H.device)
                logits_action = logits_action + torch.cat([pad_prim, bias], dim=-1)

            # top-k candidate mask (keep best-k actions)
            if self.topk_actions is not None and self.topk_actions > 0 and self.topk_actions < logits_action.size(-1):
                k = int(self.topk_actions)
                topk_vals, topk_idx = torch.topk(logits_action, k=min(k, logits_action.size(-1)), dim=-1)
                mask_keep = torch.zeros_like(logits_action, dtype=torch.bool)
                mask_keep.scatter_(dim=-1, index=topk_idx, src=torch.ones_like(topk_idx, dtype=torch.bool))
                logits_action = torch.where(mask_keep, logits_action, logits_action.new_full(logits_action.shape, -1e9))

            probs_action = torch.softmax(logits_action, dim=-1)
            # sparsity penalty: encourage peaky distribution (L1-like)
            pen_sparse.append((probs_action.sum(dim=-1) - probs_action.max(dim=-1).values).mean())
            # small action entropy bonus (for trainer via aux)
            act_entropy = -(probs_action * (probs_action.clamp_min(1e-9).log())).sum(dim=-1).mean()
            ent_list.append(act_entropy)
            # Discrete operator selection per step (epsilon-greedy + gumbel)
            if self.training and self.epsilon > 0:
                # sample exploration mask per example
                explore = (torch.rand(B, device=H.device) < self.epsilon)
            else:
                explore = torch.zeros(B, dtype=torch.bool, device=H.device)
            pick_base = torch.zeros_like(logits_action)
            if self.training:
                # exploitation via Gumbel
                exp_pick = F.gumbel_softmax(logits_action, tau=self.temperature, hard=True, dim=-1)
                pick_base = exp_pick.clone()
                # exploration: uniform among currently unmasked actions (top-k mask already applied)
                if explore.any():
                    valid = (logits_action > -1e8).float()  # 1 where allowed
                    # normalize to uniform where valid
                    valid = valid / valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
                    # sample categorical by argmax over Gumbel(0,1) + log probs
                    g = -torch.log(-torch.log(torch.rand_like(valid) + 1e-9) + 1e-9)
                    u_pick = F.one_hot((torch.argmax((valid.clamp_min(1e-9).log() + g), dim=-1)), num_classes=valid.size(-1)).float()
                    pick_base[explore] = u_pick[explore]
            else:
                idx = logits_action.argmax(-1)
                pick_base = F.one_hot(idx, num_classes=logits_action.size(-1)).float()

            # ----- run all typed-state ops -----
            ts = TypedState(H=H, mask=mask_float, val=val, boolean=boolean, env=env)

            cand_masks, cand_vals, cand_bools, cand_envs = [], [], [], []
            # primitives
            prim_effects_step = []
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
                # primitive effect estimate (norm of delta state vs input ts)
                with torch.no_grad():
                    d_mask = (m_i - ts.mask).pow(2).mean(dim=1).sqrt().mean()
                    d_val = (v_i - ts.val).pow(2).mean().sqrt()
                    d_bool = (b_i - ts.boolean).pow(2).mean().sqrt()
                    try:
                        d_env = (e_i - (ts.env if ts.env is not None else e_i.new_zeros(e_i.shape))).pow(2).mean().sqrt()
                    except Exception:
                        d_env = torch.tensor(0.0, device=H.device)
                    prim_effects_step.append((d_mask + d_val + d_bool + 0.1 * d_env).detach())
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
            # accumulate primitive effects
            if len(prim_effects_step) > 0:
                step_vec = torch.stack(prim_effects_step, dim=0)  # (num_prims,)
                effects_accum = step_vec if effects_accum is None else (effects_accum + step_vec)

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
                            if max_depth_seen is not None:
                                max_depth_seen[bi] = max(max_depth_seen[bi], 1)
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
                            new_depth = frame.get("depth", 1) + 1
                            st.append({"sid": callee, "pc": 0, "depth": new_depth})
                            if max_depth_seen is not None:
                                max_depth_seen[bi] = max(max_depth_seen[bi], new_depth)
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

            # accumulate per-slot calls from base pick (functions only)
            if fn_call_hist is not None:
                with torch.no_grad():
                    sel_idx = pick.argmax(dim=-1)  # (B,)
                    for bi in range(B):
                        a = int(sel_idx[bi].item())
                        if a >= len(self.prims):
                            sid = a - len(self.prims)
                            if 0 <= sid < fn_call_hist.numel():
                                fn_call_hist[sid] += 1.0

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
                # usage ema update: estimate per-slot base probability mass (pre-micro)
                # Note: using probs over base logits to avoid micro interpreter zeroing function mass
                with torch.no_grad():
                    fn_mass = probs_action[:, len(self.prims) :].mean(dim=0)  # (num_fn,)
                    if self._slot_usage_ema is not None and fn_mass.numel() == self._slot_usage_ema.numel():
                        self._slot_usage_ema.mul_(0.95).add_(0.05 * fn_mass)
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
            if self.stop_bias != 0.0:
                stop_logit = stop_logit + self.stop_bias
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
        # function advantage bonus: when max function logit > max primitive logit
        try:
            last_logits = action_logits[:, -1, :]
            if self.use_functions and self.num_fn_slots > 0:
                max_prim = last_logits[:, : len(self.prims)].max(dim=-1).values
                max_fn = last_logits[:, len(self.prims) :].max(dim=-1).values
                fn_adv = (max_fn - max_prim).clamp_min(0.0).mean()
            else:
                fn_adv = torch.tensor(0.0, device=H.device)
        except Exception:
            fn_adv = torch.tensor(0.0, device=H.device)

        self._aux_loss = {
            "sparse": (torch.stack(pen_sparse).mean() if len(pen_sparse) > 0 else torch.tensor(0.0, device=H.device)),
            "halt_over": (torch.stack(pen_halt).mean() if len(pen_halt) > 0 else torch.tensor(0.0, device=H.device)),
            "act_entropy": (torch.stack(ent_list).mean() if len(ent_list) > 0 else torch.tensor(0.0, device=H.device)),
            "fn_adv": fn_adv,
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
                su = self._slot_usage_ema.detach().cpu()
                telem["slot_usage_ema"] = su
                try:
                    telem["slot_usage_mean"] = float(su.mean().item())
                    telem["slot_usage_max"] = float(su.max().item())
                except Exception:
                    pass
            # function call histogram this batch
            if fn_call_hist is not None:
                telem["fn_call_hist"] = fn_call_hist.detach().cpu().tolist()
            # function body length histogram (by steps count until return)
            try:
                lengths = []
                for sl in self.op_function.slots:
                    L = 0
                    for st in sl.steps:
                        if st.kind == "return":
                            break
                        L += 1
                    if L > 0:
                        lengths.append(L)
                hist = {}
                for L in lengths:
                    hist[L] = hist.get(L, 0) + 1
                telem["fn_body_len_hist"] = hist
            except Exception:
                pass
        if self.use_functions and self.exec_mode == "micro" and stacks is not None:
            depths = [len(s_) for s_ in stacks]
            telem["stack_depths"] = depths
            telem["max_stack_depth"] = max(depths) if len(depths) > 0 else 0
            # percent of episodes with depth >= 2 (max during run)
            try:
                if max_depth_seen is not None and len(max_depth_seen) > 0:
                    pct_d2 = sum(1 for v in max_depth_seen if v >= 2) / max(1, len(max_depth_seen))
                    telem["pct_depth_ge2"] = float(pct_d2)
            except Exception:
                pass
        # percent of episodes using at least one function (via traces)
        try:
            if traces is not None:
                used = 0
                for tr in traces:
                    if any((k == "func") for (k, _) in tr):
                        used += 1
                telem["pct_used_fn"] = float(used / max(1, len(traces)))
        except Exception:
            pass
        # delta logp advantage at final step (fn vs primitive-only)
        try:
            last_logits = action_logits[:, -1, :]
            logp = F.log_softmax(last_logits, dim=-1)
            lp_prim = logp[:, : len(self.prims)].max(dim=-1).values
            if self.num_fn_slots > 0:
                lp_fn = logp[:, len(self.prims) :].max(dim=-1).values
                telem["delta_logp_fn_vs_prim"] = float((lp_fn - lp_prim).mean().item())
        except Exception:
            pass
        # expose primitive effects (avg over steps) for mining filters
        if effects_accum is not None:
            with torch.no_grad():
                self._last_prim_effects = (effects_accum / float(max(1, self.max_steps)))
        else:
            self._last_prim_effects = None
        # decay post-install bias
        if self.use_functions and self._fn_post_bias is not None:
            self._fn_post_bias.mul_(self.fn_bias_decay)

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
        # track number of sleep cycles to allow a short warmup before pruning
        if not hasattr(self, "_sleep_calls"):
            self._sleep_calls = 0
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
        # Post-filter candidates by body length, effect, and diversity
        def _prim_seq_from_steps(steps: List[StepSpec]) -> List[int]:
            seq: List[int] = []
            for st in steps:
                if st.kind == "primitive" and st.idx is not None:
                    seq.append(int(st.idx))
                elif st.kind == "return":
                    break
                else:
                    # disallow higher-order in mined bodies for now
                    return []
            return seq

        prim_effects = getattr(self, "_last_prim_effects", None)
        # build existing library vectors for cosine/overlap
        existing_seqs = self._existing_primitive_patterns()
        num_prims = len(self.prims)
        def _vec_from_seq(seq: List[int]):
            v = torch.zeros(num_prims)
            for j in seq:
                if 0 <= j < num_prims:
                    v[j] += 1.0
            return v
        exist_vecs = [_vec_from_seq(list(p)) for p in existing_seqs]

        filtered: List[FnCandidate] = []
        for cand in cands:
            seq = _prim_seq_from_steps(cand.steps)
            if len(seq) < self.mine_min_body_len:
                continue
            # effect threshold: sum recent primitive effects along sequence
            if isinstance(prim_effects, torch.Tensor):
                eff = float(sum([float(prim_effects[min(j, prim_effects.numel()-1)].item()) for j in seq]))
                if eff < self.mine_min_effect:
                    continue
            # diversity filters
            if exist_vecs:
                v = _vec_from_seq(seq)
                keep = True
                for ev, es in zip(exist_vecs, existing_seqs):
                    # cosine similarity
                    cos = 0.0
                    if v.norm().item() > 0 and ev.norm().item() > 0:
                        cos = float(torch.dot(v, ev) / (v.norm() * ev.norm() + 1e-9))
                    if cos > self.mine_cosine_max:
                        keep = False
                        break
                    # coverage overlap (Jaccard over sets of prim indices)
                    s1, s2 = set(seq), set(es)
                    jacc = len(s1 & s2) / max(1, len(s1 | s2))
                    if jacc > self.mine_overlap_max:
                        keep = False
                        break
                if not keep:
                    continue
            filtered.append(cand)

        installed = self.op_function.install(filtered)
        # Increment sleep counter and (optionally) prune based on usage ema
        self._sleep_calls += 1
        if self._slot_usage_ema is not None and self._sleep_calls >= 2:
            usage = self._slot_usage_ema.detach().cpu().tolist()
            try:
                self.op_function.prune(usage, self.prune_threshold)
            except Exception:
                pass
        # Boost newly installed functions via logit bias (decaying)
        if self.use_functions and self._fn_post_bias is not None and len(installed) > 0:
            for sid in installed:
                if 0 <= sid < self._fn_post_bias.numel():
                    self._fn_post_bias[sid] = max(float(self._fn_post_bias[sid].item()), self.fn_bias_init)
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
