import torch
import torch.nn as nn

from concept_learner.reasoning_ops import (
    TypedState,
    OpLet,
    OpLoad,
    OpAddConst,
    OpFunction,
    StepSpec,
    FnCandidate,
)


torch.manual_seed(0)


def make_state(B=2, T=5, d=8, val0=0.0, bool0=0.0):
    H = torch.randn(B, T, d)
    mask = torch.ones(B, T)
    st = TypedState.from_H_mask(H, mask)
    st.val = torch.full((B, 1), float(val0))
    st.boolean = torch.full((B, 1), float(bool0))
    return st


def test_op_let_writes_env_and_load_reads_back():
    B, T, d = 3, 6, 16
    st = make_state(B, T, d, val0=2.0)
    z = torch.zeros(B, d)

    let = OpLet(d_model=d, num_regs=4)
    # Forward should not change visible streams
    m1, v1, b1 = let(st, z)
    assert torch.equal(m1, st.mask) and torch.equal(v1, st.val) and torch.equal(b1, st.boolean)

    # env_update should create/update env: starts from zeros, alpha≈sigmoid(0)=0.5, uniform addr
    e_new = let.env_update(st, z)
    assert e_new.shape == (B, 4)
    # Sum over registers equals alpha * val (uniform w sums to 1)
    row_sums = e_new.sum(dim=-1, keepdim=True)
    assert torch.allclose(row_sums, torch.sigmoid(torch.tensor(0.0)) * st.val, atol=1e-5)

    # Now load should read back average of env and blend with current val
    load = OpLoad(d_model=d, num_regs=4)
    st2 = make_state(B, T, d, val0=1.0)
    st2.env = e_new
    m2, v2, b2 = load(st2, z)
    # alpha=beta=softplus(0)=ln(2)
    alpha = torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-6
    beta = torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-6
    # r is mean across K registers; here sum was alpha0*val0, distributed uniformly over 4 -> mean = sum/4
    r = (row_sums / 4.0)
    v_expected = beta * st2.val + alpha * r
    assert torch.allclose(v2, v_expected, atol=1e-5)
    # mask and boolean pass-through
    assert torch.equal(m2, st2.mask) and torch.equal(b2, st2.boolean)


def test_op_function_executes_primitive_sequence_and_cond():
    B, T, d = 2, 4, 8
    # base ops: +1 and +10
    ops = [OpAddConst(1), OpAddConst(10)]
    fn = OpFunction(d_model=d, base_ops=ops, num_slots=4, max_body_len=4)
    fn.eval()  # use flat execution path for determinism

    # build slot 0: +1 ; +10 ; return
    steps_seq = [
        StepSpec(kind="primitive", idx=0),
        StepSpec(kind="primitive", idx=1),
        StepSpec(kind="return"),
    ]
    sid = fn.install([FnCandidate(steps=steps_seq)])[0]
    assert sid == 0

    H = torch.randn(B, T, d)
    mask = torch.ones(B, T)
    st = TypedState.from_H_mask(H, mask)
    st.val = torch.zeros(B, 1)
    z = torch.zeros(B, d)

    m, v, b = fn.run_slot(0, st, z)
    # Expected value ≈ softplus(0)*(1+10)
    alpha = torch.nn.functional.softplus(torch.tensor(0.0)) + 1e-6
    v_expected = alpha * (1.0 + 10.0)
    assert torch.allclose(v, torch.full_like(v, v_expected), atol=1e-5)
    assert torch.equal(m, st.mask) and torch.equal(b, st.boolean)

    # Build THEN and ELSE slots
    sid_then = fn.install([FnCandidate(steps=[StepSpec(kind="primitive", idx=0), StepSpec(kind="return")])])[0]
    # Slot 3: cond(then=sid_then, else is empty -> defaults to no-op), return
    sid_cond = fn.install([FnCandidate(steps=[StepSpec(kind="cond", idx=sid_then), StepSpec(kind="return")])])[0]

    # Prepare boolean guards: first example True -> then, second False -> else
    st_cond = TypedState.from_H_mask(H, mask)
    st_cond.val = torch.zeros(B, 1)
    st_cond.boolean = torch.tensor([[1.0], [0.0]])
    m3, v3, b3 = fn.run_slot(sid_cond, st_cond, z)
    v_then = alpha * 1.0
    assert torch.allclose(v3[0], torch.tensor([v_then])), "then-branch value incorrect"
    # else branch empty -> leaves value unchanged (0.0)
    assert torch.allclose(v3[1], torch.tensor([0.0])), "else-branch (empty) should keep value"
    assert torch.equal(m3, st_cond.mask) and torch.equal(b3, st_cond.boolean)
