import math, torch, torch.nn as nn
import submission_adapter as sa


class Tok:
    eos_token_id = 1
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 20) for c in text]}


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(32, 16); self.o = nn.Linear(16, 32)
    def forward(self, input_ids, attention_mask=None):
        class O: pass
        o = O(); o.logits = self.o(self.e(input_ids)); return o


def test_masking_and_eos():
    b = sa.build_sft_batch(Tok(), ["ab", "abcd"], ["xy", "z"], 16)
    assert b["labels"][0, :2].tolist() == [-100, -100]
    assert b["labels"][0, 4].item() == 1          # EOS is supervised
    assert (b["labels"][b["attention_mask"] == 0] == -100).all()


def test_truncation_drops_eos():
    b = sa.build_sft_batch(Tok(), ["ab"], ["xyz"], 4)
    assert b["input_ids"].shape[1] == 4 and 1 not in b["input_ids"][0].tolist()


def test_logprobs_match_manual_and_order():
    torch.manual_seed(0); m, t = LM(), Tok()
    P, R = ["ab", "abcd"], ["xy", "z"]
    lp = sa.compute_response_logprobs(m, t, P, R, 16)
    one = sa.compute_response_logprobs(m, t, [P[1]], [R[1]], 16)
    assert torch.allclose(lp[1], one[0], atol=1e-5)   # row order + padding invariance
    ids = t(P[0])["input_ids"] + t(R[0])["input_ids"] + [1]
    lg = torch.log_softmax(m(torch.tensor([ids])).logits[0], -1)
    man = sum(lg[i - 1, ids[i]] for i in range(2, len(ids)))
    assert torch.allclose(lp[0], man, atol=1e-5)


def test_dpo_math():
    z = torch.zeros(3)
    assert abs(sa.compute_dpo_loss_from_logps(z, z, z, z, 0.1).item() - math.log(2)) < 1e-6
    big = sa.compute_dpo_loss_from_logps(z + 50, z, z, z, 0.1)
    assert big.item() < 0.01


def test_dpo_end_to_end_and_grad():
    torch.manual_seed(0); pol, ref, t = LM(), LM(), Tok()
    ref.load_state_dict(pol.state_dict())
    for p in ref.parameters(): p.requires_grad_(False)
    loss = sa.compute_dpo_loss(pol, ref, t, ["ab", "cd"], ["xy", "zz"], ["q", "rrr"], 0.1, 16)
    assert abs(loss.item() - math.log(2)) < 1e-5     # identical models -> log 2
    loss.backward()
    assert pol.o.weight.grad is not None and ref.o.weight.grad is None
    opt = torch.optim.SGD(pol.parameters(), lr=0.5)
    for _ in range(30):
        opt.zero_grad()
        l = sa.compute_dpo_loss(pol, ref, t, ["ab", "cd"], ["xy", "zz"], ["q", "rrr"], 0.1, 16)
        l.backward(); opt.step()
    assert l.item() < math.log(2)
