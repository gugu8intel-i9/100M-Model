#!/usr/bin/env python3
"""Train a ~100M causal LM on a capped 2B-token Chinchilla budget.

Stages:
  tokenizer  Train a 32k BPE tokenizer from streamed approved source samples.
  pretrain   Train the base model on the 2B-token weighted mixture.
  sft        Continue an existing base checkpoint on answer/rationale supervision.

The script streams datasets rather than downloading FineWeb-Edu. It refuses to
start until the dataset governance gate in the YAML is explicitly accepted.
"""
import argparse, json, math, os, random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast
import gigatoken as gt


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("data_governance", {}).get("accept_dataset_terms", False):
        raise RuntimeError("Refusing to ingest data: review dataset licenses/provenance, then set data_governance.accept_dataset_terms: true.")
    return cfg


def dist_info():
    return (int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)))


def render_pretrain(row: Dict[str, Any], fields: List[str]) -> str:
    return "\n".join(str(row.get(f, "")).strip() for f in fields if row.get(f)).strip()


def render_sft(row: Dict[str, Any], fmt: str) -> Optional[tuple[str, str]]:
    """Return prompt and supervised completion. We supervise responses, not system prompts."""
    if fmt == "math_sft":
        q, r, a = row.get("question", ""), row.get("reasoning", ""), row.get("answer", "")
        return (f"User: {q}\nAssistant:", f" {r}\nFinal answer: {a}") if q and a else None
    if fmt == "english_sft":
        instruction, inp, out = row.get("instruction", ""), row.get("input_text", ""), row.get("output_text", "")
        return (f"User: {instruction}\n{inp}\nAssistant:", f" {out}") if out else None
    if fmt == "messages_sft":
        try:
            messages = json.loads(row["messages_json"])
        except (KeyError, TypeError, json.JSONDecodeError):
            return None
        # Keep the final assistant turn only; skip tool-bearing/multiturn traces.
        if row.get("n_tool_calls", 0) or not messages or str(messages[-1].get("role", "")).lower() != "assistant":
            return None
        prompt_parts = []
        for m in messages[:-1]:
            role = str(m.get("role", "user")).lower()
            if role in {"user", "system"}:
                prompt_parts.append(f"{role.title()}: {m.get('content', '')}")
        answer = messages[-1].get("content", "")
        if not prompt_parts or not answer:
            return None
        # Avoid training an always-visible <think> convention. Keep explanatory text if present,
        # but strip wrapper tags so inference can return concise answers by default.
        answer = answer.replace("<think>", "").replace("</think>", "").strip()
        return ("\n".join(prompt_parts) + "\nAssistant:", " " + answer)
    raise ValueError(fmt)


class GigaTokenizer:
    """Gigatoken-backed adapter for the locally trained 32k BPE vocabulary.

    Gigatoken compatibility mode preserves token IDs from the Hugging Face BPE
    tokenizer, so the vocabulary size and ~100M parameter accounting remain
    unchanged while all training-data encoding goes through Gigatoken.
    """
    def __init__(self, path: str):
        self.fast = PreTrainedTokenizerFast.from_pretrained(path)
        self.backend = gt.Tokenizer(self.fast._tokenizer).as_hf()
        self.eos_token_id = self.fast.eos_token_id
        self.pad_token_id = self.fast.pad_token_id

    def ids(self, text: str) -> List[int]:
        return self.backend.encode(text).ids


class PackedMixture(torch.utils.data.IterableDataset):
    def __init__(self, sources, tokenizer, seq_len, stage, seed):
        self.sources, self.tok, self.seq_len, self.stage, self.seed = sources, tokenizer, seq_len, stage, seed

    def _stream(self, source):
        kwargs = {"split": source.get("split", "train"), "streaming": True}
        if source.get("subset"):
            return load_dataset(source["dataset"], source["subset"], **kwargs)
        return load_dataset(source["dataset"], **kwargs)

    def __iter__(self):
        rank, world = dist_info(); rng = random.Random(self.seed + rank)
        sources = self.sources
        streams = [iter(self._stream(s).shuffle(seed=self.seed + i + rank, buffer_size=10_000)) for i, s in enumerate(sources)]
        weights = [s["weight"] for s in sources]
        emitted = [0] * len(sources)
        while True:
            viable = [i for i, s in enumerate(sources) if emitted[i] < int(s.get("max_tokens", 10**30))]
            if not viable: return
            probs = [weights[i] if i in viable else 0 for i in range(len(sources))]
            idx = rng.choices(range(len(sources)), weights=probs, k=1)[0]
            try: row = next(streams[idx])
            except StopIteration:
                streams[idx] = iter(self._stream(sources[idx])); row = next(streams[idx])
            s = sources[idx]
            if self.stage == "pretrain":
                text = render_pretrain(row, s["text_fields"])
                ids = self.tok.ids(text)
                for offset in range(0, max(0, len(ids) - 1), self.seq_len):
                    chunk = ids[offset:offset + self.seq_len + 1]
                    if len(chunk) == self.seq_len + 1:
                        emitted[idx] += self.seq_len
                        yield {"input_ids": torch.tensor(chunk[:-1]), "labels": torch.tensor(chunk[1:])}
            else:
                if s.get("require_verifier_passed") and not row.get("verifier_passed", False): continue
                rendered = render_sft(row, s["format"])
                if not rendered: continue
                prompt, answer = rendered
                p = self.tok.ids(prompt)
                a = self.tok.ids(answer) + [self.tok.eos_token_id]
                # Causal LM alignment: input[t] predicts target[t] == full[t + 1].
                # The last prompt token predicts the first assistant token; prompt-only
                # positions are masked so the instruction is never optimized as a target.
                full = (p + a)[:self.seq_len + 1]
                ids = full[:-1]
                labels = ([-100] * max(0, len(p) - 1) + a)[:len(ids)]
                if a and len(ids) >= len(p):
                    emitted[idx] += len(ids)
                    pad = self.seq_len - len(ids)
                    yield {"input_ids": torch.tensor(ids + [self.tok.pad_token_id] * pad), "labels": torch.tensor(labels + [-100] * pad)}


class RMSNorm(nn.Module):
    def __init__(self, d): super().__init__(); self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x): return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype) * self.weight


def rotate_half(x):
    x = x.view(*x.shape[:-1], -1, 2); return torch.stack((-x[..., 1], x[..., 0]), -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__(); self.h, self.d = c["n_heads"], c["d_model"] // c["n_heads"]
        self.qkv = nn.Linear(c["d_model"], 3*c["d_model"], bias=False); self.out = nn.Linear(c["d_model"], c["d_model"], bias=False)
        inv = 1.0 / (c.get("rope_theta", 10000.0) ** (torch.arange(0, self.d, 2).float() / self.d)); self.register_buffer("inv", inv)
    def forward(self, x):
        b,t,c = x.shape; q,k,v = self.qkv(x).chunk(3,-1)
        q,k,v = [z.view(b,t,self.h,self.d).transpose(1,2) for z in (q,k,v)]
        freqs = torch.outer(torch.arange(t, device=x.device).float(), self.inv).repeat_interleave(2,-1)
        cos,sin = freqs.cos()[None,None],freqs.sin()[None,None]
        q,k = q*cos + rotate_half(q)*sin, k*cos + rotate_half(k)*sin
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,c)
        return self.out(y)


class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); d,f=c["d_model"],c["intermediate_size"]
        self.n1,self.a,self.n2=RMSNorm(d),Attention(c),RMSNorm(d)
        self.gate,self.up,self.down=nn.Linear(d,f,bias=False),nn.Linear(d,f,bias=False),nn.Linear(f,d,bias=False)
    def forward(self,x):
        x=x+self.a(self.n1(x)); return x+self.down(F.silu(self.gate(self.n2(x)))*self.up(self.n2(x)))


class LM(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c; d,v=c["d_model"],c["vocab_size"]
        self.embed=nn.Embedding(v,d); self.blocks=nn.ModuleList([Block(c) for _ in range(c["n_layers"])]); self.norm=RMSNorm(d); self.head=nn.Linear(d,v,bias=False)
        self.head.weight=self.embed.weight
        self.apply(self.init)
    @staticmethod
    def init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
    def forward(self, ids, labels=None):
        x=self.embed(ids)
        for b in self.blocks: x=b(x)
        logits=self.head(self.norm(x)); loss=None
        if labels is not None: loss=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100)
        return logits,loss


def count_parameters(model): return sum(p.numel() for p in model.parameters())

def tokenizer_stage(cfg):
    tc=cfg["tokenizer"]; out=Path(tc["path"]); out.mkdir(parents=True,exist_ok=True)
    def texts():
        for source in cfg["pretrain_mixture"]:
            ds=load_dataset(source["dataset"], source.get("subset"), split=source.get("split","train"), streaming=True)
            for row in ds.take(50_000):
                t=render_pretrain(row,source["text_fields"])
                if t: yield t
    tok=Tokenizer(models.BPE(unk_token="<unk>")); tok.pre_tokenizer=pre_tokenizers.ByteLevel(); tok.decoder=decoders.ByteLevel()
    tok.train_from_iterator(texts(),trainer=trainers.BpeTrainer(vocab_size=tc["vocab_size"],special_tokens=["<pad>","<bos>","<eos>","<unk>"]))
    tok.save(str(out/"tokenizer.json")); PreTrainedTokenizerFast(tokenizer_file=str(out/"tokenizer.json"),pad_token="<pad>",bos_token="<bos>",eos_token="<eos>",unk_token="<unk>").save_pretrained(out)
    print(f"Tokenizer saved to {out}")

def train(cfg, stage):
    tcfg=cfg["training"]; scfg=cfg.get("sft",{}) if stage=="sft" else tcfg
    tok=GigaTokenizer(cfg["tokenizer"]["path"])
    model=LM(cfg["model"]).cuda() if torch.cuda.is_available() else LM(cfg["model"])
    device=next(model.parameters()).device; print(f"parameters: {count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M), device={device}")
    if cfg.get("resume_from"):
        model.load_state_dict(torch.load(cfg["resume_from"],map_location=device)["model"])
    sources=cfg["sft"]["mixture"] if stage=="sft" else cfg["pretrain_mixture"]
    loader=torch.utils.data.DataLoader(PackedMixture(sources,tok,tcfg["seq_len"],stage,cfg["seed"]),batch_size=tcfg["micro_batch_size"],num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=scfg.get("learning_rate",tcfg["learning_rate"]),betas=(tcfg["beta1"],tcfg["beta2"]),weight_decay=tcfg["weight_decay"])
    target,seen=int(scfg["target_tokens"]),0; warm=int(scfg.get("warmup_tokens",tcfg["warmup_tokens"])); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda" and not tcfg["bf16"])
    out=Path(cfg["output_dir"]); out.mkdir(parents=True,exist_ok=True); model.train(); opt.zero_grad(set_to_none=True)
    for step,batch in enumerate(loader,1):
        ids,labels=batch["input_ids"].to(device),batch["labels"].to(device); seen += int((labels!=-100).sum())
        frac=min(1, seen / target)
        base_lr = scfg.get("learning_rate", tcfg["learning_rate"])
        if seen < warm:
            lr = base_lr * min(1, seen / max(1, warm))
        else:
            cosine = 0.5 * (1 + math.cos(math.pi * frac))
            lr = base_lr * (tcfg["min_lr_ratio"] + (1 - tcfg["min_lr_ratio"]) * cosine)
        for g in opt.param_groups:g["lr"]=lr
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda" and tcfg["bf16"]): _,loss=model(ids,labels); loss=loss/tcfg["grad_accumulation_steps"]
        scaler.scale(loss).backward()
        if step%tcfg["grad_accumulation_steps"]==0:
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),tcfg["grad_clip_norm"]); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if step%tcfg["log_every"]==0: print(json.dumps({"micro_step":step,"tokens":seen,"loss":round(loss.item()*tcfg["grad_accumulation_steps"],4),"lr":lr}))
        if step%tcfg["save_every"]==0: torch.save({"model":model.state_dict(),"config":cfg,"tokens":seen},out/f"{stage}-{seen}.pt")
        if seen>=target: break
    torch.save({"model":model.state_dict(),"config":cfg,"tokens":seen},out/f"{stage}-final.pt")

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/chinchilla_cot_100m.yaml"); p.add_argument("--stage",choices=["tokenizer","pretrain","sft"],required=True); a=p.parse_args(); cfg=load_cfg(a.config); tokenizer_stage(cfg) if a.stage=="tokenizer" else train(cfg,a.stage)
