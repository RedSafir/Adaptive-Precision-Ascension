#!/usr/bin/env python3
"""Forensic Logging Demo — APA (Adaptive Precision Ascension)

Demonstrates how to enable forensic logging and read the resulting log.

What this script does:
1. Builds a small 3-layer model using APALinear
2. Enables forensic logging (enable_forensic_logging=True)
3. Trains for a short burst — injecting artificial overflow in the first
   forward pass to ensure escalation events fire quickly
4. Prints the resulting forensic log file in a human-readable format
5. Shows how to load the log programmatically for analysis

Run:
    python examples/forensic_demo.py

Expected output: several forensic JSON snapshots printed to stdout and
written to `forensic_demo_run_forensic.jsonl`.

PERFORMANCE NOTE
----------------
This demo runs with forensic mode enabled — training throughput will be
significantly lower than a normal APA run.  The per-tensor CPU-GPU sync
(``.item()``) is the cause.  This is intentional: forensic mode trades
speed for analysis completeness.  Never use it in production benchmarks.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from apa import APAConfig, APALinear, APAManager


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SmallForensicNet(nn.Module):
    """Minimal 3-layer network for forensic demo."""

    def __init__(self, config: APAConfig):
        super().__init__()
        self.fc1 = APALinear(32, 64, config=config)
        self.fc2 = APALinear(64, 64, config=config)
        self.fc3 = APALinear(64, 10, config=config)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.norm(x)
        x = F.gelu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_demo():
    device = 'cpu'  # demo runs on CPU for portability
    log_file = 'forensic_demo_run.jsonl'

    print("=" * 65)
    print("  APA Forensic Logging Demo")
    print("=" * 65)
    print(f"  Forensic log   : will auto-derive from '{log_file}'")
    print(f"  Device         : {device}")
    print()

    config = APAConfig(
        fp8_simulation_mode=True,   # runs on any GPU/CPU without native FP8
        device=device,
        check_interval=1,           # evaluate every step to catch overflow fast
        log_file=log_file,
        # --- Forensic options ---
        enable_forensic_logging=True,
        # forensic_log_file auto-derived → forensic_demo_run_forensic.jsonl
        forensic_capture_tensor_stats=True,    # include mean/std per role
        forensic_capture_argmax_index=False,   # keep False unless deep debugging
    )

    print(f"  Forensic output: {config.forensic_log_file}")
    print()

    model = SmallForensicNet(config).to(device)
    manager = APAManager(model, config)
    optimizer = torch.optim.AdamW(manager.get_trainable_parameters(), lr=1e-3)

    print(f"  APALinear modules discovered: {list(manager.apa_modules.keys())}")
    print(f"  Forensic forward hooks      : {len(manager._forward_hook_handles)}")
    print()

    # -----------------------------------------------------------------------
    # Training loop — 20 steps, first step injects an artificial overflow
    # -----------------------------------------------------------------------
    print("  Training (20 steps, forensic mode active)...")
    print("  " + "-" * 61)

    for step in range(20):
        manager.pre_step()
        optimizer.zero_grad(set_to_none=True)

        x = torch.randn(8, 32, device=device)
        labels = torch.randint(0, 10, (8,), device=device)

        # Inject artificial overflow in step 0 so escalation fires immediately
        # — in real training you would never do this; it's just for demo speed.
        if step == 0:
            with torch.no_grad():
                for m in manager.apa_modules.values():
                    m.gpu_amax.fill_(500.0)   # force FP8 → FP16 escalation

        output = model(x)
        loss = F.cross_entropy(output, labels)
        loss.backward()

        should_step = manager.post_backward_sync_and_eval()
        if should_step:
            optimizer.step()

        if step % 5 == 0 or step < 2:
            module_levels = {
                name: ['FP8', 'FP16', 'TF32'][m.level]
                for name, m in manager.apa_modules.items()
            }
            print(f"  Step {step:2d} | loss={loss.item():.4f} | levels={module_levels}")

    print("  " + "-" * 61)
    print()

    # -----------------------------------------------------------------------
    # Read and display the forensic log
    # -----------------------------------------------------------------------
    forensic_path = config.forensic_log_file
    if not os.path.exists(forensic_path):
        print("  No escalation events occurred — forensic log not created.")
        return

    with open(forensic_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"  Forensic log: '{forensic_path}'")
    print(f"  Total escalation events recorded: {len(records)}")
    print()

    for i, rec in enumerate(records):
        print(f"  ── Event #{i+1} ─────────────────────────────────────────")
        print(f"     Step             : {rec['step']}")
        print(f"     Module           : {rec['module_name']}")
        print(f"     Reason           : {rec['reason']}")
        print(f"     Level            : {rec['level_before']} → {rec['level_after']}")
        print(f"     Culprit role     : {rec['culprit_tensor_role']}")
        print(f"     amax at event    : {rec['amax_value']:.4f}")
        print(f"     Threshold        : {rec['threshold_at_time']:.4f}")
        print(f"     Tensor shape     : {rec['tensor_shape']}")
        print(f"     Dtype at time    : {rec['dtype_at_time']}")
        print(f"     Preceding module : {rec['preceding_module_in_forward_order']}")
        print(f"     Per-role amax    :")
        for role, val in (rec['per_role_amax'] or {}).items():
            v_str = f"{val:.6f}" if val is not None else "N/A (not tracked this step)"
            print(f"       {role:<22s} = {v_str}")
        if rec.get('per_role_stats'):
            print(f"     Per-role stats   :")
            for role, stats in rec['per_role_stats'].items():
                if stats is not None:
                    print(f"       {role:<22s} mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        print(f"     Timestamp        : {rec['timestamp_utc']}")
        print()

    # -----------------------------------------------------------------------
    # Programmatic usage example
    # -----------------------------------------------------------------------
    print("  ── Programmatic access example ──────────────────────────")
    print("  Load the forensic log with:")
    print()
    print("    import json")
    print(f"    with open('{forensic_path}') as f:")
    print("        records = [json.loads(l) for l in f if l.strip()]")
    print()
    print("    # Find all OVERFLOW events")
    print("    overflows = [r for r in records if r['reason'] == 'OVERFLOW']")
    print()
    print("    # Find which role caused most escalations")
    print("    from collections import Counter")
    print("    Counter(r['culprit_tensor_role'] for r in records).most_common()")
    print()
    print("=" * 65)


if __name__ == '__main__':
    run_demo()
