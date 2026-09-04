#!/usr/bin/env python3
"""
CLI Comparative Analyzer for APA vs Pure FP32 Baseline
Generates paper-ready figures (.png) and summary comparison (.csv) without needing Jupyter.

Usage:
    python scripts/plot_comparative_results.py \
        --apa result/vit_apa_100ep_comp.jsonl \
        --fp32 result/vit_fp32_100ep_comp.jsonl \
        --output_dir result/plots/
"""

import os
import sys
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Plot comparative results between APA and FP32 baseline.")
    parser.add_argument('--apa', type=str, required=True, help="Path to APA JSONL log file")
    parser.add_argument('--fp32', type=str, required=True, help="Path to FP32 baseline JSON/JSONL log file")
    parser.add_argument('--output_dir', type=str, default='result/plots', help="Directory to save generated plots and summary CSV")
    return parser.parse_args()

def parse_log(filepath):
    if not filepath or not os.path.exists(filepath):
        print(f"[WARN] File not found: {filepath}")
        return pd.DataFrame(), []
    
    epochs = []
    escalations = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            if 'epoch' in d and ('train_loss' in d or 'test_loss' in d):
                rec = dict(d)
                if 'train_acc' in rec:
                    rec['train_acc_pct'] = rec['train_acc'] * 100 if rec['train_acc'] <= 1.0 else rec['train_acc']
                if 'test_acc' in rec:
                    rec['test_acc_pct'] = rec['test_acc'] * 100 if rec['test_acc'] <= 1.0 else rec['test_acc']
                epochs.append(rec)
            elif d.get('event') == 'escalation':
                escalations.append(d)
                
    df = pd.DataFrame(epochs).drop_duplicates(subset=['epoch']).sort_values('epoch') if epochs else pd.DataFrame()
    return df, escalations

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Ingesting APA log : {args.apa}")
    df_apa, esc_apa = parse_log(args.apa)
    print(f"  -> Found {len(df_apa)} epochs, {len(esc_apa)} escalation events.")
    
    print(f"Ingesting FP32 log: {args.fp32}")
    df_fp32, _ = parse_log(args.fp32)
    print(f"  -> Found {len(df_fp32)} epochs.")
    
    if df_apa.empty and df_fp32.empty:
        print("[ERROR] No valid epoch data found in either file. Exiting.")
        sys.exit(1)
        
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Loss Convergence Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if not df_apa.empty and 'train_loss' in df_apa.columns:
        ax1.plot(df_apa['epoch'], df_apa['train_loss'], label='APA (Adaptive Precision)', color='#2563EB', linewidth=2.2, marker='o', markersize=3)
    if not df_fp32.empty and 'train_loss' in df_fp32.columns:
        ax1.plot(df_fp32['epoch'], df_fp32['train_loss'], label='Pure FP32 Baseline', color='#64748B', linewidth=2.2, linestyle='--', marker='s', markersize=3)
    ax1.set_title('Training Loss Convergence', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)
    
    if not df_apa.empty and 'test_loss' in df_apa.columns:
        ax2.plot(df_apa['epoch'], df_apa['test_loss'], label='APA (Adaptive Precision)', color='#2563EB', linewidth=2.2, marker='o', markersize=3)
    if not df_fp32.empty and 'test_loss' in df_fp32.columns:
        ax2.plot(df_fp32['epoch'], df_fp32['test_loss'], label='Pure FP32 Baseline', color='#64748B', linewidth=2.2, linestyle='--', marker='s', markersize=3)
    ax2.set_title('Test / Validation Loss Convergence', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(args.output_dir, 'loss_convergence.png')
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"Saved: {loss_path}")
    
    # 2. Accuracy Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if not df_apa.empty and 'train_acc_pct' in df_apa.columns:
        ax1.plot(df_apa['epoch'], df_apa['train_acc_pct'], label='APA (Adaptive Precision)', color='#2563EB', linewidth=2.2, marker='o', markersize=3)
    if not df_fp32.empty and 'train_acc_pct' in df_fp32.columns:
        ax1.plot(df_fp32['epoch'], df_fp32['train_acc_pct'], label='Pure FP32 Baseline', color='#64748B', linewidth=2.2, linestyle='--', marker='s', markersize=3)
    ax1.set_title('Training Accuracy Trajectory', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, alpha=0.3)
    
    if not df_apa.empty and 'test_acc_pct' in df_apa.columns:
        ax2.plot(df_apa['epoch'], df_apa['test_acc_pct'], label='APA (Adaptive Precision)', color='#10B981', linewidth=2.2, marker='o', markersize=3)
        best_apa = df_apa['test_acc_pct'].max()
        best_apa_ep = df_apa.loc[df_apa['test_acc_pct'].idxmax(), 'epoch']
        ax2.scatter([best_apa_ep], [best_apa], color='#10B981', s=100, zorder=6)
        ax2.annotate(f'Peak APA: {best_apa:.2f}% (Ep {best_apa_ep})', xy=(best_apa_ep, best_apa),
                     xytext=(best_apa_ep, best_apa + 1.5), fontweight='bold', color='#047857',
                     arrowprops=dict(arrowstyle='->', color='#047857', lw=1.5))
                     
    if not df_fp32.empty and 'test_acc_pct' in df_fp32.columns:
        ax2.plot(df_fp32['epoch'], df_fp32['test_acc_pct'], label='Pure FP32 Baseline', color='#64748B', linewidth=2.2, linestyle='--', marker='s', markersize=3)
        best_fp32 = df_fp32['test_acc_pct'].max()
        best_fp32_ep = df_fp32.loc[df_fp32['test_acc_pct'].idxmax(), 'epoch']
        ax2.scatter([best_fp32_ep], [best_fp32], color='#64748B', s=100, zorder=6)
        ax2.annotate(f'Peak FP32: {best_fp32:.2f}% (Ep {best_fp32_ep})', xy=(best_fp32_ep, best_fp32),
                     xytext=(best_fp32_ep, best_fp32 - 3.5), fontweight='bold', color='#334155',
                     arrowprops=dict(arrowstyle='->', color='#334155', lw=1.5))
                     
    ax2.set_title('Test / Validation Accuracy Comparison', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.legend(loc='lower right', frameon=True)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(args.output_dir, 'accuracy_comparison.png')
    plt.savefig(acc_path, dpi=300)
    plt.close()
    print(f"Saved: {acc_path}")
    
    # 3. Training Time & Wall-Time
    if 'epoch_time_sec' in df_apa.columns and 'epoch_time_sec' in df_fp32.columns and not df_apa.empty and not df_fp32.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(df_apa['epoch'], df_apa['epoch_time_sec'], label='APA Epoch Duration', color='#2563EB', linewidth=2.2, marker='o', markersize=3)
        ax1.plot(df_fp32['epoch'], df_fp32['epoch_time_sec'], label='FP32 Baseline Epoch Duration', color='#DC2626', linewidth=2.2, linestyle='--', marker='s', markersize=3)
        ax1.set_title('Computation Time per Epoch', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Time (s)', fontweight='bold')
        ax1.legend(frameon=True)
        ax1.grid(True, alpha=0.3)
        
        apa_cum = df_apa['epoch_time_sec'].cumsum() / 60
        fp32_cum = df_fp32['epoch_time_sec'].cumsum() / 60
        ax2.plot(df_apa['epoch'], apa_cum, label='APA Cumulative Wall-Time', color='#2563EB', linewidth=2.2)
        ax2.plot(df_fp32['epoch'], fp32_cum, label='FP32 Cumulative Wall-Time', color='#DC2626', linewidth=2.2, linestyle='--')
        min_len = min(len(apa_cum), len(fp32_cum))
        ax2.fill_between(df_apa['epoch'][:min_len], apa_cum[:min_len], fp32_cum[:min_len], color='#10B981', alpha=0.25, label='Time Saved')
        ax2.set_title('Cumulative Training Wall-Time', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Total Time (Minutes)', fontweight='bold')
        ax2.legend(frameon=True)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        time_path = os.path.join(args.output_dir, 'training_time.png')
        plt.savefig(time_path, dpi=300)
        plt.close()
        print(f"Saved: {time_path}")
        
    # 4. Precision Drift (if available in APA)
    if not df_apa.empty and 'precision_distribution' in df_apa.columns:
        drift_list = []
        for _, r in df_apa.iterrows():
            p = r['precision_distribution']
            if isinstance(p, dict):
                tot = p.get('fp8', 0) + p.get('fp16', 0) + p.get('tf32', 0)
                if tot > 0:
                    drift_list.append({
                        'epoch': r['epoch'],
                        'FP8': (p.get('fp8', 0) / tot) * 100,
                        'FP16': (p.get('fp16', 0) / tot) * 100,
                        'TF32': (p.get('tf32', 0) / tot) * 100
                    })
        if drift_list:
            df_drift = pd.DataFrame(drift_list)
            plt.figure(figsize=(12, 4.5))
            plt.stackplot(df_drift['epoch'], df_drift['FP8'], df_drift['FP16'], df_drift['TF32'],
                          labels=['FP8 (Max Throughput)', 'FP16 (Medium Range)', 'TF32 (Full Precision)'],
                          colors=['#10B981', '#3B82F6', '#EF4444'], alpha=0.85)
            plt.xlabel('Epoch', fontweight='bold')
            plt.ylabel('Network Precision Composition (%)', fontweight='bold')
            plt.ylim(0, 100)
            plt.title('APA Precision Drift Evolution Across 100 Epochs', fontweight='bold', fontsize=12)
            plt.legend(loc='upper right', frameon=True)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            drift_path = os.path.join(args.output_dir, 'precision_drift.png')
            plt.savefig(drift_path, dpi=300)
            plt.close()
            print(f"Saved: {drift_path}")
            
    # 5. Summary Table
    summary_rows = []
    has_apa = not df_apa.empty
    has_fp32 = not df_fp32.empty
    
    apa_train_acc = f"{df_apa.iloc[-1]['train_acc_pct']:.2f}%" if has_apa and 'train_acc_pct' in df_apa.columns else '-'
    fp32_train_acc = f"{df_fp32.iloc[-1]['train_acc_pct']:.2f}%" if has_fp32 and 'train_acc_pct' in df_fp32.columns else '-'
    summary_rows.append({'Metric': 'Final Train Accuracy', 'APA': apa_train_acc, 'Pure FP32 Baseline': fp32_train_acc})

    apa_test_acc = f"{df_apa.iloc[-1]['test_acc_pct']:.2f}%" if has_apa and 'test_acc_pct' in df_apa.columns else '-'
    fp32_test_acc = f"{df_fp32.iloc[-1]['test_acc_pct']:.2f}%" if has_fp32 and 'test_acc_pct' in df_fp32.columns else '-'
    summary_rows.append({'Metric': 'Final Test Accuracy', 'APA': apa_test_acc, 'Pure FP32 Baseline': fp32_test_acc})

    apa_peak = f"{df_apa['test_acc_pct'].max():.2f}% (Ep {df_apa.loc[df_apa['test_acc_pct'].idxmax(), 'epoch']})" if has_apa and 'test_acc_pct' in df_apa.columns else '-'
    fp32_peak = f"{df_fp32['test_acc_pct'].max():.2f}% (Ep {df_fp32.loc[df_fp32['test_acc_pct'].idxmax(), 'epoch']})" if has_fp32 and 'test_acc_pct' in df_fp32.columns else '-'
    summary_rows.append({'Metric': 'Peak Test Accuracy', 'APA': apa_peak, 'Pure FP32 Baseline': fp32_peak})

    apa_train_loss = f"{df_apa.iloc[-1]['train_loss']:.4f}" if has_apa and 'train_loss' in df_apa.columns else '-'
    fp32_train_loss = f"{df_fp32.iloc[-1]['train_loss']:.4f}" if has_fp32 and 'train_loss' in df_fp32.columns else '-'
    summary_rows.append({'Metric': 'Final Train Loss', 'APA': apa_train_loss, 'Pure FP32 Baseline': fp32_train_loss})

    apa_test_loss = f"{df_apa.iloc[-1]['test_loss']:.4f}" if has_apa and 'test_loss' in df_apa.columns else '-'
    fp32_test_loss = f"{df_fp32.iloc[-1]['test_loss']:.4f}" if has_fp32 and 'test_loss' in df_fp32.columns else '-'
    summary_rows.append({'Metric': 'Final Test Loss', 'APA': apa_test_loss, 'Pure FP32 Baseline': fp32_test_loss})

    apa_avg_time = f"{df_apa['epoch_time_sec'].mean():.2f}s" if has_apa and 'epoch_time_sec' in df_apa.columns else '-'
    fp32_avg_time = f"{df_fp32['epoch_time_sec'].mean():.2f}s" if has_fp32 and 'epoch_time_sec' in df_fp32.columns else '-'
    summary_rows.append({'Metric': 'Avg Time / Epoch', 'APA': apa_avg_time, 'Pure FP32 Baseline': fp32_avg_time})

    apa_tot_time = f"{df_apa['epoch_time_sec'].sum()/60:.2f} min" if has_apa and 'epoch_time_sec' in df_apa.columns else '-'
    fp32_tot_time = f"{df_fp32['epoch_time_sec'].sum()/60:.2f} min" if has_fp32 and 'epoch_time_sec' in df_fp32.columns else '-'
    summary_rows.append({'Metric': 'Total Training Time', 'APA': apa_tot_time, 'Pure FP32 Baseline': fp32_tot_time})

    if has_apa and has_fp32 and 'epoch_time_sec' in df_apa.columns and 'epoch_time_sec' in df_fp32.columns:
        t_apa = df_apa['epoch_time_sec'].sum()
        t_fp32 = df_fp32['epoch_time_sec'].sum()
        speedup = t_fp32 / t_apa if t_apa > 0 else 1.0
        speedup_str = f"{speedup:.2f}x Faster" if speedup >= 1.0 else f"{1/speedup:.2f}x Slower"
    else:
        speedup_str = '-'
    summary_rows.append({'Metric': 'Speedup Multiplier', 'APA': speedup_str, 'Pure FP32 Baseline': '1.00x (Ref)'})

    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.output_dir, 'benchmark_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("🏆 BENCHMARK COMPARISON: APA vs PURE FP32 BASELINE")
    print("=" * 70)
    try:
        from tabulate import tabulate
        print(tabulate(df_summary, headers='keys', tablefmt='github', showindex=False))
    except ImportError:
        print(df_summary.to_string(index=False))
    print("=" * 70)
    print(f"Summary CSV saved to: {csv_path}\n")

if __name__ == '__main__':
    main()
