"""Visualization for AdaGuard experiments."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import torch

matplotlib.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')


class AdaGuardPlotter:
    """Generates all AdaGuard experiment plots."""

    def __init__(self, output_dir='./plots'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def _save(self, fig, name):
        path = f"{self.output_dir}/{name}.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")

    def plot_fl_rounds(self, results, title='FL Rounds'):
        """Plot LeakScore metrics across FL rounds."""
        df = pd.DataFrame([
            {k: v for k, v in r.items() if isinstance(v, (int, float))}
            for r in results
        ])

        if 'round' not in df.columns:
            return

        rounds = df['round']
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'AdaGuard — {title}', fontsize=14, fontweight='bold')

        # Label LeakScore
        ax = axes[0, 0]
        for col, label in [('glmip_score', 'GLMIP'), ('confidence_gap', 'Conf Gap'),
                           ('cosine_leak_score', 'Cosine')]:
            if col in df.columns:
                ax.plot(rounds, df[col], 'o-', label=label, lw=2)
        ax.set_title('Label LeakScore')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Entropy LeakScore
        ax = axes[0, 1]
        for col, label in [('shannon_leak_score', 'Shannon'), ('renyi_leak_score', 'Rényi'),
                           ('min_entropy_leak_score', 'Min-Ent')]:
            if col in df.columns:
                ax.plot(rounds, df[col], 'o-', label=label, lw=2)
        ax.set_title('Entropy LeakScore')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Empirical LeakScore
        ax = axes[0, 2]
        for col, label in [('empirical_gradinversion', 'GradInv'), ('empirical_ginas', 'GI-NAS'),
                           ('empirical_ggcdm', 'GGCDM')]:
            if col in df.columns:
                ax.plot(rounds, df[col], 'o-', label=label, lw=2)
        ax.set_title('Empirical LeakScore')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Fisher & MaskCrypt
        ax = axes[1, 0]
        if 'fisher_concentration' in df.columns:
            ax.plot(rounds, df['fisher_concentration'], 'o-', color='#2ECC71',
                    label='Fisher Conc', lw=2)
        if 'maskcrypt_vuln_score' in df.columns:
            ax.plot(rounds, df['maskcrypt_vuln_score'], 's-', color='#E74C3C',
                    label='MaskCrypt Vuln', lw=2)
        ax.set_title('Fisher & MaskCrypt')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Combined LeakScore + Encryption
        ax = axes[1, 1]
        if 'combined_leakscore' in df.columns:
            ax.plot(rounds, df['combined_leakscore'], 'o-', color='#8E44AD',
                    label='Combined', lw=2.5)
        if 'actual_pct_encrypted' in df.columns:
            ax.plot(rounds, df['actual_pct_encrypted'], 's--', color='#E67E22',
                    label='Encrypt %', lw=2)
        ax.set_title('Combined LeakScore & Encryption')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[1, 2]
        if 'accuracy' in df.columns:
            ax.plot(rounds, df['accuracy'] * 100, 'o-', color='#3498DB', lw=2.5)
        ax.set_title('Model Accuracy (%)')
        ax.set_ylabel('Accuracy %')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, f'fl_rounds_{title.lower().replace(" ", "_")}')
        plt.show()

    def plot_encryption_comparison(self, all_results):
        """Compare encryption strategies side by side."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Encryption Strategy Comparison', fontsize=14, fontweight='bold')

        strategies = list(all_results.keys())
        colors = {'none': '#95A5A6', 'fisher': '#2ECC71',
                  'maskcrypt': '#E74C3C', 'full': '#3498DB'}

        # Accuracy over rounds
        ax = axes[0]
        for strategy in strategies:
            results = all_results[strategy]
            rounds = [r['round'] for r in results]
            accs = [r.get('accuracy', 0) * 100 for r in results]
            ax.plot(rounds, accs, 'o-', label=strategy.capitalize(),
                    color=colors.get(strategy, '#333'), lw=2)
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy %')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Combined LeakScore
        ax = axes[1]
        for strategy in strategies:
            results = all_results[strategy]
            rounds = [r['round'] for r in results]
            scores = [r.get('combined_leakscore', 0) for r in results]
            ax.plot(rounds, scores, 'o-', label=strategy.capitalize(),
                    color=colors.get(strategy, '#333'), lw=2)
        ax.set_title('Combined LeakScore')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Encryption overhead
        ax = axes[2]
        for strategy in strategies:
            results = all_results[strategy]
            rounds = [r['round'] for r in results]
            pcts = [r.get('actual_pct_encrypted', 0) * 100 for r in results]
            ax.plot(rounds, pcts, 'o-', label=strategy.capitalize(),
                    color=colors.get(strategy, '#333'), lw=2)
        ax.set_title('Parameters Encrypted (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, 'encryption_comparison')
        plt.show()

    def plot_weight_study(self, all_results):
        """Plot alpha/beta/gamma weight study results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Weight Study: Effect of alpha, beta, gamma',
                     fontsize=14, fontweight='bold')

        for (a, b, g), results in all_results.items():
            label = f"a={a},b={b},g={g}"
            rounds = [r['round'] for r in results]
            combined = [r.get('combined_leakscore', 0) for r in results]
            pcts = [r.get('actual_pct_encrypted', 0) * 100 for r in results]

            axes[0].plot(rounds, combined, 'o-', label=label, lw=1.5)
            axes[1].plot(rounds, pcts, 'o-', label=label, lw=1.5)

        axes[0].set_title('Combined LeakScore')
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title('Parameters Encrypted (%)')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, 'weight_study')
        plt.show()

    def plot_batch_size_experiment(self, results):
        """Plot batch size effect on leakage scores."""
        batch_sizes = sorted(results.keys())
        metrics_to_plot = [
            ('shannon_leak_score', 'Shannon'),
            ('renyi_leak_score', 'Rényi'),
            ('fisher_concentration', 'Fisher Conc'),
            ('maskcrypt_vuln_score', 'MaskCrypt Vuln'),
            ('empirical_gradinversion', 'GradInversion'),
            ('empirical_ginas', 'GI-NAS'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Batch Size Effect on LeakScores', fontsize=14, fontweight='bold')

        for idx, (metric, label) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            vals = [results[bs].get(metric, 0) for bs in batch_sizes]
            ax.bar(range(len(batch_sizes)), vals, color='#3498DB', alpha=0.7)
            ax.set_xticks(range(len(batch_sizes)))
            ax.set_xticklabels(batch_sizes)
            ax.set_title(label)
            ax.set_xlabel('Batch Size')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, 'batch_size_experiment')
        plt.show()

    def plot_noise_experiment(self, results):
        """Plot noise level effect on leakage scores."""
        noise_levels = sorted(results.keys())
        metrics_to_plot = [
            ('shannon_leak_score', 'Shannon'),
            ('empirical_gradinversion', 'GradInversion'),
            ('fisher_concentration', 'Fisher Conc'),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Noise Level Effect on LeakScores', fontsize=14, fontweight='bold')

        for idx, (metric, label) in enumerate(metrics_to_plot):
            ax = axes[idx]
            vals = [results[nl].get(metric, 0) for nl in noise_levels]
            ax.plot(range(len(noise_levels)), vals, 'o-', color='#E74C3C', lw=2)
            ax.set_xticks(range(len(noise_levels)))
            ax.set_xticklabels([str(nl) for nl in noise_levels])
            ax.set_title(label)
            ax.set_xlabel('Noise Sigma')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save(fig, 'noise_experiment')
        plt.show()

    def plot_fisher_maskcrypt_overlap(self, results, enc_pct=0.1):
        """Venn-style overlap between Fisher and MaskCrypt encryption targets."""
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
        if len(results) == 1:
            axes = [axes]

        fig.suptitle(f'Encryption Target Overlap: Fisher vs MaskCrypt (top {enc_pct*100:.0f}%)',
                     fontsize=13, fontweight='bold')

        for idx, (key, metrics) in enumerate(results.items()):
            ax = axes[idx]
            fi_mask = metrics.get('encrypt_mask', torch.tensor([]))
            mc_mask = metrics.get('maskcrypt_encrypt_mask', torch.tensor([]))

            if fi_mask.numel() == 0 or mc_mask.numel() == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(str(key))
                continue

            n = min(fi_mask.numel(), mc_mask.numel())
            fi_mask = fi_mask[:n].bool()
            mc_mask = mc_mask[:n].bool()

            both = (fi_mask & mc_mask).sum().item()
            fi_only = (fi_mask & ~mc_mask).sum().item()
            mc_only = (~fi_mask & mc_mask).sum().item()
            total = n

            labels = [f'Both\n{both} ({100*both/total:.1f}%)',
                      f'Fisher only\n{fi_only} ({100*fi_only/total:.1f}%)',
                      f'MaskCrypt only\n{mc_only} ({100*mc_only/total:.1f}%)']
            sizes = [both, fi_only, mc_only]
            colors = ['#8E44AD', '#2ECC71', '#E74C3C']

            if sum(sizes) > 0:
                ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
            ax.set_title(str(key))

        plt.tight_layout()
        self._save(fig, 'fisher_maskcrypt_overlap')
        plt.show()

    def plot_heatmaps(self, results_dict, experiment_name='Experiment'):
        """Plot heatmaps for LeakScore metrics across conditions."""
        from ..metrics import LABEL_METRICS, ENTROPY_METRICS, EMPIRICAL_METRICS

        all_metrics = LABEL_METRICS + ENTROPY_METRICS + EMPIRICAL_METRICS
        conditions = list(results_dict.keys())

        data = []
        for cond in conditions:
            row = [results_dict[cond].get(m, 0) for m in all_metrics]
            data.append(row)

        df = pd.DataFrame(data, index=[str(c) for c in conditions], columns=all_metrics)

        fig, ax = plt.subplots(figsize=(12, max(3, len(conditions) * 0.6)))
        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn_r', vmin=0, vmax=1,
                    ax=ax, linewidths=0.5, cbar_kws={'label': 'LeakScore'})
        ax.set_title(f'{experiment_name} — LeakScore Heatmap')
        plt.tight_layout()
        self._save(fig, f'heatmap_{experiment_name.lower().replace(" ", "_")}')
        plt.show()
