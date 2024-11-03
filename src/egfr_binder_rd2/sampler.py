import modal
from typing import List, Optional, Dict, Any
import logging
import pandas as pd
import torch
from datetime import datetime
import numpy as np
import json
import os
from pathlib import Path
import time

from egfr_binder_rd2.bt import BTEnsemble
from egfr_binder_rd2 import (
    LOGGING_CONFIG, MODAL_VOLUME_PATH, OUTPUT_DIRS, 
    ExpertType, ExpertConfig, EvolutionMetadata, PartialEnsembleExpertConfig, DEFAULT_EXPERT_CONFIGS)
from egfr_binder_rd2.fold import get_a3m_path
from egfr_binder_rd2.utils import hash_seq
# Set up logging
logging.basicConfig(level=LOGGING_CONFIG["level"], format=LOGGING_CONFIG["format"])
logger = logging.getLogger(__name__)

# Define the container image
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "pandas",
    "lightning",
    "wandb",
    "datasets",
    "peft",
    "torchmetrics",
)

app = modal.App("directed-evolution")
volume = modal.Volume.from_name("egfr_binders", create_if_missing=True)
wandb_secret = modal.Secret.from_name("anaka_personal_wandb_api_key")

with image.imports():
    import numpy as np

def safe_log_metrics(df: pd.DataFrame, metric_name: str, logger: logging.Logger) -> Dict[str, Any]:
    """Safely extract and log metrics, returning default values if columns don't exist."""
    try:
        if metric_name == 'pae_interaction':
            mean = df['pae_interaction_mean'].mean()
            std = df['pae_interaction_std'].mean()
            ucb = df['pae_interaction_ucb'].mean()
            return {'mean': mean, 'std': std, 'ucb': ucb}
        elif metric_name == 'i_ptm':
            mean = df['i_ptm_mean'].mean()
            std = df['i_ptm_std'].mean()
            ucb = df['i_ptm_ucb'].mean()
            return {'mean': mean, 'std': std, 'ucb': ucb}
        elif metric_name == 'binder_plddt':
            mean = df['binder_plddt_mean'].mean()
            std = df['binder_plddt_std'].mean()
            ucb = df['binder_plddt_ucb'].mean()
            return {'mean': mean, 'std': std, 'ucb': ucb}
        elif metric_name == 'sequence_log_pll':
            mean = df['sequence_log_pll'].mean()
            std = df['sequence_log_pll_std'].mean()
            ucb = df['sequence_log_pll_ucb'].mean()
            return {'mean': mean, 'std': std, 'ucb': ucb}
        else:
            mean = df[metric_name].mean()
            std = df[metric_name].std() if f'{metric_name}_std' in df else None
            return {'mean': mean, 'std': std}
    except KeyError as e:
        logger.error(f"Failed to extract {metric_name} metrics: {str(e)}")
        return {'mean': None, 'std': None, 'ucb': None}

@app.cls(
    image=image,
    timeout=86400,  # 24 hours
    volumes={MODAL_VOLUME_PATH: volume},
    secrets=[wandb_secret],
)
class DirectedEvolution:
    
    @modal.enter()
    def setup(self):
        # Look up remote functions
        self.sample_sequences = modal.Function.lookup("bt-training", "sample_sequences")
        self.train_experts = modal.Function.lookup("bt-training", "train_experts_async")
        self.process_sequences = modal.Function.lookup("esm2-inference", "process_sequences")
        self.update_pll_metrics = modal.Function.lookup("esm2-inference", "update_pll_metrics")
        self.fold_binder = modal.Function.lookup("simplefold", "fold_binder")
        self.parallel_fold_binder = modal.Function.lookup("simplefold", "parallel_fold_binder")
        self.update_metrics = modal.Function.lookup("simplefold", "update_metrics_for_all_folded")
        self.get_msa = modal.Function.lookup("simplefold", "get_msa_for_binder")
        
        logger.info("Initialized DirectedEvolution workflow.")

        # Add a class variable to store all tested sequences
        self.all_tested_sequences_df = pd.DataFrame()

    @staticmethod
    def should_retrain(generation: int, retrain_frequency: int = 5) -> bool:
        return generation > 0 and generation % retrain_frequency == 0

    @staticmethod
    def get_cyclic_temperature(generation: int, period: int = 5, min_temp: float = 0.5, max_temp: float = 2.0) -> float:
        """Calculate temperature based on cosine cycle.
        
        Args:
            generation: Current generation number
            period: Number of generations per cycle
            min_temp: Minimum temperature value
            max_temp: Maximum temperature value
            
        Returns:
            float: Current temperature value starting at min_temp
        """
        # Using -cos to start at minimum temperature
        cycle_progress = -np.cos(2 * np.pi * (generation-1) / period)
        temp_range = max_temp - min_temp
        current_temp = min_temp + (cycle_progress + 1) * temp_range / 2
        return current_temp

    @modal.method()
    def run_evolution_cycle(
        self,
        parent_binder_seqs: List[str],
        generations: int = 5,
        n_to_fold: int = 10,
        num_parents: int = 5,
        top_k: int = 10,
        n_parallel_chains: int = 32,
        n_serial_chains: int = 1,
        n_steps: int = 250,
        max_mutations: int = -1,
        evoprotgrad_top_fraction: float = 0.2,
        parent_selection_temperature: float = 2.0,
        temp_cycle_period: int = 5,
        min_sampling_temp: float = 0.5,
        max_sampling_temp: float = 2.0,
        retrain_frequency: int = 5,
        seed: int = 42,
        select_from_current_gen_only: bool = False,
        expert_configs: Optional[List[ExpertConfig]] = None,
    ):
        """Run multiple cycles of directed evolution with multiple parent sequences."""
        # Use default expert configs if none provided
        if expert_configs is None:
            expert_configs = DEFAULT_EXPERT_CONFIGS

        current_parent_seqs = parent_binder_seqs.copy()
        all_final_sequences = []
        
        # Create metadata tracker
        config = {
            "generations": generations,
            "n_to_fold": n_to_fold,
            "num_parents": num_parents,
            "top_k": top_k,
            "n_parallel_chains": n_parallel_chains,
            "n_serial_chains": n_serial_chains,
            "n_steps": n_steps,
            "max_mutations": max_mutations,
            "evoprotgrad_top_fraction": evoprotgrad_top_fraction,
            "parent_selection_temperature": parent_selection_temperature,
            "temp_cycle_period": temp_cycle_period,
            "min_sampling_temp": min_sampling_temp,
            "max_sampling_temp": max_sampling_temp,
            "retrain_frequency": retrain_frequency,
            "seed": seed
        }
        metadata = EvolutionMetadata.create(config, parent_binder_seqs)
        
        for gen in range(1, generations + 1):
            logger.info(f"=== Generation {gen} ===")
            logger.info(f"Number of parent sequences: {len(current_parent_seqs)}")
            
            # Check if retraining is needed
            if self.should_retrain(gen, retrain_frequency):
                logger.info(f"Generation {gen}: Starting expert retraining in background...")
                # Fire and forget - don't store or wait for the future
                _ = self.train_experts.spawn()
                logger.info("Expert retraining started asynchronously")
            
            # Calculate sequences to sample per parent
            seqs_per_parent = max(1, n_to_fold // len(current_parent_seqs))
            logger.info(f"Sampling {seqs_per_parent} sequences per parent to reach total of ~{n_to_fold}")
            
            # Calculate current temperature based on cosine cycle
            current_temp = self.get_cyclic_temperature(
                gen,
                period=temp_cycle_period,
                min_temp=min_sampling_temp,
                max_temp=max_sampling_temp
            )
            logger.info(f"Using sampling temperature: {current_temp:.2f} for generation {gen}")
            
            # Process each parent sequence
            all_variants = []
            logger.info("Sampling new binder sequences using EvoProtGrad...")
            evoprotgrad_df = self.sample_sequences.remote(
                sequences=current_parent_seqs,  # Pass all parent sequences at once
                expert_configs=expert_configs,
                n_parallel_chains=n_parallel_chains,
                n_serial_chains=n_serial_chains,
                n_steps=n_steps,
                max_mutations=max_mutations,
                seed=seed + gen,
                run_inference=True,
            )
            logger.info(f"Generated {len(evoprotgrad_df)} variant sequences")

            evoprotgrad_df['i_ptm_ucb_rank'] = evoprotgrad_df['i_ptm_ucb'].rank(pct=True)
            evoprotgrad_df['pae_interaction_ucb_rank'] = evoprotgrad_df['pae_interaction_ucb'].rank(pct=True)
            evoprotgrad_df['sequence_log_pll_ucb_rank'] = evoprotgrad_df['sequence_log_pll_ucb'].rank(pct=True)
            evoprotgrad_df['fitness_ucb'] = (evoprotgrad_df['i_ptm_ucb_rank'] + evoprotgrad_df['pae_interaction_ucb_rank'] + evoprotgrad_df['sequence_log_pll_ucb_rank']) / 3
            evoprotgrad_df = evoprotgrad_df.sort_values('fitness_ucb', ascending=False).reset_index(drop=True)

            # Sample sequences from the top fraction, now considering parent information
            all_variants_with_parents = []  # New list to track variants with their parents
            logger.info(f"Sampling ~{seqs_per_parent} sequences per parent to reach total of ~{n_to_fold}")
            for parent_idx, parent_seq in enumerate(current_parent_seqs):
                parent_variants = evoprotgrad_df[evoprotgrad_df['parent_seq'] == parent_seq]
                if len(parent_variants) > 0:
                    sampled_variants = self.sample_from_evoprotgrad_sequences(
                        parent_variants,
                        top_fraction=evoprotgrad_top_fraction,
                        sample_size=seqs_per_parent,
                        temperature=current_temp
                    )
                    # Store variants with their parent information
                    all_variants_with_parents.extend([(variant, parent_seq) for variant in sampled_variants])
                    all_variants.extend(sampled_variants)
                else:
                    logger.warning(f"No variants generated for parent sequence {parent_idx + 1}")
            
            # Save selected variants metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            selected_variants_df = evoprotgrad_df[evoprotgrad_df['sequence'].isin(all_variants)].copy()
            selected_variants_df['generation'] = gen
            selected_variants_df['timestamp'] = datetime.now().isoformat()
            
            # Log inference predictions before folding
            logger.info("\nInference Predictions for Selected Sequences:")
            logger.info(f"Number of sequences selected for folding: {len(selected_variants_df)}")
            logger.info("\nPredicted Metrics Statistics:")
            try:
                pae_metrics = safe_log_metrics(selected_variants_df, 'pae_interaction', logger)
                ptm_metrics = safe_log_metrics(selected_variants_df, 'i_ptm', logger)
                plddt_metrics = safe_log_metrics(selected_variants_df, 'binder_plddt', logger)
                pll_metrics = safe_log_metrics(selected_variants_df, 'sequence_log_pll_ucb', logger)

                if pae_metrics['mean'] is not None:
                    logger.info(f"iPAE (mean ± std): {pae_metrics['mean']:.2f} ± {pae_metrics['std']:.2f}")
                    logger.info(f"iPAE UCB: {pae_metrics['ucb']:.2f}")
                if ptm_metrics['mean'] is not None:
                    logger.info(f"iPTM (mean ± std): {ptm_metrics['mean']:.2f} ± {ptm_metrics['std']:.2f}")
                    logger.info(f"iPTM UCB: {ptm_metrics['ucb']:.2f}")
                if plddt_metrics['mean'] is not None:
                    logger.info(f"pLDDT (mean ± std): {plddt_metrics['mean']:.2f} ± {plddt_metrics['std']:.2f}")
                if pll_metrics['mean'] is not None:
                    logger.info(f"PLL: {pll_metrics['mean']:.2f}")
            except Exception as e:
                logger.error(f"Error logging metrics statistics: {str(e)}")

            # Log a few example sequences with their predictions
            for _, row in selected_variants_df.iterrows():
                logger.info(
                   
                    f"bt iPAE={row['pae_interaction_mean']:.2f} (UCB={row['pae_interaction_ucb']:.2f}), "
                    f"bt iPTM={row['i_ptm_mean']:.2f} (UCB={row['i_ptm_ucb']:.2f}), "
                    f"bt pLDDT={row['binder_plddt_mean']:.2f}, "
                    f"bt PLL={row['sequence_log_pll_ucb']:.2f} "
                    f"Sequence={row['sequence']} "
                )

            # Save to CSV file with timestamp in name
            output_path = MODAL_VOLUME_PATH / OUTPUT_DIRS['inference_results'] / f'selected_variants_metrics_gen_{gen}_{timestamp}.csv'
            output_path.parent.mkdir(exist_ok=True)
            selected_variants_df.to_csv(output_path, index=False)
            logger.info(f"Saved selected variants metrics to {output_path}")

            # # Save PLL values in ESM2 format
            # results_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS['esm2_pll_results']
            # results_dir.mkdir(parents=True, exist_ok=True)

            # for _, row in selected_variants_df.iterrows():
            #     seq = row['sequence']
            #     seq_hash = f"bdr_{hash_seq(seq)}"
            #     result = {
            #         "sequence": seq,
            #         "sequence_hash": seq_hash,
            #         "sequence_log_pll": row['sequence_log_pll'],
            #         "normalized_log_pll": row['sequence_log_pll'] / len(seq),  # Approximate normalization
            #         "sequence_length": len(seq)
            #     }
                
            #     result_path = results_dir / f"{seq_hash}.json"
            #     with open(result_path, 'w') as f:
            #         json.dump(result, f, indent=2)

            # logger.info(f"Saved {len(selected_variants_df)} PLL results in ESM2 format")

            
            
            # Process all variants together
            logger.info(f"\nProcessing {len(all_variants)} total variants...")
            
            # Extract sequences and parents for folding
            variant_seqs = [v[0] for v in all_variants_with_parents]
            parent_seqs = [v[1] for v in all_variants_with_parents]
            
            # ensure that parent a3ms exist
            seqs_that_need_msa = []
            for parent_seq in parent_seqs:
                a3m_path = get_a3m_path(parent_seq)
                if not a3m_path.exists():
                    seqs_that_need_msa.append(parent_seq)
            
            seqs_that_need_msa = list(set(seqs_that_need_msa))
            if len(seqs_that_need_msa) > 0:
                logger.info(f"Retrieving MSA for {len(seqs_that_need_msa)} parent sequences...")
                msa_paths = self.get_msa.remote(seqs_that_need_msa)
                logger.info(f"Retrieved MSA for {len(msa_paths)} parent sequences")
            volume.reload()
            
            # Get folding metrics, now passing parent sequences
            logger.info(f"Folding {len(variant_seqs)} binder sequences...")
            fold_results = self.parallel_fold_binder.remote(variant_seqs, parent_seqs)
            volume.reload()
            

            #Get PLL metrics
            logger.info(f"Processing sequences to obtain PLL metrics...")
            self.process_sequences.remote(variant_seqs)
            
            
            volume.reload()
            metrics_df = self.update_metrics.remote()
            logger.info(f"Obtained folding metrics for {len(metrics_df)} sequences")
            pll_df = self.update_pll_metrics.remote()
            logger.info(f"Obtained PLL metrics for {len(pll_df)} sequences")


            # Merge and calculate fitness
            logger.info("Merging metrics and calculating fitness scores...")
            combined_df = pll_df.merge(metrics_df, left_on='sequence', right_on='binder_sequence', how="inner")
            logger.info(f"Successfully merged metrics for {len(combined_df)} sequences")
            
            # Add retry logic for current generation sequences
            max_retries = 3
            retry_delay = 30  # seconds
            
            if select_from_current_gen_only:
                expected_sequences = len(all_variants)  # Number of sequences we expect to have folded
                for attempt in range(max_retries):
                    logger.info(f"Attempt {attempt + 1}/{max_retries}: Calculating fitness using current generation sequences")
                    logger.info(f'Current generation sequences: {len(all_variants)} expected')
                    ranking_df = combined_df[combined_df['binder_sequence'].isin(all_variants)].copy()  # Make explicit copy
                    logger.info(f"Found {len(ranking_df)}/{expected_sequences} sequences from current generation")
                    
                    if len(ranking_df) >= expected_sequences * 0.9:  # Allow for 10% failure rate
                        break
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Missing {expected_sequences - len(ranking_df)} sequences. Waiting {retry_delay} seconds before retry...")
                        time.sleep(retry_delay)
                        # Reload the metrics
                        volume.reload()
                        metrics_df = self.update_metrics.remote()
                        pll_df = self.update_pll_metrics.remote()
                        combined_df = pll_df.merge(metrics_df, left_on='sequence', right_on='binder_sequence', how="inner")
                
                if len(ranking_df) < expected_sequences * 0.5:  # If we have less than 50% of expected sequences
                    logger.warning(f"Only found {len(ranking_df)}/{expected_sequences} sequences after retries. Falling back to using all historical sequences.")
                    ranking_df = combined_df
                else:
                    logger.info(f"Proceeding with {len(ranking_df)}/{expected_sequences} sequences from current generation")

            else:
                logger.info("Calculating fitness using all historical sequences")
                ranking_df = combined_df.copy()  # Make explicit copy
                logger.info(f"Using all {len(ranking_df)} historical sequences")

            if len(ranking_df) == 0:
                logger.warning("No sequences found for ranking. Using parent sequences for next generation.")
                current_parent_seqs = current_parent_seqs
                continue

            # Calculate ranks and fitness
            ranking_df.loc[:, 'pae_interaction_rank'] = 1 - ranking_df['pae_interaction'].rank(pct=True)
            ranking_df.loc[:, 'i_ptm_rank'] = ranking_df['i_ptm'].rank(pct=True)
            ranking_df.loc[:, 'sequence_log_pll_rank'] = ranking_df['sequence_log_pll'].rank(pct=True)
            ranking_df.loc[:, 'fitness'] = (
                ranking_df['pae_interaction_rank'] + 
                ranking_df['i_ptm_rank'] + 
                ranking_df['sequence_log_pll_rank']
            ) / 3
            
            # Sort by fitness
            ranking_df = ranking_df.sort_values('fitness', ascending=False).reset_index(drop=True)
            logger.info(f"Successfully ranked {len(ranking_df)} sequences")

            # Log statistics for this generation
            logger.info("\nGeneration Statistics:")
            logger.info(f"Mean PLL: {combined_df['sequence_log_pll'].mean():.2f}")
            logger.info(f"Mean iPAE: {combined_df['pae_interaction'].mean():.2f}")
            logger.info(f"Mean i_ptm: {combined_df['i_ptm'].mean():.2f}")
            logger.info(f"Best PLL: {combined_df['sequence_log_pll'].max():.2f}")
            logger.info(f"Best iPAE: {combined_df['pae_interaction'].min():.2f}")
            logger.info(f"Best i_ptm: {combined_df['i_ptm'].max():.2f}")
            
            # Select top sequences
            logger.info(f"\nSelecting top {top_k} sequences based on fitness scores...")
            if select_from_current_gen_only:
                logger.info("Selecting from current generation only")
            else:
                logger.info("Selecting from all historical sequences")
            top_sequences = ranking_df.head(top_k)
            
            # Log the top sequences with detailed metrics
            logger.info("\nTop sequences selected for generation:")
            for idx, row in top_sequences.iterrows():
                logger.info(
                    f"Sequence={row['binder_sequence']} "
                    f"Sequence {idx + 1:03d}: "
                    f"Length={len(row['binder_sequence'])}, "
                    f"Fitness={row['fitness']:.3f} "
                    f"(PLL_rank={row['sequence_log_pll_rank']:.3f}, "
                    f"iPAE_rank={row['pae_interaction_rank']:.3f}, "
                    f"iPTM_rank={row['i_ptm_rank']:.3f}), "
                    f"Raw: PLL={row['sequence_log_pll']:.2f}, "
                    f"iPAE={row['pae_interaction']:.2f}, "
                    f"iPTM={row['i_ptm']:.2f}, "
                    
                )
            
            # Select parents for next generation
            current_parent_seqs = self.select_multiple_parents_probabilistically(
                top_sequences,
                num_parents=num_parents,
                temperature=parent_selection_temperature
            )
            
            # Log selected parents
            logger.info(f"\nSelected {len(current_parent_seqs)} parents for next generation:")
            for idx, parent in enumerate(current_parent_seqs, 1):
                parent_metrics = top_sequences[top_sequences['binder_sequence'] == parent].iloc[0]
                logger.info(
                    f"Sequence={parent} "
                    f"Parent {idx:03d}: "
                    f"Length={len(parent)}, "
                    f"PLL={parent_metrics['sequence_log_pll']:.2f}, "
                    f"PAE={parent_metrics['pae_interaction']:.2f}, "
                    f"iPTM={parent_metrics['i_ptm']:.2f}, "
                    f"Fitness={parent_metrics['fitness']:.2f}, "
                    
                )
            
            # Store top sequences for final return
            all_final_sequences.extend(top_sequences['binder_sequence'].tolist())
            
            # Log generation metrics with more detail
            gen_metrics = {
                "mean_pll": float(combined_df['sequence_log_pll'].mean()),
                "mean_ipae": float(combined_df['pae_interaction'].mean()),
                "mean_iptm": float(combined_df['i_ptm'].mean()),
                "best_pll": float(combined_df['sequence_log_pll'].max()),
                "best_ipae": float(combined_df['pae_interaction'].min()),
                "best_iptm": float(combined_df['i_ptm'].max()),
                "parent_sequences": current_parent_seqs,
                "top_sequences": [{
                    "sequence": row['binder_sequence'],
                    "fitness": float(row['fitness']),
                    "pll": float(row['sequence_log_pll']),
                    "pll_rank": float(row['sequence_log_pll_rank']),
                    "ipae": float(row['pae_interaction']),
                    "ipae_rank": float(row['pae_interaction_rank']),
                    "iptm": float(row['i_ptm']),
                    "iptm_rank": float(row['i_ptm_rank'])
                } for _, row in top_sequences.iterrows()]
            }
            metadata.add_generation(gen, gen_metrics)

            # When comparing predicted vs actual metrics
            logger.info("\nComparing predicted vs actual metrics:")

            # Get predictions from EvoProtGrad for the folded sequences
            predictions_df = selected_variants_df[selected_variants_df['sequence'].isin(variant_seqs)]
            actuals_df = combined_df[combined_df['binder_sequence'].isin(variant_seqs)]

            if len(predictions_df) > 0 and len(actuals_df) > 0:
                # Calculate average metrics
                try:
                    pred_metrics = {
                        'pae_interaction': predictions_df['pae_interaction_mean'].mean(),  # Predicted uses _mean
                        'i_ptm': predictions_df['i_ptm_mean'].mean(),  # Predicted uses _mean
                        'pll': predictions_df['sequence_log_pll'].mean(),
                    }
                    
                    actual_metrics = {
                        'pae_interaction': actuals_df['pae_interaction'].mean(),  # Actual doesn't use _mean
                        'i_ptm': actuals_df['i_ptm'].mean(),  # Actual doesn't use _mean
                        'pll': actuals_df['sequence_log_pll'].mean(),
                    }
                    
                    logger.info("Average metrics comparison (predicted vs actual):")
                    logger.info(f"iPAE: {pred_metrics['pae_interaction']:.2f} vs {actual_metrics['pae_interaction']:.2f}")
                    logger.info(f"iPTM: {pred_metrics['i_ptm']:.2f} vs {actual_metrics['i_ptm']:.2f}")
                    logger.info(f"PLL: {pred_metrics['pll']:.2f} vs {actual_metrics['pll']:.2f}")
                except Exception as e:
                    logger.error(f"Error calculating average metrics: {str(e)}")

                # Calculate correlations
                try:
                    # Rename columns during merge to avoid confusion
                    predictions_df = predictions_df.rename(columns={
                        'pae_interaction_mean': 'pae_interaction_pred',
                        'i_ptm_mean': 'i_ptm_pred',
                        'sequence_log_pll': 'sequence_log_pll_pred'
                    })
                    
                    merged_df = predictions_df.merge(
                        actuals_df,
                        left_on='sequence',
                        right_on='binder_sequence'
                    )
                    
                    if len(merged_df) > 1:  # Need at least 2 points for correlation
                        pae_corr = np.corrcoef(merged_df['pae_interaction_pred'], merged_df['pae_interaction'])[0,1]
                        ptm_corr = np.corrcoef(merged_df['i_ptm_pred'], merged_df['i_ptm'])[0,1]
                        pll_corr = np.corrcoef(merged_df['sequence_log_pll_pred'], merged_df['sequence_log_pll'])[0,1]
                        
                        logger.info("\nCorrelations between predicted and actual values:")
                        logger.info(f"iPAE correlation: {pae_corr:.3f}")
                        logger.info(f"iPTM correlation: {ptm_corr:.3f}")
                        logger.info(f"PLL correlation: {pll_corr:.3f}")
                    else:
                        logger.warning("Not enough matched sequences to calculate correlations")
                except Exception as e:
                    logger.error(f"Error calculating correlations: {str(e)}")
            else:
                logger.warning("Not enough data to compare predicted vs actual metrics")

            # After getting metrics, append new sequences with actual results to our accumulated dataframe
            generation_df = actuals_df.copy()  # Only use actuals_df instead of combined_df
            generation_df['generation'] = gen
            self.all_tested_sequences_df = pd.concat([self.all_tested_sequences_df, generation_df], ignore_index=True)
            
            # Remove duplicates, keeping the most recent evaluation
            self.all_tested_sequences_df = (self.all_tested_sequences_df
                .sort_values('generation', ascending=False)
                .drop_duplicates(subset='binder_sequence', keep='first')
                .reset_index(drop=True))

            # Use all_tested_sequences_df for ranking
            logger.info("Calculating fitness using all historically tested sequences")
            ranking_df = self.all_tested_sequences_df.copy()
            logger.info(f"Using {len(ranking_df)} historically tested sequences for ranking")

            if len(ranking_df) == 0:
                logger.warning("No sequences found for ranking. Using parent sequences for next generation.")
                current_parent_seqs = current_parent_seqs
                continue

            # Calculate ranks and fitness using all historical data
            ranking_df.loc[:, 'pae_interaction_rank'] = 1 - ranking_df['pae_interaction'].rank(pct=True)
            ranking_df.loc[:, 'i_ptm_rank'] = ranking_df['i_ptm'].rank(pct=True)
            ranking_df.loc[:, 'sequence_log_pll_rank'] = ranking_df['sequence_log_pll'].rank(pct=True)
            ranking_df.loc[:, 'fitness'] = (
                ranking_df['pae_interaction_rank'] + 
                ranking_df['i_ptm_rank'] + 
                ranking_df['sequence_log_pll_rank']
            ) / 3

            # Sort by fitness
            ranking_df = ranking_df.sort_values('fitness', ascending=False).reset_index(drop=True)
            logger.info(f"Successfully ranked {len(ranking_df)} sequences")

            # Log statistics for this generation
            logger.info("\nGeneration Statistics:")
            logger.info(f"Mean PLL: {combined_df['sequence_log_pll'].mean():.2f}")
            logger.info(f"Mean iPAE: {combined_df['pae_interaction'].mean():.2f}")
            logger.info(f"Mean i_ptm: {combined_df['i_ptm'].mean():.2f}")
            logger.info(f"Best PLL: {combined_df['sequence_log_pll'].max():.2f}")
            logger.info(f"Best iPAE: {combined_df['pae_interaction'].min():.2f}")
            logger.info(f"Best i_ptm: {combined_df['i_ptm'].max():.2f}")
            
            # Select top sequences
            logger.info(f"\nSelecting top {top_k} sequences based on fitness scores...")
            if select_from_current_gen_only:
                logger.info("Selecting from current generation only")
            else:
                logger.info("Selecting from all historical sequences")
            top_sequences = ranking_df.head(top_k)
            
            # Log the top sequences with detailed metrics
            # logger.info("\nTop sequences selected for generation:")
            # for idx, row in top_sequences.iterrows():
            #     logger.info(
            #         f"Sequence={row['binder_sequence']} "
            #         f"Sequence {idx + 1:03d}: "
            #         f"Length={len(row['binder_sequence'])}, "
            #         f"Fitness={row['fitness']:.3f} "
            #         f"(PLL_rank={row['sequence_log_pll_rank']:.3f}, "
            #         f"iPAE_rank={row['pae_interaction_rank']:.3f}, "
            #         f"iPTM_rank={row['i_ptm_rank']:.3f}), "
            #         f"Raw: PLL={row['sequence_log_pll']:.2f}, "
            #         f"iPAE={row['pae_interaction']:.2f}, "
            #         f"iPTM={row['i_ptm']:.2f}, "
                    
            #     )
            
            current_parent_seqs = self.select_multiple_parents_probabilistically(
                top_sequences,
                num_parents=num_parents,
                temperature=parent_selection_temperature
            )
            
            # Log selected parents
            logger.info(f"\nSelected {len(current_parent_seqs)} parents for next generation:")
            for idx, parent in enumerate(current_parent_seqs, 1):
                parent_metrics = top_sequences[top_sequences['binder_sequence'] == parent].iloc[0]
                logger.info(
                    f"Sequence={parent} "
                    f"Parent {idx:03d}: "
                    f"Length={len(parent)}, "
                    f"PLL={parent_metrics['sequence_log_pll']:.2f} (rank={parent_metrics['sequence_log_pll_rank']:.2f}), "
                    f"PAE={parent_metrics['pae_interaction']:.2f} (rank={parent_metrics['pae_interaction_rank']:.2f}), "
                    f"iPTM={parent_metrics['i_ptm']:.2f} (rank={parent_metrics['i_ptm_rank']:.2f}), "
                    f"Fitness={parent_metrics['fitness']:.2f}, "
                    
                )
            
            # Store top sequences for final return
            all_final_sequences.extend(top_sequences['binder_sequence'].tolist())
            
            # Log generation metrics with more detail
            gen_metrics = {
                "mean_pll": float(combined_df['sequence_log_pll'].mean()),
                "mean_ipae": float(combined_df['pae_interaction'].mean()),
                "mean_iptm": float(combined_df['i_ptm'].mean()),
                "best_pll": float(combined_df['sequence_log_pll'].max()),
                "best_ipae": float(combined_df['pae_interaction'].min()),
                "best_iptm": float(combined_df['i_ptm'].max()),
                "parent_sequences": current_parent_seqs,
                "top_sequences": [{
                    "sequence": row['binder_sequence'],
                    "fitness": float(row['fitness']),
                    "pll": float(row['sequence_log_pll']),
                    "pll_rank": float(row['sequence_log_pll_rank']),
                    "ipae": float(row['pae_interaction']),
                    "ipae_rank": float(row['pae_interaction_rank']),
                    "iptm": float(row['i_ptm']),
                    "iptm_rank": float(row['i_ptm_rank'])
                } for _, row in top_sequences.iterrows()]
            }
            metadata.add_generation(gen, gen_metrics)
            os.makedirs(MODAL_VOLUME_PATH /OUTPUT_DIRS["evolution_trajectories"], exist_ok=True)
            metadata.save(MODAL_VOLUME_PATH /OUTPUT_DIRS["evolution_trajectories"])

        logger.info("Directed evolution cycle completed.")
        # Save metadata
        metadata.save(MODAL_VOLUME_PATH /OUTPUT_DIRS["evolution_trajectories"])
        return list(set(all_final_sequences))  # Remove duplicates


    @staticmethod
    def sample_from_evoprotgrad_sequences(df, top_fraction=0.25, sample_size=10, temperature=1.0):
        """
        Sample sequences from the top fraction based on fitness_ucb scores.
        
        Args:
            df (pd.DataFrame): DataFrame with sequences and fitness_ucb scores
            top_fraction (float): Fraction of top sequences to consider (0-1)
            sample_size (int): Number of sequences to sample
            temperature (float): Temperature for softmax; higher values = more diversity
            
        Returns:
            list[str]: Sampled sequences
        """
        # Calculate number of sequences to keep
        n_keep = max(int(len(df) * top_fraction), sample_size)
        
        # Use fitness_ucb for weighted sampling
        top_df = df.nlargest(n_keep, 'fitness_ucb') 
        scores = top_df['fitness_ucb'].values
        
        # Apply temperature scaling
        scores = scores / temperature
        probabilities = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        probabilities = probabilities / probabilities.sum()
        
        # Sample sequences
        selected_indices = np.random.choice(
            len(top_df), 
            size=min(sample_size, len(top_df)), 
            p=probabilities,
            replace=False
        )
        
        return top_df.iloc[selected_indices]['sequence'].tolist()

    @staticmethod
    def select_parent_probabilistically(df, temperature=2.0):
        """
        Select parent sequence probabilistically based on fitness scores.
        
        Args:
            df (pd.DataFrame): DataFrame with sequences and fitness_score
            temperature (float): Temperature for softmax; higher values = more diversity
            
        Returns:
            str: Selected parent sequence
        """
        # Convert fitness scores to probabilities (lower is better)
        scores = df['fitness'].values
        scores = scores / temperature  # Apply temperature scaling
        probabilities = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        probabilities = probabilities / probabilities.sum()
        
        # Sample one sequence
        selected_idx = np.random.choice(len(df), p=probabilities)
        return df.iloc[selected_idx]

    @staticmethod
    def select_multiple_parents_probabilistically(df, num_parents: int, temperature: float = 2.0):
        """
        Select multiple parent sequences probabilistically based on fitness scores.
        
        Args:
            df (pd.DataFrame): DataFrame with sequences and fitness_score
            num_parents (int): Number of parent sequences to select
            temperature (float): Temperature for softmax; higher values = more diversity
            
        Returns:
            List[str]: Selected parent sequences
        """
        # Convert fitness scores to probabilities (lower is better)
        scores = df['fitness'].values
        scores = scores / temperature  # Apply temperature scaling
        probabilities = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        probabilities = probabilities / probabilities.sum()
        
        # Sample sequences
        selected_indices = np.random.choice(
            len(df),
            size=min(num_parents, len(df)),
            p=probabilities,
            replace=False
        )
        return df.iloc[selected_indices]['binder_sequence'].tolist()

class UCBSequenceSelector:
    def __init__(self, exploration_weight: float = 2.0):
        self.exploration_weight = exploration_weight
        self.sequence_counts = {}
        self.sequence_rewards = {}
        self.total_trials = 0
        
    def calculate_ucb(self, sequence: str, mean_pred: float, std_pred: float) -> float:
        n = self.sequence_counts.get(sequence, 0)
        
        # If sequence never seen, use high uncertainty
        if n == 0:
            return mean_pred + self.exploration_weight * std_pred
        
        # Calculate UCB score
        exploitation = self.sequence_rewards[sequence] / n
        exploration = self.exploration_weight * np.sqrt(np.log(self.total_trials) / n)
        return exploitation + exploration
    
    def select_sequences(
        self,
        candidates: List[str],
        ensemble_predictions: Dict[str, np.ndarray],
        n_select: int
    ) -> List[str]:
        """Select sequences using UCB strategy"""
        ucb_scores = []
        
        for i, seq in enumerate(candidates):
            mean_pred = ensemble_predictions["mean"][i]
            std_pred = ensemble_predictions["std"][i]
            ucb_score = self.calculate_ucb(seq, mean_pred, std_pred)
            ucb_scores.append(ucb_score)
        
        # Select top sequences by UCB score
        selected_indices = np.argsort(ucb_scores)[-n_select:]
        return [candidates[i] for i in selected_indices]
    
    def update(self, sequence: str, reward: float):
        """Update sequence statistics after evaluation"""
        self.sequence_counts[sequence] = self.sequence_counts.get(sequence, 0) + 1
        current_reward = self.sequence_rewards.get(sequence, 0)
        self.sequence_rewards[sequence] = current_reward + reward
        self.total_trials += 1

@app.local_entrypoint()
def main():
    # Define multiple parent sequences
    # parent_binder_seqs = [
    # #    'SYDGYCLNGGVCMHIESLDSYTCNCIGYSGDRCQTRDLRWWELR'
    #     # 'SYDGYCLNGGVCMHIESLDSYTCNCIGYSGDRCQTRDLRWWELR'
    #     # 'SSFSACPSSYDGICSNGGVCRYIQTLTSYTCQCPPGYTGDRCQTFDIRLLELRG',
    #     'SYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELR',
    # ] * 10
    
    # parent_binder_seqs = [
    #     'SYDGYCLNKGVCHHIESLDSYTCQCVIGYSGDRCQTRDLRFLELQ'
    # ]
    # parent_binder_seqs = [
    #     'AERMRRRFESIVEIHEEWAKEVLENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS',
         
    # ]
    # parent_binder_seqs = [
    #     'SLFSRCPRRYHGICGNNGQCRYAINLRTYTCRCVSGYTGDRCQELDIRYLLLLN',
    #     'SLFSKCPSKFHGICNNNGVCRYAINLRSYTCICLEGYTGDRCQEIDIRYLLLQL',
    #     'SLFSACPYRYHGICKNNGQCRYAISLRSGTCHCVSGYTGYRCQEIDIRYLLLRY',
    #     'SLFSKCPRRYHGICGNNGLCRYAINLRTYTCRCVSGYTGDRCQELDIRYLLLLN',
    #     'SLFSRCPRRYHGICGNNGLCRYAINLRTYTCRCVSGYTGDRCQELDIRYLLLRN',
    #     'NLFSICPRRYRGICKNNGSCRYAINLRTYTCQCVSGYTGARCQELDIRYLLLRY',
    #     'SLFSACPSRYHGICKNNGQCRYAISLRSGTCHCVSGYTGYRCQEIDIRYLLLRY',
    #     'GLFSICPRRYQGICKNNGTCRYALNLRTYTCQCVSGYTGARCQELDIRYLLLRY',
    #     'GLFSICPRRYQGICKNNGTCRYALNLRTYTCQCVSGYTGARCQELDIRYLLLLY',
    #     'SLFSKCPSKFHGICNNNGVCRYAINLRSYTCICLEGYTGDRCQELDIRHLLLRL'
    # ]
    parent_binder_seqs = [
        'SYDGYCLNGCIGYSGDRCQTRDLRWWELR',
        'SYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELR', 
        'SYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELRR', 
        'SYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELRRR',
        'SYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELRRRR',
        'SYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELRRRRR',
        'SYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELRRRRRR',
        'SYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELRRRRRRR',
        'SYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELRRRRRRRR',
        'SYDGYCLNGGVCMHIESLDSYTCNCVIGYSGDRCQTRDLRWWELRRRRRRRRR',
    ]
    parent_binder_seqs = [
        'NLFSNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLLLY',
        'NLFSNCPRRYRGICENNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLLLY',
        'SLFSRCPKRYHGICNNNGQCRYAINLRTYTCICKSGYTGDRCQELDIRYLLLLN',
        'SLFSRCPRRYHGICGNNGQCRYAINLRTYTCRCVSGYTGDRCQELDIRYLLLLN',
        'NLFSNCPRRYRGICNNNGSCQYAINLRTYTCQCSSGYTGARCQELDIRYLLLLY',
        'GLFSRCPKRYHGICGNNGQCRYAINLRTYTCRCVSGYTGPRCQELDIRYLLLLN',
        'NLFSNCPRRYRGICTNNGSCQYAINLRTYTCQCSSGYTGARCQELDIRYLLLLY',
        'NLFSRCPKRYHGICENNGQCRYAINLRTYTCICDSGYTGDRCQELDIRYLLLLN',
        'SLFSRCPRRYHGICHNNGQCRYAINLRTYTCRCVSGYTGDRCQEKDIRYLLLLY',
        'NLFSICPRRYRGICTNNGSCRYAINLRTYTCQCVSGYTGARCQELDIRYLLLLY',
        'SLFSRCPKRYHGICNNNGQCRYAINLRTYTCICVSGYTGDRCQELDIRYLLLLN',
        'GLFSRCPKRYHGICGNNGQCRYAINLRTYTCRCVSGYTGDRCQELDIRYLLLLN',
        'SLFSRCPYRYHGICNNNGQCRYAINLRTYTCICVSGYTGDRCQELDIRYLLLLN',
        'ELFSRCPKRYHGICGNNGQCRYAINLRTYTCRCVSGYTGDRCQELDIRYLLLLN',
        'NLFSRCPKRYHGICGNNGQCRYAIHLRTGTCRCVSGYTGRRCQELDIRYLLLLY',
        'SLFSRCPRRYYGICGNNGLCKYAINLRTGTCRCVSGYTGDRCQELDIRYLLLLN',
        'NLFSKCPRRYYGICGNNGRCKYAINLRTYTCRCVSGYTGQRCQEKDIRYLLLLN',
        'GLFSRCPKRYHGICGNNGQCRYAINLRTYTCRCVSGYTGDRCQELDIRYLLRLN',
        'SLFSKCPSKFHGICNNNGVCRYAINLRSYTCICLEGYTGDRCQEIDIRYLLLQL',
        'SLFSACPYRYHGICKNNGQCRYAISLRSGTCHCVSGYSGYRCHEIDIRYLLLRY'
    ]
    parent_binder_seqs = [
        'AERMRRRFEHIVKIHEEWAKEVLENLKKQGSSEEDLKFMEEYLKQDVEELRERAEKMVEEYRKSS',
        # 'AERMRRRFEHIVEIHEEWAKEVLENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS',
        'SHMVEEALEIVKKLKEAGSDMKKIDELIEKLKEVFEKMPLEDVWPVLYEAEKAAFELMWNAEEGEEDVARAGNYLITKVWDLYFWLREREAARNAAAM',
        'SKEEEYYEEHQKLAKPVEELWEKLDELEKTGKLTGEHRPLVTEFRRLWSDAMVLIAMYMWYLEEVDKNPSEENRKKAQEYLEKVEEKKKEMEELLKKL',
        'SMEKEKLLEIVKELKEAGSDMEKIDELIEKMWEVMKKMELKDIWPVLYELEKVAFELMWNAEEGEEEVARAGNYLITKAWDLYFKAREAEAARNAALM',
        'SSSLVPKAKEEAEDKIIEKIFEVQALSYIHVGETGDRKRHEEIHKKMHEIWEELQKFKKDPSVKTVEEVEKFEKELLEKLEKLKKSL',
        'PVVPEPTMEDFEREFWELYAEYSKAYAEGTEEGLKKTRELRKKIMDVIGRQLEYIWNM',
        'SLDDEITEKLREIVELSDEYALLHWRAKKLSGEEKKEVEEKMKEIKKKIDELWEEVGKLFDESMEKDEAAGKE',
        # 'SLDDEITEKLREIVELSDEYALLHWRAKKLTGKEKEEVLKKMEEIKKRIDELWEEVGKLFDESMAKDEAAGKE',
        'MPVPTPTMEDFEREFWKLYAEYSKAYAEGTEEGLAKTRELRKQIMDVIGRQLEYIWNM',
        'MESIHEIVDKAAYEVIQTFIEKGASDEAKEEQTKIWDEAMAKINAIRAAA',
        'APPLSEIEKEGQKVIAEIEKALTPVPEYHNRLTWENYQKAFVLEMEFFEKFPTEEAATVMSDFWSKFMKP',
        'SMAEQKAEQEAKLAVWEAEMQADTDRMIADVMHRATPENAAEAQEVVDFLRLTRDEIIARRREIIEKKLAE',
        # 'SMEEIRKEQKEKLEVWKKEMEKDTEKMINDVMHRATEENKEEAEEVVNFLRLTREEIIKRREEIIEKKLKE',
        # 'SSLDEWLASLDPQVGQDIRDYIEERQAE',
        'SSLDEWLASLDPQVGADIRAYIEERQAE',
        'SEEKLAEIEAKLKEAAETLNVYEFFEVYSQIVVTTVLTREEYRRVDALSDKYFKKAPKRS',
        # 'SEELLKEIEKKLKEAAKTLDVYEFFEVYSQIVVTTVLTREEYRRVDEISDKYFKEAPERS',
        'MESIHEIVDKTAYEVIQTFIELGASEEAKEKQTKLWDEAMKKINEIRAKE',
        'SEEELREQLREAVRPYIEDISQPMLEAIEALKREGDTELAEVMRKDEEEQIEKWTEMYVDILMKQL',
        # 'TIREHHARQWARYQLEWLRDLVRDENLTDVDPEVLALIESDFTETVPAEELEAVIDELFKAVMDVIDKK',
        # 'MKELKETIEKVRKEVEEMEKKVEEVIKDKSKLEEMMHELFYKASEAQFRVEMAKWRLVKQYGESVKEEVEEAEKVIEEVDEAFRKWWEKLESKM',
        # 'EKEKVPKALEEAEDKIIEKIFEVQALSYIHVGETGDRELHEKIHEEMHEIWEELQKKKKDESIKTVEEVEKFEEELLKKLEELKESL',
        # 'SMIKEEIEKVKKEWEEAKKKGVEGMMWVSFFNADWLLMYLKWKLKKRPELEPELKPLIEELEKLIKEIKEEIESL',
        # 'MKIELSEEERQIVRDYVAKHAPWVSEKAIEAAIDAVLNAENFEEHQKYKEEAVKNGVDEDEFEVGMIVADPAVRWMAHPDPEMRQHYLHVFTP',
        # 'MEELRKVMEEVKEEVKRLKEKVEEVIKDKEKLEEMMHELFYTASENQFRVEMAKWRLKKQYGESVKEEEEEAEKVIKEVDETFREYWEKLEKLM',
        # 'PPVPEPFTEYRRWMRKLYEALDKYKYYEAKGDEKKAAEAKKEFEEVVANPPASMTDENVKAVVEEITAWQDVLGEAYKAAFTEE',
        # 'MLSIDEILERMEEAVIRWAEEHPEKRHEMLKVAYSMRFMLRILYDENNGDLEKTFEEVEKIYAPKSEVAQEVLEVMREAIK',
        # 'SAAEVHHLKRSKEYHLHFSNTVIDHMKKDGHTEDAKEMEAIKEKYLEEQEKKIKEAE',
        # 'MLSIDELLEKMEEAVIKWAEEHPEVRHEALKTMYSIRFMFRIKYDENNGDFEKTFEEIKKIYEPKSEVAKEVLKVIEEAIK',
        # 'SMEEFMKEWAKLQIEWYAEKAKGLPEGKKKHIFEITEQDLIWIARDLPELQEEIKKWVKEAEEK',
        # 'EKEKELEKDYEDLKKVLWTGIEYEVRDEAYIEGRPDLPEEELKKRVEERVEKRVEEIRSMS',
        # 'MRPKEVEELVPMYEKLLEKYKDDPWIVREIQADYYFHEAFIEYQVENGKEEEARRALEMFKKWVKEKYGEEFVE',
        # 'MVPEEVLELRPVYEKLLEKYKDDPWIVREIQADYYFHEAFIEYQYENGKPEEARRALEMFKEWVRERYGEEFVP',
        # 'MKIKLSEEEKQIVRDYVAKHAPWVSEKAIEAAIEAVLNAENFEEHQEFKKKAVEEGVDEDEFEVGMIVADPAVRWMAHPDPEMRQHYLHVFTP',
        # 'MSPIEKEVLKDIEGAKEAIEKAKTTKDWEERFFMVYKMWSRLSYYRNDPEIMESLPEELQKKVEEEYERAYKEIWDISYQHIMDVVENG',
        # 'MLSDREIMLEGLKMALEIFAPYMMQEYIEKIKEVVEEGIELVVKGVSFEEAREKAIEKVRNLPGVDEDTKIHLSFWVDSLLRYVYWYYQHYKR',
        # 'MMSVEEWFERFKDPSADKEKVKAELEEAVKNKELSLNEVYELSVRIHMELYPRHKEYGLTKEEVRELFWFVNDLFFDMLIEGWTFEFK',
        # 'SSEVEKLKEEFEEFLKEMEKYHHGDRETMEEALKRAEELAKRATELFQEAQKNNASDQDIHTLFYISTQMEHYAQWFKHML',
        # 'MSEEQKKVLEYIEGAKKAIEEAKTIPNWEERFFLVYKMWSRLSYVRNNKELMESLPEELQKKVEEEYEKAYKEIWDISYQHIMDVVKNG',
        # 'SLVDEYRAHMTEEEKKIFDRIVELLSKAFGKSKEEVEEFLVSLIPLMDNVGEQDKIAKGLLEKLTSAGALSEDEFWDLYYELQAVLLDVDRRMHRP',
        # 'SMAEVHHLERSKEYHLHFSDTVIEHMRKDGHYEDAEEMEKIKKKYEEEMDKKIEEAK',
        # 'SIQEKYDRLEKLHRDFMWKHHPSPNGPKEVLKMMKSEETKELYKKYYELWQRIENLFWKVLIEGEKVLDEAVPEMEKLIKEASEIKKKVEELYKKELEE',
        # 'SMDELIEKVIEKYNITDEDEIWELRAWATPGFVRFVKEMAHERGKEKFLEEFKKTSRSKVEVMLVEEIVKQVP',
        # 'SKEEEEKLIEFMKEFLKKLAEEEKAKLEAAGDKVTMREVFAIYWYRMFERALEELRKRGSTQYRYVLHKKFEEIVDEMIKEAE',
        # 'RFSEEELELLERLAPEIPEVAEARERLERGEPIPAELLKKIAEALYNLPDSTKFTPEEHRELWRLYSNVVMDWHIAEADTPP',
        # 'RFSEEELELLERLAPEIPEVAEMRRRMERGEPIPPELLKKIADALYNLPDSTKFTPEEHRELWRLYSNVVMDWHIAEADTPK',
        # 'MPMSDEEKAKKMHIMIQHYYYEVQFMAWVLGLSKDSPEYKELMEPLKKMEELWEKFFEKGADTDKIYEEIKKVYEEGMAKANAFWEEKFNPK',
        # 'MTVEEFVEKMMELYEHRYSTPSEIEAFRYHVMGVADIIYHHNNGKVPTIEEVERVLGLPPATPP',
        # 'SAEEEREKQLKLLEEFKKEANELSVEFEKVMVDKHKETGKISQTEYQKKMSEFYDKIFTLHVEFAELLHNGAPEEEVLKFKEEALAELKQMVEDFKKE'
    ]
    parent_binder_seqs = [
        'AERMRRRFEHIVKIHEEWAKEVLENLKKQGSSEEDLKFMEEYLKQDVEELRERAEKMVEEYRKSS',
        'AERMRRRFEHIVEIHEEWAKEVLENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS',
    ]
    parent_binder_seqs = [
        'LFSNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLLLY',
        'FSNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLLLY',
        'SNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLLLY',
        'NCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLLLY',
        'CPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLLLY',
        'PRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLLLY',
        'NLFSNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLLL',
        'NLFSNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLL',
        'NLFSNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLL',
        'NLFSNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYL',
        'NLFSNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRY',
        'LFSNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLLL',
        'FSNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLLL',
        'SNCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYLL',
        'NCPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRYL',
        'CPRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIRY',
        'PRRYRGICTNNGSCQYAINLRTYTCQCLSGYTGARCQELDIR',
    ]
    evolution = DirectedEvolution()
    final_sequences = evolution.run_evolution_cycle.remote(
        parent_binder_seqs=parent_binder_seqs,
        generations=30,
        n_to_fold=50,                # Total sequences to fold per generation
        num_parents=10,               # Number of parents to keep
        top_k=200,                    # Top sequences to consider
        n_parallel_chains=10,        # Parallel chains per sequence
        n_serial_chains=1,           # Sequential runs per sequence
        n_steps=50,                  # Steps per chain
        max_mutations=5,             # Max mutations per sequence
        evoprotgrad_top_fraction=0.25,
        parent_selection_temperature=0.3,
        temp_cycle_period=5,  # Complete cycle every 5 generations
        min_sampling_temp=0.3,  # Minimum temperature value
        max_sampling_temp=2.0,  # Maximum temperature value
        retrain_frequency=2,
        seed=42,
        select_from_current_gen_only=True,  # Add this parameter
    )
    
    print(f"Final evolved sequences: {final_sequences}")
