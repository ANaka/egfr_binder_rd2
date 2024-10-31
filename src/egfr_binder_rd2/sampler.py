import modal
from typing import List, Optional
import logging
import pandas as pd
import torch
from datetime import datetime

from egfr_binder_rd2 import LOGGING_CONFIG, MODAL_VOLUME_PATH, OUTPUT_DIRS, ExpertType, ExpertConfig, EvolutionMetadata

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
        self.train_bt_model = modal.Function.lookup("bt-training", "train_bt_model")
        self.process_sequences = modal.Function.lookup("esm2-inference", "process_sequences")
        self.update_pll_metrics = modal.Function.lookup("esm2-inference", "update_pll_metrics")
        self.fold_binder = modal.Function.lookup("simplefold", "fold_binder")
        self.update_metrics = modal.Function.lookup("simplefold", "update_metrics_for_all_folded")
        
        logger.info("Initialized DirectedEvolution workflow.")
        return self

    @modal.method()
    def retrain_experts(self):
        """Retrain the Bradley-Terry experts using accumulated data."""
        logger.info("Retraining experts...")
        
        # Train PAE interaction expert
        pae_model_path = self.train_bt_model.remote(
            yvar="pae_interaction",
            wandb_project="egfr-binder-rd2",
            wandb_entity="anaka",
            transform_type="rank",
            make_negative=True,
        )
        logger.info(f"Trained PAE interaction expert: {pae_model_path}")
        
        # Return updated expert configs
        return [
            ExpertConfig(
                type=ExpertType.ESM,  # This is correct
                weight=1.0,
                temperature=1.0,
            ),
            ExpertConfig(
                type=ExpertType.iPAE,  # Changed from PAE to iPAE
                weight=1.0,
                temperature=1.0,
                make_negative=True,
                transform_type="rank",
            ),
        ]

    @staticmethod
    def should_retrain(generation: int, retrain_frequency: int = 5) -> bool:
        return generation > 0 and generation % retrain_frequency == 0

    @modal.method()
    def run_evolution_cycle(
        self,
        parent_binder_seqs: List[str],      # List of initial parent sequences to evolve from
        generations: int = 5,                # Number of evolution cycles to run
        n_to_fold: int = 50,                # Total number of sequences to fold per generation
        num_parents: int = 5,               # Number of parent sequences to keep for next generation
        top_k: int = 10,                    # Number of top sequences to consider for parent selection
        n_parallel_chains: int = 32,         # Number of parallel MCMC chains per sequence
        n_serial_chains: int = 1,           # Number of sequential runs per sequence
        n_steps: int = 250,                 # Number of steps per chain
        max_mutations: int = -1,             # Maximum number of mutations allowed per sequence
        evoprotgrad_top_fraction: float = 0.2,    # Fraction of top sequences to consider from EvoProtGrad output
        parent_selection_temperature: float = 2.0, # Temperature for parent selection (higher = more diversity)
        sequence_sampling_temperature: float = 2.0,# Temperature for sequence sampling (higher = more diversity)
        retrain_frequency: int = 5,         # How often to retrain experts (every N generations)
        seed: int = 42,                     # Random seed for reproducibility
        select_from_current_gen_only: bool = False,  # New parameter
    ):
        """Run multiple cycles of directed evolution with multiple parent sequences.
        
        Args:
            parent_binder_seqs: List of initial parent sequences to evolve from
            generations: Number of evolution cycles to run
            n_to_fold: Total number of sequences to fold per generation
            num_parents: Number of parent sequences to retain for next generation
            top_k: Number of top sequences to consider for parent selection
            n_parallel_chains: Number of parallel MCMC chains per sequence
            n_serial_chains: Number of sequential runs per sequence
            n_steps: Number of steps per chain
            max_mutations: Maximum number of mutations allowed per sequence
            evoprotgrad_top_fraction: Fraction of top sequences to consider from EvoProtGrad output
            parent_selection_temperature: Temperature parameter for parent selection (higher = more diversity)
            sequence_sampling_temperature: Temperature parameter for sequence sampling (higher = more diversity)
            retrain_frequency: How often to retrain experts (every N generations)
            seed: Random seed for reproducibility
            select_from_current_gen_only: If True, select top_k sequences only from current generation.
                If False, select from all sequences ever evaluated.
        """
        current_parent_seqs = parent_binder_seqs.copy()
        expert_configs = None
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
            "sequence_sampling_temperature": sequence_sampling_temperature,
            "retrain_frequency": retrain_frequency,
            "seed": seed
        }
        metadata = EvolutionMetadata.create(config, parent_binder_seqs)
        
        for gen in range(1, generations + 1):
            logger.info(f"=== Generation {gen} ===")
            logger.info(f"Number of parent sequences: {len(current_parent_seqs)}")
            
            # Check if retraining is needed
            if self.should_retrain(gen, retrain_frequency):
                logger.info(f"Generation {gen}: Retraining experts...")
                expert_configs = self.retrain_experts()
                logger.info("Expert retraining completed.")
            
            # Calculate sequences to sample per parent
            seqs_per_parent = max(1, n_to_fold // len(current_parent_seqs))
            logger.info(f"Sampling {seqs_per_parent} sequences per parent to reach total of ~{n_to_fold}")
            
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
            )
            logger.info(f"Generated {len(evoprotgrad_df)} variant sequences")
            
            # Sample sequences from the top fraction, now considering parent information
            all_variants_with_parents = []  # New list to track variants with their parents
            for parent_idx, parent_seq in enumerate(current_parent_seqs):
                parent_variants = evoprotgrad_df[evoprotgrad_df['parent_seq'] == parent_seq]
                if len(parent_variants) > 0:
                    sampled_variants = self.sample_from_evoprotgrad_sequences(
                        parent_variants,
                        top_fraction=evoprotgrad_top_fraction,
                        sample_size=seqs_per_parent,
                        temperature=sequence_sampling_temperature
                    )
                    # Store variants with their parent information
                    all_variants_with_parents.extend([(variant, parent_seq) for variant in sampled_variants])
                    all_variants.extend(sampled_variants)
                else:
                    logger.warning(f"No variants generated for parent sequence {parent_idx + 1}")
            
            # Process all variants together
            logger.info(f"\nProcessing {len(all_variants)} total variants...")
            
            # Extract sequences and parents for folding
            variant_seqs = [v[0] for v in all_variants_with_parents]
            parent_seqs = [v[1] for v in all_variants_with_parents]
            
            # Get PLL metrics
            logger.info("Processing sequences to obtain PLL metrics...")
            self.process_sequences.remote(all_variants)
            pll_df = self.update_pll_metrics.remote()
            logger.info(f"Obtained PLL metrics for {len(pll_df)} sequences")
            
            # Get folding metrics, now passing parent sequences
            logger.info("Folding binder sequences...")
            self.fold_binder.remote(variant_seqs, parent_seqs)
            metrics_df = self.update_metrics.remote()
            logger.info(f"Obtained folding metrics for {len(metrics_df)} sequences")
            
            # Merge and calculate fitness
            logger.info("Merging metrics and calculating fitness scores...")
            combined_df = pll_df.merge(metrics_df, left_on='sequence', right_on='binder_sequence', how="inner")
            logger.info(f"Successfully merged metrics for {len(combined_df)} sequences")
            
            # Filter for current generation if requested
            if select_from_current_gen_only:
                logger.info(f"Calculating fitness using only current generation sequences")
                ranking_df = combined_df[combined_df['binder_sequence'].isin(all_variants)]
                logger.info(f"Found {len(ranking_df)} sequences from current generation")
            else:
                logger.info("Calculating fitness using all historical sequences")
                ranking_df = combined_df
                logger.info(f"Using all {len(ranking_df)} historical sequences")

            if len(ranking_df) == 0:
                logger.warning("No sequences found for ranking. Using parent sequences for next generation.")
                return current_parent_seqs
            
            # Calculate ranks and fitness
            ranking_df['pae_interaction_rank'] = 1 - ranking_df['pae_interaction'].rank(pct=True)
            ranking_df['i_ptm_rank'] = ranking_df['i_ptm'].rank(pct=True)
            ranking_df['sequence_log_pll_rank'] = ranking_df['sequence_log_pll'].rank(pct=True)
            ranking_df['fitness'] = (
                ranking_df['pae_interaction_rank'] + 
                ranking_df['i_ptm_rank'] + 
                ranking_df['sequence_log_pll_rank']
            ) / 3
            
            # Sort by fitness
            ranking_df = ranking_df.sort_values('fitness', ascending=False).reset_index(drop=True)
            
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
                    f"Sequence {idx + 1}: "
                    f"Fitness={row['fitness']:.3f} "
                    f"(PLL_rank={row['sequence_log_pll_rank']:.3f}, "
                    f"iPAE_rank={row['pae_interaction_rank']:.3f}, "
                    f"iPTM_rank={row['i_ptm_rank']:.3f}), "
                    f"Raw: PLL={row['sequence_log_pll']:.2f}, "
                    f"iPAE={row['pae_interaction']:.2f}, "
                    f"iPTM={row['i_ptm']:.2f}, "
                    f"Sequence={row['binder_sequence']}"
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
                    f"Parent {idx}: "
                    f"PLL={parent_metrics['sequence_log_pll']:.2f}, "
                    f"PAE={parent_metrics['pae_interaction']:.2f}, "
                    f"Fitness={parent_metrics['fitness_score']:.2f}"
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
        
        logger.info("Directed evolution cycle completed.")
        # Save metadata
        metadata.save(OUTPUT_DIRS["evolution_trajectories"])
        return list(set(all_final_sequences))  # Remove duplicates

    @staticmethod
    def calculate_fitness(df):
        """
        Calculate fitness scores based on ranked metrics.
        
        Args:
            df (pd.DataFrame): DataFrame containing sequence_log_pll and pae_interaction
            
        Returns:
            pd.DataFrame: DataFrame with added fitness_score column
        """
        # Rank sequences by sequence_log_pll (higher is better)
        pll_ranks = df['sequence_log_pll'].rank(ascending=False)
        
        # Rank sequences by pae_interaction (lower is better)
        pae_ranks = df['pae_interaction'].rank(ascending=True)
        
        # Calculate average rank (lower is better)
        df['fitness_score'] = (pll_ranks + pae_ranks) / 2
        
        return df

    @staticmethod
    def sample_from_evoprotgrad_sequences(df, top_fraction=0.25, sample_size=10, temperature=1.0):
        """
        Sample sequences from the top fraction based on scores.
        
        Args:
            df (pd.DataFrame): DataFrame with sequences and optionally scores
            top_fraction (float): Fraction of top sequences to consider (0-1)
            sample_size (int): Number of sequences to sample
            temperature (float): Temperature for softmax; higher values = more diversity
            
        Returns:
            list[str]: Sampled sequences
        """
        # Calculate number of sequences to keep
        n_keep = max(int(len(df) * top_fraction), sample_size)
        
        if 'score' in df.columns:
            # If we have scores, use them for weighted sampling
            top_df = df.nlargest(n_keep, 'score') 
            scores = top_df['score'].values
        else:
            # If no scores, just randomly sample from top fraction
            top_df = df.sample(n=n_keep)
            scores = np.ones(len(top_df))  # Equal probabilities
        
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

@app.local_entrypoint()
def main():
    # Define multiple parent sequences
    parent_binder_seqs = [
       'AERMRRRFESIVEIHEEWAKEVLENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS'
    ] * 5
    
    evolution = DirectedEvolution()
    final_sequences = evolution.run_evolution_cycle.remote(
        parent_binder_seqs=parent_binder_seqs,
        generations=4,
        n_to_fold=10,                # Total sequences to fold per generation
        num_parents=10,               # Number of parents to keep
        top_k=50,                    # Top sequences to consider
        n_parallel_chains=8,        # Parallel chains per sequence
        n_serial_chains=1,           # Sequential runs per sequence
        n_steps=50,                  # Steps per chain
        max_mutations=-1,             # Max mutations per sequence
        evoprotgrad_top_fraction=0.2,
        parent_selection_temperature=2.0,
        sequence_sampling_temperature=2.0,
        retrain_frequency=2,
        seed=42,
        select_from_current_gen_only=True,  # Add this parameter
    )
    
    print(f"Final evolved sequences: {final_sequences}")
