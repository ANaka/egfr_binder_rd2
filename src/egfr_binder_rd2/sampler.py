import modal
from typing import List, Optional
import logging
import pandas as pd
import torch
from datetime import datetime

from egfr_binder_rd2 import LOGGING_CONFIG, MODAL_VOLUME_PATH, OUTPUT_DIRS, ExpertType, ExpertConfig

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
        parent_binder_seq: str,
        generations: int = 5,
        population_size: int = 50,  # This should be passed to sample_sequences
        top_k: int = 10,
        n_chains: int = 4,
        n_steps: int = 250,
        max_mutations: int = 5,
        evoprotgrad_top_fraction: float = 0.2,  # Could be used with sample_from_top_sequences
        parent_selection_temperature: float = 2.0,
        sequence_sampling_temperature: float = 1.0,  # Could be used with sample_from_top_sequences
        retrain_frequency: int = 5,
        seed: int = 42,
    ):
        """Run a single cycle of directed evolution."""
        current_parent_seq = parent_binder_seq
        expert_configs = None
        
        for gen in range(1, generations + 1):
            logger.info(f"=== Generation {gen} ===")
            logger.info(f'Current parent sequence length: {len(current_parent_seq)}')
            logger.info(f"Current parent sequence: {current_parent_seq}")
            
            # Check if retraining is needed
            if self.should_retrain(gen, retrain_frequency):
                logger.info(f"Generation {gen}: Retraining experts...")
                expert_configs = self.retrain_experts()
                logger.info("Expert retraining completed.")

            # Step 1: Generate Variants
            logger.info("Sampling new binder sequences using EvoProtGrad...")
            evoprotgrad_df = self.sample_sequences.remote(
                sequence=current_parent_seq,
                expert_configs=expert_configs,
                n_chains=n_chains,
                n_steps=n_steps,
                max_mutations=max_mutations,
                seed=seed + gen,
            )
            logger.info(f"Generated {len(evoprotgrad_df)} variant sequences")
            
            # Sample population_size sequences from the top fraction using the score column
            sampled_variants = self.sample_from_top_sequences(
                evoprotgrad_df,  # Pass the DataFrame directly, no need to wrap it
                top_fraction=evoprotgrad_top_fraction,
                sample_size=population_size,
                temperature=sequence_sampling_temperature
            )
            logger.info(f"Sampled {len(sampled_variants)} sequences from top {evoprotgrad_top_fraction*100:.1f}% of variants")

            # Continue with sampled variants
            logger.info("Processing sequences to obtain PLL metrics...")
            self.process_sequences.remote(sampled_variants)
            pll_df = self.update_pll_metrics.remote()
            logger.info(f"Obtained PLL metrics for {len(pll_df)} sequences")

            logger.info("Folding binder sequences...")
            self.fold_binder.remote(sampled_variants)
            metrics_df = self.update_metrics.remote()
            logger.info(f"Obtained folding metrics for {len(metrics_df)} sequences")

            # Step 3: Integrate Metrics
            logger.info("Merging metrics and calculating fitness scores...")
            combined_df = pll_df.merge(metrics_df, left_on='sequence', right_on='binder_sequence', how="inner")
            logger.info(f"Successfully merged metrics for {len(combined_df)} sequences")
            combined_df = self.calculate_fitness(combined_df)

            # After getting combined_df, use sample_from_top_sequences:
            selected_sequences = self.sample_from_top_sequences(
                combined_df,
                top_fraction=evoprotgrad_top_fraction,
                sample_size=population_size,
                temperature=sequence_sampling_temperature
            )

            # Log statistics for this generation
            logger.info("Generation Statistics:")
            logger.info(f"Mean PLL: {combined_df['sequence_log_pll'].mean():.2f}")
            logger.info(f"Mean iPAE: {combined_df['pae_interaction'].mean():.2f}")
            logger.info(f"Best PLL: {combined_df['sequence_log_pll'].max():.2f}")
            logger.info(f"Best iPAE: {combined_df['pae_interaction'].min():.2f}")

            # Step 4: Select Top Performers based on fitness score
            logger.info(f"Selecting top {top_k} sequences based on fitness scores...")
            top_sequences = combined_df.sort_values(
                by="fitness_score", ascending=True  # Lower rank is better
            ).head(top_k)

            # Log the top sequences and their metrics
            logger.info("\nTop sequences selected for generation:")
            for idx, row in top_sequences.iterrows():
                logger.info(
                    f"Sequence {idx + 1}: "
                    f"PLL={row['sequence_log_pll']:.2f}, "
                    f"PAE={row['pae_interaction']:.2f}, "
                    f"Fitness={row['fitness_score']:.2f}, "
                    f"Sequence={row['binder_sequence']}"
                )

            # Probabilistically select parent for next generation
            selected_parent = self.select_parent_probabilistically(
                top_sequences,
                temperature=parent_selection_temperature
            )
            current_parent_seq = selected_parent['binder_sequence']
            
            logger.info(f"\nSelected parent for next generation:")
            logger.info(
                f"PLL={selected_parent['sequence_log_pll']:.2f}, "
                f"PAE={selected_parent['pae_interaction']:.2f}, "
                f"Fitness={selected_parent['fitness_score']:.2f}"
            )

        logger.info("Directed evolution cycle completed.")
        return top_sequences['binder_sequence'].tolist()

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
    def sample_from_top_sequences(df, top_fraction=0.2, sample_size=16, temperature=1.0):
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
        scores = -df['fitness_score'].values  # Negative because lower fitness is better
        scores = scores / temperature  # Apply temperature scaling
        probabilities = np.exp(scores - np.max(scores))  # Subtract max for numerical stability
        probabilities = probabilities / probabilities.sum()
        
        # Sample one sequence
        selected_idx = np.random.choice(len(df), p=probabilities)
        return df.iloc[selected_idx]

@app.local_entrypoint()
def main():
    parent_binder_seq = 'AERMRRRFEHIVEIHEEWAKEVLENLKKQGSKEEDLKFMEEYLEQDVEELRKRAEEMVEEYEKSS'
    
    evolution = DirectedEvolution()
    final_sequences = evolution.run_evolution_cycle.remote(
    parent_binder_seq=parent_binder_seq,
        generations=20,
        population_size=16,
        top_k=50,
        n_chains=4,
        n_steps=250,
        max_mutations=5,
        evoprotgrad_top_fraction=0.2,
        parent_selection_temperature=2.0,
        sequence_sampling_temperature=1.0,
        retrain_frequency=2,
        seed=42,
    )
    
    print(f"Final evolved sequences: {final_sequences}")
