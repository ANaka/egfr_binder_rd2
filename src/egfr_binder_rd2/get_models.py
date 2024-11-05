import modal
from pathlib import Path
import logging
from egfr_binder_rd2 import MODAL_VOLUME_NAME, MODAL_VOLUME_PATH, OUTPUT_DIRS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Modal app
app = modal.App("egfr-binders")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

@app.function(
    volumes={MODAL_VOLUME_PATH: volume},
)
def list_latest_models():
    """List the latest version of each model type."""
    source_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["bt_models"]
    
    if not source_dir.exists():
        logger.error(f"Source directory {source_dir} does not exist!")
        return
    
    # Model types we're interested in
    expert_types = [
        "pae_interaction",  # iPAE
        "i_ptm",           # iPTM
        "binder_plddt",    # pLDDT
        "p_soluble",       # p_soluble
        "sequence_log_pll" # PLL
    ]
    
    print("\nTo download the latest models, run these commands:")
    print("----------------------------------------")
    for expert_type in expert_types:
        # Look for timestamped models first
        pattern = f"ensemble_{expert_type}_standardize_*.pt"
        model_files = list(source_dir.glob(pattern))
        
        if model_files:
            # Get the latest model by timestamp
            latest_model = sorted(model_files, key=lambda x: x.stem.split('_')[-1], reverse=True)[0]
        else:
            # Try old naming format
            latest_model = source_dir / f"ensemble_{expert_type}_standardize.pt"
            if not latest_model.exists():
                logger.warning(f"No model found for {expert_type}")
                continue
        
        # Use relative path from MODAL_VOLUME_PATH
        relative_path = latest_model.relative_to(Path(MODAL_VOLUME_PATH))
        print(f"modal volume get {MODAL_VOLUME_NAME} {relative_path} /home/naka/code/egfr_binder_rd2/data/models/{latest_model.name}")
    print("----------------------------------------")

@app.local_entrypoint()
def main():
    list_latest_models.remote()

if __name__ == "__main__":
    main()