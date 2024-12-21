import modal
from pathlib import Path
import logging
from egfr_binder_rd2 import MODAL_VOLUME_NAME, MODAL_VOLUME_PATH, OUTPUT_DIRS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_files_to_copy(dir_path: Path) -> list[Path]:
        """Recursively get all files excluding PNGs"""
        files = [
            f for f in dir_path.rglob("*") 
            if f.is_file() and f.suffix != '.png'
        ]
        logger.info(f"Found {len(files)} files in {dir_path}")
        if len(files) > 0:
            logger.info(f"Example files: {files[:5]}")
        return files

# Initialize Modal app
app = modal.App("egfr-binders")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

@app.function(
    volumes={MODAL_VOLUME_PATH: volume},
    timeout=3600,
    # mounts=[modal.Mount.from_local_dir(
    #     "/home/naka/code/egfr_binder_rd2/data/colabfold_outputs/high_quality_folded",
    #     remote_path=f"{MODAL_VOLUME_PATH}/folded_high_quality"
    # )]
)
def populate_volume():
    source_dir = Path("/data/high_quality_folded")
    dest_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded_high_quality"]
    
    # Debug path existence
    logger.info(f"Source directory exists: {source_dir.exists()}")
    logger.info(f"Source directory contents: {list(source_dir.glob('*')) if source_dir.exists() else 'N/A'}")
    
    # Create destination directory
    dest_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Created destination directory: {dest_dir}")
    
    
    
    # Get list of files to copy
    files = get_files_to_copy(source_dir)
    logger.info(f"Found {len(files)} files to copy")
    
    # Copy files in batches
    BATCH_SIZE = 100
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(files)-1)//BATCH_SIZE + 1}")
        
        for file in batch:
            try:
                relative_path = file.relative_to(source_dir)
                dest_path = dest_dir / relative_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file and preserve metadata
                with open(file, 'rb') as src, open(dest_path, 'wb') as dst:
                    dst.write(src.read())
                logger.info(f"Copied {relative_path}")
                
            except Exception as e:
                logger.error(f"Failed to copy {file}: {str(e)}")
        
        # Commit after each batch
        volume.commit()
        logger.info(f"Committed batch {i//BATCH_SIZE + 1}")
    
    logger.info(f"Successfully copied {len(files)} files")

@app.function(
    volumes={MODAL_VOLUME_PATH: volume},
    timeout=3600,
)
def download_from_volume():
    # Define source and destination directory pairs
    dir_pairs = [
        (OUTPUT_DIRS["folded"], "folded"),
        (OUTPUT_DIRS["msa_results"], "msa_results"),
        (OUTPUT_DIRS["folded_high_quality"], "high_quality_folded"),
    ]
    
    for volume_subdir, local_subdir in dir_pairs:
        source_dir = Path(MODAL_VOLUME_PATH) / volume_subdir
        dest_dir = Path("./home/naka/code/egfr_binder_rd2/data/colabfold_outputs") / local_subdir
        
        # Debug path existence
        logger.info(f"\nProcessing directory: {volume_subdir}")
        logger.info(f"Source directory exists: {source_dir.exists()}")
        logger.info(f"Source directory contents: {list(source_dir.glob('*')) if source_dir.exists() else 'N/A'}")
        
        # Create destination directory locally
        dest_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created destination directory: {dest_dir}")
        
        # Get list of files to copy
        files = get_files_to_copy(source_dir)
        logger.info(f"Found {len(files)} files to copy")
        
        # Copy files in batches
        BATCH_SIZE = 100
        for i in range(0, len(files), BATCH_SIZE):
            batch = files[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(files)-1)//BATCH_SIZE + 1}")
            
            for file in batch:
                try:
                    relative_path = file.relative_to(source_dir)
                    dest_path = dest_dir / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file and preserve metadata
                    with open(file, 'rb') as src, open(dest_path, 'wb') as dst:
                        dst.write(src.read())
                    logger.info(f"Copied {relative_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to copy {file}: {str(e)}")
            
            logger.info(f"Completed batch {i//BATCH_SIZE + 1}")

@app.local_entrypoint()
def main(download: bool = False):
    if download:
        logger.info("Starting volume download...")
        download_from_volume.remote()
    else:
        logger.info("Starting volume population...")
        populate_volume.remote()
    logger.info("Done!")

if __name__ == "__main__":
    main()