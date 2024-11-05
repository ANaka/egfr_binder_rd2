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
    timeout=3600,
    mounts=[modal.Mount.from_local_dir(
        "/home/naka/code/egfr_binder_rd2/data/colabfold_outputs/high_quality_folded",
        remote_path="/data/high_quality_folded"
    )]
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


# # ... existing code ...

# @app.function(
#     image=image,
#     timeout=3600,
#     volumes={MODAL_VOLUME_PATH: volume},
#     mounts=[modal.Mount.from_local_dir(
#         "data/retrieved_structures",  # Local directory to store retrieved files
#         remote_path="/data/retrieved"
#     )]
# )
# def retrieve_structure_files(seq_hash: str, output_dir: str = "data/retrieved_structures"):
#     """Retrieve structure files from Modal volume for a given sequence hash.
    
#     Args:
#         seq_hash: Hash of the sequence to retrieve files for
#         output_dir: Local directory to store retrieved files
#     """
#     # Set up paths
#     volume_folded_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded"]
#     volume_high_quality_dir = Path(MODAL_VOLUME_PATH) / OUTPUT_DIRS["folded_high_quality"]
#     local_output_dir = Path("/data/retrieved")  # This maps to output_dir locally
    
#     # Create output directory
#     local_output_dir.mkdir(parents=True, exist_ok=True)
    
#     def copy_matching_files(source_dir: Path, pattern: str):
#         """Copy files matching pattern from source directory."""
#         matching_files = list(source_dir.glob(f"{seq_hash}*{pattern}"))
#         for file in matching_files:
#             dest_path = local_output_dir / file.name
#             logger.info(f"Copying {file.name}")
#             with open(file, 'rb') as src, open(dest_path, 'wb') as dst:
#                 dst.write(src.read())
#         return len(matching_files)

#     # Look for files in both regular and high-quality folders
#     total_files = 0
#     for source_dir in [volume_folded_dir, volume_high_quality_dir]:
#         if source_dir.exists():
#             # Copy PDB files
#             total_files += copy_matching_files(source_dir, ".pdb")
#             # Copy score files
#             total_files += copy_matching_files(source_dir, "_scores*.json")
#             # Copy PAE files
#             total_files += copy_matching_files(source_dir, "_predicted_aligned_error*.json")
    
#     if total_files == 0:
#         logger.warning(f"No files found for sequence hash {seq_hash}")
#     else:
#         logger.info(f"Retrieved {total_files} files for sequence hash {seq_hash}")

# @app.local_entrypoint()
# def test_retrieve():
#     """Test the structure retrieval functionality."""
#     test_hash = "example_hash"  # Replace with an actual hash
#     retrieve_structure_files.remote(test_hash)

@app.local_entrypoint()
def main():
    logger.info("Starting volume population...")
    populate_volume.remote()
    logger.info("Done!")

if __name__ == "__main__":
    main()