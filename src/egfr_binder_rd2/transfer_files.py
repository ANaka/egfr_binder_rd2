import shutil
from pathlib import Path
import logging
from typing import List, Tuple
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def transfer_files(
    source_dir: str | Path = "/home/naka/code/egfr_binder_rd2/data/colabfold_outputs/bindcraft",
    folded_dir: str | Path = "/home/naka/code/egfr_binder_rd2/data/colabfold_outputs/folded",
    msa_dir: str | Path = "/home/naka/code/egfr_binder_rd2/data/colabfold_outputs/msa_results"
) -> Tuple[List[Path], List[Path]]:
    """
    Transfer files from bindcraft directory to appropriate folders.
    
    Args:
        source_dir: Source directory containing bindcraft output files
        folded_dir: Destination directory for folded structure files (PDBs and JSONs)
        msa_dir: Destination directory for MSA files (a3m)
        
    Returns:
        Tuple of (transferred_structure_files, transferred_msa_files)
    """
    # Convert to Path objects
    source_dir = Path(source_dir)
    folded_dir = Path(folded_dir)
    msa_dir = Path(msa_dir)
    
    # Create destination directories if they don't exist
    folded_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    
    # Track transferred files
    transferred_structure_files = []
    transferred_msa_files = []
    
    # Process all files in source directory
    for file_path in source_dir.glob("*"):
        try:
            if file_path.suffix == '.a3m':
                # Transfer MSA files
                dest_path = msa_dir / file_path.name
                if not dest_path.exists():
                    shutil.copy2(file_path, dest_path)
                    transferred_msa_files.append(dest_path)
                    logger.info(f"Transferred MSA file: {file_path.name}")
            
            elif file_path.suffix in ['.pdb', '.json']:
                # Transfer structure files
                dest_path = folded_dir / file_path.name
                if not dest_path.exists():
                    shutil.copy2(file_path, dest_path)
                    transferred_structure_files.append(dest_path)
                    logger.info(f"Transferred structure file: {file_path.name}")
        
        except Exception as e:
            logger.error(f"Error transferring {file_path}: {str(e)}")
    
    logger.info(f"Transferred {len(transferred_structure_files)} structure files and {len(transferred_msa_files)} MSA files")
    return transferred_structure_files, transferred_msa_files

if __name__ == "__main__":
    transfer_files()


# # Download structure files
# modal volume get egfr_binders folded /home/naka/code/egfr_binder_rd2/data/colabfold_outputs/

# # Download MSA files
# modal volume get egfr_binders msa_results /home/naka/code/egfr_binder_rd2/data/colabfold_outputs/

# # Upload structure files
# modal volume put egfr_binders /home/naka/code/egfr_binder_rd2/data/colabfold_outputs/folded/  folded

# # Upload MSA files
# modal volume put egfr_binders /home/naka/code/egfr_binder_rd2/data/colabfold_outputs/msa_results/  msa_results