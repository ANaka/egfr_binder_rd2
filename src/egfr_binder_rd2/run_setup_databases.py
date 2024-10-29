import modal
import subprocess
from egfr_binder_rd2 import MODAL_VOLUME_NAME

# Initialize Modal App
app = modal.App("setup-databases-app")

# Define a persistent volume
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

# Specify the Docker image with necessary dependencies
image = (
    modal.Image.debian_slim()
    .apt_install('wget')
    .apt_install('tar', 'rsync', 'curl', "awscli")
    .apt_install("aria2", )
    .apt_install("mmseqs2")
)

SCRIPT_CONTAINER_PATH = "/root/setup_databases.sh"

@app.function(
    volumes={"/mnt": volume},
    timeout=60 * 60 * 24,
    image=image,
    mounts=[modal.Mount.from_local_file("setup_databases.sh", SCRIPT_CONTAINER_PATH)]
)
def run_setup():
    """
    Executes the setup_databases.sh script within the mounted volume.
    """
    workdir = "/mnt"
    
    # Use the known container path
    subprocess.run(["chmod", "+x", SCRIPT_CONTAINER_PATH], check=True)
    subprocess.run([SCRIPT_CONTAINER_PATH, workdir], check=True)

if __name__ == "__main__":
    # Deploy the Modal app
    app.deploy()