# build_all.py
import subprocess
import sys
from src.utils import logger


def run_cmd(cmd):
    logger.info("Running: %s", cmd)
    subprocess.run([sys.executable, "-m"] + cmd.split(), check=True)


def main():
    # 1) build index
    run_cmd("src.build_index")
    logger.info("Index built")


if __name__ == "__main__":
    main()
