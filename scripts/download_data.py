"""Download the Credit Card Fraud Detection dataset from Kaggle."""

import hashlib
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

# Expected SHA256 hash of the creditcard.csv file
EXPECTED_HASH = None  # Will be computed on first download

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "creditcard.csv"


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Show download progress."""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size)
    bar_length = 50
    filled = int(bar_length * percent / 100)
    bar = "=" * filled + "-" * (bar_length - filled)
    sys.stdout.write(f"\rDownloading: [{bar}] {percent:.1f}%")
    sys.stdout.flush()


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def download_via_kaggle() -> bool:
    """Download using Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        print("Using Kaggle API to download dataset...")
        api = KaggleApi()
        api.authenticate()
        
        # Download and extract
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=DATA_DIR,
            unzip=True
        )
        print("\n✓ Downloaded via Kaggle API")
        return True
        
    except ImportError:
        print("Kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Kaggle API error: {e}")
        return False


def download_via_url() -> bool:
    """
    Download from a mirror URL.
    
    Note: The official dataset is on Kaggle and requires authentication.
    This function provides an alternative if Kaggle API is not available.
    """
    # Alternative: OpenML mirror
    OPENML_URL = "https://www.openml.org/data/get_csv/1673544/phpKo8OWT"
    
    print("Attempting download from OpenML mirror...")
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        temp_file = DATA_DIR / "creditcard_temp.csv"
        
        urlretrieve(OPENML_URL, temp_file, reporthook=download_progress)
        print()  # New line after progress bar
        
        # Rename to final location
        temp_file.rename(OUTPUT_FILE)
        print("✓ Downloaded from OpenML mirror")
        return True
        
    except Exception as e:
        print(f"Mirror download failed: {e}")
        if temp_file.exists():
            temp_file.unlink()
        return False


def verify_download() -> bool:
    """Verify the downloaded file."""
    if not OUTPUT_FILE.exists():
        print("✗ File not found")
        return False
    
    # Check file size (should be ~150MB)
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    if size_mb < 100:
        print(f"✗ File too small ({size_mb:.1f} MB), expected ~150 MB")
        return False
    
    # Verify it's a valid CSV with expected columns
    with open(OUTPUT_FILE, 'r') as f:
        header = f.readline().strip()
        expected_cols = ['Time', 'V1', 'V28', 'Amount', 'Class']
        if not all(col in header for col in expected_cols):
            print("✗ Invalid CSV format - missing expected columns")
            return False
    
    print(f"✓ File verified ({size_mb:.1f} MB)")
    return True


def main() -> int:
    """Main download function."""
    print("=" * 60)
    print("Credit Card Fraud Detection Dataset Downloader")
    print("=" * 60)
    print(f"Target: {OUTPUT_FILE}")
    print()
    
    # Check if already exists
    if OUTPUT_FILE.exists():
        print("File already exists.")
        if verify_download():
            print("\nDataset is ready to use!")
            return 0
        else:
            print("Existing file is invalid, re-downloading...")
            OUTPUT_FILE.unlink()
    
    # Create directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try Kaggle first
    if download_via_kaggle():
        if verify_download():
            print("\n✓ Dataset downloaded and verified successfully!")
            return 0
    
    # Fall back to URL download
    print("\nTrying alternative download method...")
    if download_via_url():
        if verify_download():
            print("\n✓ Dataset downloaded and verified successfully!")
            return 0
    
    print("\n" + "=" * 60)
    print("DOWNLOAD FAILED")
    print("=" * 60)
    print("\nPlease download manually from Kaggle:")
    print("  1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  2. Download creditcard.csv")
    print(f"  3. Place in: {DATA_DIR}/")
    print("\nOr set up Kaggle API:")
    print("  1. pip install kaggle")
    print("  2. Create API token at: https://www.kaggle.com/settings")
    print("  3. Place kaggle.json in ~/.kaggle/")
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
