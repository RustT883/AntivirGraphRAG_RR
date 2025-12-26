#!/usr/bin/env python3
"""Minimal checksum generator for GLIREL repository."""
import hashlib
import json
from pathlib import Path
from datetime import datetime

# Your Zenodo model checksums - REPLACE THESE
ZENODO_CHECKSUMS = {
    "all_texts_for_drugs_processed.csv": "dad3b7d672c30f8130d7b6c10ec7b0a6d8feaaed30667978f95d4c82ac9e3af8",
    "NER_Model.tar.gz": "eefdc0684707f1ee724210ab8e44b00bf768ef30dec4026e30f0b90af56aea5a",
    "Drugprot_REL_model.tar.gz": "f1bb2b36daf466a623b67ad73e8718e954f104d8bad72af53c21c028a10c9b43"
}

def sha256_file(filepath):
    """Get SHA256 of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def generate():
    """Generate checksums for all files."""
    checksums = {}
    
    # Skip these
    skip = {'.git', '__pycache__', '.DS_Store', 'Thumbs.db', '*.pyc'}
    
    for path in Path('.').rglob('*'):
        if path.is_file():
            # Skip ignored paths
            if any(pattern in str(path) for pattern in skip):
                continue
            
            # Skip checksum files themselves
            if path.name in ['checksums.json', 'CHECKSUMS.md']:
                continue
            
            checksums[str(path)] = sha256_file(path)
    
    # Save JSON
    with open('checksums.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'zenodo_models': ZENODO_CHECKSUMS,
            'repository_files': checksums
        }, f, indent=2)
    
    # Save human-readable
    with open('CHECKSUMS.md', 'w') as f:
        f.write(f"# Checksums - Generated {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write("## Zenodo Models\n```\n")
        for model, h in ZENODO_CHECKSUMS.items():
            f.write(f"{model}: {h}\n")
        f.write("```\n\n## Repository Files\n```\n")
        for file, h in sorted(checksums.items()):
            f.write(f"{h}  {file}\n")
        f.write("```\n")
    
    print(f"Generated checksums for {len(checksums)} files")
    print(f"Included {len(ZENODO_CHECKSUMS)} Zenodo models")

if __name__ == '__main__':
    generate()
