# DiffuseMix
 DiffuseMix for Data Augmentation

**Before Running**:
- Install the required dependencies: pip install -r requirements.txt
- Change the file paths in runner.py: github_project_path and output_dir

# SAM2
Windows:
- mkdir checkpoints && curl -o checkpoints/sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
Linux/Mac:
- wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt