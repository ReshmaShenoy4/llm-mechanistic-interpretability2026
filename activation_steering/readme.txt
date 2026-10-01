# Running the Python Scripts

## Environment Setup

The required Python packages and their versions are listed in `requirements.txt`.
To install the dependencies:
pip install -r requirements.txt


## Running a Python Script

The scripts can be run from the command line using:

python <script_name>.py [arguments]

For example:
python <script_name>.py \
    --start 0 \
    --end 100 \
    --job_id 1 \
    --output_dir <output_directory> \
    --dtst_path <dataset_path>


## Running on SLURM

A SLURM template is also provided for running the Python scripts on a computing cluster.

Update the required paths, resources, Python script name, and arguments in the SLURM template, then submit it using:

sbatch <slurm_file>.slurm

The SLURM resource configuration may need to be adjusted depending on the cluster and the experiment being run.
