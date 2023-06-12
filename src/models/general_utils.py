"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Defines general utility functions that are relevant for all models.
"""

# ============= Import Libraries =============
import torch


# ============= General Utility Functions =============
def get_exp_save_path(base_fd):
    """
    Estimate save path for current experiment run, so that it doesn't overwrite previous runs.

    Parameters:
        - exp_fd: Folder directory for saving experiments

    Outputs:
        - save_fd: Folder directory for saving current experiment run.
    """
    # Add separator if it does not exist
    if exp_fd[-1] != "/":
        exp_fd += "/"

    # If no current run exists
    if not os.path.exists(exp_fd):

        # Make Save_fd as run 1
        save_fd = exp_fd + f"run1-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"

        # Make folder and add logs
        os.makedirs(save_fd)
        os.makedirs(save_fd + "logs/")

        # return experiment directory
        return save_fd

    # Else find first run that has not been previously computed - get list of runs and add 1 to max run
    list_run_dirs = [fd.name for fd in os.scandir(exp_fd) if fd.is_dir()]
    list_runs = [int(run.split("-")[0][3:]) for run in list_run_dirs]
    new_run_num = max(list_runs) + 1

    # Save as new run
    save_fd = exp_fd + f"run{new_run_num}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
    assert not os.path.exists(save_fd)

    os.makedirs(save_fd)
    os.makedirs(save_fd + "logs/")

    # Return experiment directory
    return save_fd

