"""

Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Useful functions for logging.
"""

def make_csv_if_not_exists(csv_path, header):
    """
    Create a CSV file if it does not exist, and write the header.
    """
    import os
    import csv

    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def write_csv_row(csv_path, row):
    """
    Write a row to a CSV file.
    """
    import csv

    with open(csv_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)
