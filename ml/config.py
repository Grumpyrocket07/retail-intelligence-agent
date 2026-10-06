"""Settings for the churn dataset. Change them here, nowhere else."""

# A customer counts as churned if they place no order in the WINDOW_DAYS after the snapshot date.
WINDOW_DAYS = 90

# Each snapshot date produces one row per eligible customer.
# Train on earlier dates, validate on a later one, test on the latest: the model is always judged on the future.
# The last snapshot must be at least WINDOW_DAYS before the data ends (9 Dec 2011).
SNAPSHOTS = {
    "train": ["2010-09-09", "2011-03-09"],
    "validation": ["2011-06-09"],
    "test": ["2011-09-09"],
}

# Who is eligible at a snapshot date?
#   None -> every customer who has ordered at least once before it
#   365  -> only customers who ordered in the 365 days before it (drops long-dormant customers). CHOSEN.
ACTIVE_WITHIN_DAYS = 365

# Reported alongside the chosen definition so the two can be compared.
COMPARISON_ACTIVE_WITHIN_DAYS = 365
