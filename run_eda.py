import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EDA_NOTEBOOKS = [
    "01_revenue_and_profitability_eda.ipynb",
    "02_customer_segmentation_eda.ipynb",
    "03_product_analysis_eda.ipynb",
    "04_marketing_and_channels_eda.ipynb",
    "05_operations_and_inventory_eda.ipynb",
    "06_data_quality_audit.ipynb",
    "07a_first_experience_trap.ipynb",
    "07b_review_decay.ipynb",
    "07c_toxic_customer_profile.ipynb",
    "08a_advanced_diagnostics.ipynb",
    "08b_granger_dilution.ipynb",
    "08c_anomaly_geo.ipynb"
]

if __name__ == "__main__":
    print("Executing Exploratory Data Analysis (EDA) pipeline...")
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
    
    for nb_name in EDA_NOTEBOOKS:
        nb_path = os.path.join(base_dir, nb_name)
        if not os.path.exists(nb_path):
            print(f"Skipping {nb_name} (not found)")
            continue
            
        try:
            print(f"Running {nb_name}...")
            # Execute the notebook headlessly
            subprocess.run(
                ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", nb_path],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error executing {nb_name}")
            sys.exit(1)
            
    print("EDA pipeline completed.")
