import os
import sys
import subprocess
import time

def run_notebook(notebook_path):
    """Execute a Jupyter notebook and save the output."""
    start_time = time.time()
    print(f"Running {os.path.basename(notebook_path)}...")
    
    try:
        # Use nbconvert to execute the notebook
        subprocess.run(
            [
                "jupyter", "nbconvert", 
                "--to", "notebook", 
                "--execute",
                "--inplace",
                notebook_path
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"✅ Completed {os.path.basename(notebook_path)} in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {os.path.basename(notebook_path)}")
        print(f"Error details: {e.stderr}")
        return False

def main():
    # Define the order in which notebooks should be run
    notebooks = [
        # Data preparation notebooks first
        "notebooks/MLProject - DEA.ipynb",
        
        # Model training notebooks
        "notebooks/MLProject - AdaBoost_and_GradientBoosting.ipynb",
        "notebooks/MLProject - KNN_Algorithm.ipynb",
        "notebooks/MLProject - Logistic_Regression.ipynb",
        "notebooks/MLProject - Decision_Tree.ipynb",
        "notebooks/MLProject - SVM.ipynb",
        "notebooks/MLProject - Random_Forest.ipynb",
        
        # Final evaluation notebooks
        # "notebooks/MLProject - Model_Comparison.ipynb",
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    success_count = 0
    total_notebooks = len(notebooks)
    
    # Run each notebook in sequence
    for idx, notebook in enumerate(notebooks, 1):
        print(f"\n[{idx}/{total_notebooks}] Processing notebook: {notebook}")
        
        if os.path.exists(notebook):
            if run_notebook(notebook):
                success_count += 1
        else:
            print(f"❌ Notebook not found: {notebook}")
    
    print(f"\nExecution completed. {success_count}/{total_notebooks} notebooks ran successfully.")

if __name__ == "__main__":
    print("Starting notebooks execution sequence...")
    main()