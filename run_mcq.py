import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Executing MCQ Solver...")
    try:
        import src.pipelines.mcq_solver
        print("MCQ Solver completed.")
    except Exception as e:
        print(f"MCQ Solver failed: {e}")
        sys.exit(1)
