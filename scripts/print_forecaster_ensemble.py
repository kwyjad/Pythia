import os
import sys
from forecaster.providers import DEFAULT_ENSEMBLE, default_ensemble_summary

if __name__ == "__main__":
    print(f"PYTHIA_DEBUG_MODELS={os.getenv('PYTHIA_DEBUG_MODELS','')}")
    print(f"DEFAULT_ENSEMBLE size={len(DEFAULT_ENSEMBLE)}")
    print(default_ensemble_summary())
    if len(DEFAULT_ENSEMBLE) == 0:
        print("ERROR: DEFAULT_ENSEMBLE has no active models. Check secrets/config.")
        sys.exit(1)
