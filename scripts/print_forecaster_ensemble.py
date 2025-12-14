import os
import sys

from forecaster.providers import (
    DEFAULT_ENSEMBLE,
    _PROVIDER_STATES,
    default_ensemble_summary,
)
from pythia.llm_profiles import get_current_models, get_current_profile


if __name__ == "__main__":
    print(f"PYTHIA_DEBUG_MODELS={os.getenv('PYTHIA_DEBUG_MODELS','')}")

    llm_profile = get_current_profile()
    print(f"PYTHIA_LLM_PROFILE={llm_profile}")

    profile_models = get_current_models()
    if profile_models:
        print("LLM profile models:")
        for name, model in sorted(profile_models.items()):
            print(f"  {name}: {model}")
    else:
        print("LLM profile models: (none)")

    print(f"DEFAULT_ENSEMBLE size={len(DEFAULT_ENSEMBLE)}")
    print(default_ensemble_summary())

    print("Provider states:")
    for name in sorted(_PROVIDER_STATES.keys()):
        state = _PROVIDER_STATES.get(name, {})
        enabled = bool(state.get("enabled"))
        model_present = bool(state.get("model"))
        key_present = bool(state.get("api_key"))
        active = bool(state.get("active"))
        env_key = state.get("env_key") or ""
        line = (
            f"{name}: enabled={str(enabled).lower()} "
            f"model_present={str(model_present).lower()} "
            f"key_present={str(key_present).lower()} "
            f"active={str(active).lower()}"
        )
        if env_key:
            line = f"{line} env_key={env_key}"
        print(line)

    if len(DEFAULT_ENSEMBLE) == 0:
        print("ERROR: DEFAULT_ENSEMBLE has no active models. Check secrets/config.")
        sys.exit(1)
