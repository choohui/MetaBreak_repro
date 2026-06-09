"""Pipeline stages (imported by run_all via importlib because of the NN_ prefix).

  01_build_data       — datasets + splits per model (model-free)
  02_prepare_defenses — per-model calibration of each defense (needs model)
  03_evaluate         — defended eval on attack / gsm8k+header / benign (needs model)
  04_report           — aggregate to REPORT.md (model-free)
"""
