# Axial Hip Rotation Analysis
Full analysis code for the study:

**Morphological Constraints on Axial Hip Rotation Across Mammals**

This repository contains the complete Python implementation of the modelling pipeline described in the manuscript.

---

## Contents
- `hip_rotation_analysis_full.py`  
  Full analysis code including:
  - Species parameters
  - Geometric ROM → stiffness mapping
  - Passive dynamic simulation (single-DOF model)
  - Morphological substitution tests

---

## Requirements
pip install numpy
---

## How to Run
Run the full analysis:
python hip_rotation_analysis_full.py

This outputs:
- Species ROM & stiffness table  
- Passive dynamic metrics (RMS, 95% range)  
- Human–other morphology substitution results  

---

## Using Functions in a Notebook
```python
from hip_rotation_analysis_full import (
    SPECIES_PARAMS,
    assign_stiffness_to_all_species,
    run_all_dynamic_simulations,
)

assign_stiffness_to_all_species(SPECIES_PARAMS)
results = run_all_dynamic_simulations(SPECIES_PARAMS)
print(results)

Files
hip_rotation_analysis_full.py   # Main analysis script
README.md                       # This document
