# Bachelor_WhenPriorsReshapePerception

This repository contains the analysis code and supporting materials for the bachelor thesis:

**“When Priors Reshape Perception: A MEG Characterization of Large-Scale Network Reconfiguration during Mooney Image Disambiguation”**

The project investigates how prior knowledge reshapes visual perception under ambiguity, using MEG data and the BROAD-NESS framework to characterize large-scale brain network dynamics.


## Data

**not publicly available**.

---

## Stimuli

The experimental paradigm used **Mooney images**, presented under three conditions:

- **Degraded1** – initial presentation of degraded images (no prior)
- **ClearCue** – presentation of the corresponding clear image
- **Degraded2** – repeated presentation of the degraded image after disambiguation

Example images are located in the `stimuli/` directory.

---

## Analysis

All analyses were conducted in Python/MATLAB (adjust if needed) and are organized in the `analysis/` directory.

Key steps include:
- MEG preprocessing and source reconstruction
- Estimation of large-scale brain networks using **BROAD-NESS**
- Time-resolved statistical contrasts between experimental conditions
- Visualization of network dynamics and trajectories

The BROAD-NESS toolbox can be found here:  
**[BROADNESS_ToolboxBROADNESS_Toolbox](https://github.com/leonardob92/BROADNESS_MEG_AuditoryRecognition/tree/977e8ad2bf8e8c634da295d76e56fca60120502c/BROADNESS_Toolbox)**

---

## Output overview

- Time-resolved statistical contrasts between experimental conditions
- NIFTI files used for visualization
- Figures corresponding to those reported in the thesis

File names follow a consistent naming convention to facilitate traceability between code, output, and thesis figures.

---

## Author

Sissel Højgaard Vang-Pedersen
Bachelor thesis, [Aarhus University/ Cognitive Science]  
06-01-2026
