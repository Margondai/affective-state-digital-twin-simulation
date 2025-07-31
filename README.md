# Affective State Modeling to Predict Training Dropout in Military Academies

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![Conference](https://img.shields.io/badge/Conference-MODSIM%20World%202025-green.svg)](https://modsimworld.org/)

## Abstract

This study demonstrates a digital twin simulation framework integrating physiological data, particularly heart rate variability (HRV) via RMSSD, to proactively predict dropout risk by modeling trainees' affective states (stable, fatigued, burnout). A Markovian state transition model with adaptive training interventions was applied to 100 virtual trainees, split into a control group following standard protocols and an intervention group with adjusted training loads based on real-time HRV thresholds. Results showed significant burnout reductions in the intervention group, with 63.03% time in stable states and only 10.14% in burnout, compared to the control group's 21.16% stable and 36.69% burnout states, marking a 72.4% reduction in burnout.

## Overview

Military training environments demand peak physical and mental performance under extreme stressors like sleep deprivation, prolonged exertion, and high cognitive loads. While attrition is often attributed to injuries and poor fitness, emotional fatigue and cognitive overload are equally impactful, yet remain underexplored. This research addresses this gap by developing a neuroadaptive system that uses real-time HRV data to drive personalized training adjustments.

### Key Features

- **Real-time HRV Monitoring**: Uses RMSSD (Root Mean Square of Successive Differences) as a non-invasive stress indicator
- **Digital Twin Architecture**: Multi-agent simulation where each virtual trainee mirrors real physiological characteristics
- **Markov State Transitions**: Models affective states (Stable, Fatigued, Burnout) with evidence-based transition probabilities
- **Adaptive Training Protocol**: Automatically adjusts training intensity based on HRV thresholds
- **Statistical Validation**: Comprehensive analysis with real PhysioNet data validation

## System Architecture

The digital twin framework consists of three main components:

1. **Physiological Monitoring**: Continuous HRV (RMSSD) tracking with validated stress thresholds
2. **State Classification**: Markov model predicting transitions between Stable, Fatigued, and Burnout states
3. **Adaptive Intervention**: Real-time training load adjustment based on physiological feedback

### HRV Thresholds

- **Stable State**: RMSSD > 5.8 ln(ms) - Optimal recovery state
- **Fatigued State**: RMSSD 4.3-5.8 ln(ms) - Intermediate stress accumulation
- **Burnout State**: RMSSD < 4.3 ln(ms) - High dropout risk requiring intervention

## Results

### Synthetic Simulation Results

**Control Group (Standard Training):**
- 21.16% time in Stable state
- 36.69% time in Burnout state  
- 60.98% dropout risk

**Intervention Group (HRV-Guided Training):**
- 63.03% time in Stable state
- 10.14% time in Burnout state
- 0% dropout risk
- **72.4% reduction in burnout**

### Real Data Validation

Validation with 36 PhysioNet participants confirmed model robustness:
- **76.2% burnout reduction** in real data conditions
- Model convergence within 0.1% of synthetic results
- Consistent effectiveness across demographic subgroups

## Installation

### Prerequisites

- Python 3.8 or higher
- Required dependencies listed in `requirements.txt`

### Setup

```bash
git clone https://github.com/yourusername/affective-state-digital-twin-simulation.git
cd affective-state-digital-twin-simulation
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete simulation with both synthetic and real data validation:

```bash
python affective_state_simulation.py
```

The system will automatically:
- Generate 100 synthetic trainee agents with realistic demographics
- Run control and intervention simulations (50,000 time steps each)
- Validate results against PhysioNet data
- Generate comprehensive statistical analysis
- Create visualization plots and reports

### Output Files

The simulation generates several output files in the `outputs/` directory:

- **CSV Data**: `cindy_simulation_synthetic.csv`, `cindy_simulation_real.csv`
- **Statistical Reports**: `statistical_analysis_*.txt` files
- **Visualizations**: State distribution and HRV trend plots
- **Comparison Analysis**: Synthetic vs. real data validation results

## Methodology

### Digital Twin Model Structure

Each virtual trainee is represented as a software agent with:

- **Demographic Characteristics**: Age, sex, BMI, physical activity level, nicotine use
- **Baseline HRV**: Calculated using physiological research parameters
- **Stress Resilience Profile**: High-risk vs. standard risk categorization
- **Individual Variation**: Gaussian noise modeling genetic/environmental differences

### Markov State Transitions

The model uses evidence-based transition probabilities:

**Standard Population:**
- Stable → Fatigued: 30%, Stable → Burnout: 5%
- Fatigued → Stable: 15%, Fatigued → Burnout: 20%
- Burnout → Stable: 10%, Burnout → Fatigued: 30%

**High-Risk Population:**
- Modified matrices with increased burnout susceptibility
- Reduced recovery rates reflecting stress tolerance differences

### Adaptive Training Protocol

**Intervention Triggers:**
- **RMSSD < 4.3**: Mandatory rest (+0.3 RMSSD recovery boost)
- **RMSSD 4.3-5.8**: Low-intensity training (+0.15 RMSSD improvement)
- **RMSSD > 5.8**: Standard training intensity maintained

## Statistical Analysis

The framework includes comprehensive statistical validation:

- **T-tests** for continuous outcome comparisons
- **Chi-square tests** for dropout rate analysis
- **ANOVA** with effect size calculations (eta-squared)
- **Tukey HSD** for multiple comparisons
- **Cohen's d** for effect size interpretation

### Key Statistical Results

- **Control vs. Intervention Stable States**: t = -33.73, p < .001
- **Control vs. Intervention Burnout States**: t = 16.65, p < .001
- **Dropout Risk Elimination**: χ² = 44.77, p < .001
- **Large Effect Sizes**: η² > 0.79 for all primary outcomes

## File Structure

```
affective-state-digital-twin-simulation/
├── affective_state_simulation.py    # Main simulation file
├── README.md                        # Project documentation
├── LICENSE                          # MIT license
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package installation
├── .gitignore                      # Git ignore rules
└── outputs/                        # Generated results (created automatically)
    ├── simulation_data/            # CSV files with raw results
    ├── statistical_analysis/      # Statistical reports
    ├── visualizations/            # Plots and charts
    └── logs/                      # Execution logs
```

## Applications Beyond Military Training

This framework demonstrates broad applicability across high-stress domains:

### Healthcare and Medical Training
- Medical residency programs with high burnout rates
- Simulation-based medical education optimization
- Reducing physician turnover costs

### Corporate and Industrial Applications  
- High-stress corporate environments (finance, consulting, technology)
- Executive training and development programs
- Employee wellness and retention initiatives

### Educational Institutions
- Intensive academic programs (engineering, pre-medical tracks)
- Graduate student stress management
- Academic attrition prevention

### Emergency Services
- Police academy training optimization
- Firefighter stress resilience programs
- Emergency medical services training

## Contributing

This is an academic research project. For contributions or collaborations, please contact the authors directly.

## Conference Presentation

This work was presented at MODSIM World 2025. Conference materials including the full paper, presentation slides, and supplementary materials are available in the `conference/` directory.

## Citation

If you use this software in your research, please cite:

```
Von Ahlefeldt, C., Margondai, A., & Mouloua, M. (2025). 
Affective State Modeling to Predict Training Dropout in Military Academies. 
MODSIM World 2025, Orlando, FL.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about this research, please contact:

- Cindy Von Ahlefeldt: Cindy.Vonahlefeldt@ucf.edu  
- Ancuta Margondai: Ancuta.Margondai@ucf.edu
- Dr. Mustapha Mouloua: Mustapha.Mouloua@ucf.edu

University of Central Florida  
Orlando, Florida

## Acknowledgments

This research utilized the PhysioNet Wearable Stress Dataset and was conducted at the University of Central Florida Human Factors and Cognitive Psychology Laboratory. Special thanks to the MODSIM World 2025 conference organization and the broader digital twin research community.
