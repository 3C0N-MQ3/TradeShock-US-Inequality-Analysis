# Trade and Labor-Market Outcomes: Impact of the China Trade Shock on U.S. Regional Inequality

This repository contains materials for the project analyzing the impact of a significant international trade shock, specifically the “China shock,” on labor-market outcomes and regional inequality in the United States. The primary focus is to replicate a well-known study and adjust labor-market outcomes based on the composition of various factors.

## Reference paper 
Autor, Dorn, Hanson, 2013, "The China Syndrome: Local Labor Market Effects of Import Competition in the US", AER, 103(6): 2121-2168.
(https://www.nber.org/system/files/working_papers/w18054/w18054.pdf)

Online Data and Theory Appendix:
(https://ddorn.net/papers/Autor-Dorn-Hanson-ChinaSyndrome-Appendix.pdf)

## Project Overview

- **Research Question**: What is the impact of the China trade shock on inequality across U.S. regions?
- **Objective**: Replicate key tables from the referenced study and adjust the labor-market outcomes using updated data and composition adjustments.
- **Tables to Replicate**:
  - Table 3
  - Panel A of Table 5 (excluding column 5)
  - Table 6

## Data Sources

This project uses data from two primary sources:

- **Data I**: Dependent variables are constructed from the 1990 and 2000 Censuses, and the 2007 3-year American Community Survey (2006-2008). Data obtained from [IPUMS USA](https://usa.ipums.org/usa/).
- **Data II**: Independent variables are sourced from a referenced paper, available at [David Dorn's website](http://www.ddorn.net/data.htm).

## Methodology

### Step I: Construct Ten-Year Equivalent Changes in IPUMS Samples

Using data from the 1990, 2000 Censuses and the 2007 3-year ACS (2006-2008), construct the following variables at the **commuting zone (CZ)** level, composition-adjusted:

1. **Average Wage**: 

$$
\log(average \quad wage_{r2007}) - \log(average \quad wage_{r2000})
$$
   
2. **Unemployment Rate**: 

$$
\frac{unemployment \quad rate_{r2007}}{unemployment \quad rate_{r2000}}
$$

3. **Labor Force Participation (LFP) Rate**: 

$$
\frac{LFP \quad rate_{r2007}}{LFP \quad rate_{r2000}}
$$
   
These variables will be calculated for **working-age individuals** (and also for the period 1990-2000 for comparison).

Key variables required from the IPUMS datasets:
- **STATEFIP** and **PUMA**: Used to match the data with Dorn's data.
- **EMPSTAT**: For creating unemployment and labor force participation variables.
- **INCWAGE**: To construct wage variables.

### Step II: Estimation Using 2SLS

Once the variables are constructed, we will use Two-Stage Least Squares (2SLS) to estimate the impact of the “China shock” on commuting zone (CZ)-level labor market outcomes. The model will progressively control for additional CZ-level controls as described in the referenced paper.

## Setting Up the Environment

To ensure that all collaborators use the same environment, you can create the conda environment using the `environment.yml` file provided in the repository.

1. Clone the repository:

```bash
git clone https://github.com/3C0N-MQ3/TradeShock-US-Inequality-Analysis.git
```
2. Navigate to the project directory:

```bash
cd TradeShock-US-Inequality-Analysis
```
3. Create the conda environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

4. Activate the environment:

```bash
conda activate trade-shock-env
```
5. Updating the Environment

If there are changes or updates to the environment (e.g., new packages are added), you can update your local environment using the following command:

```bash
conda env update --file environment.yml --prune
```

The `--prune` option removes any dependencies that are no longer needed, ensuring your environment stays clean and up-to-date.

This will allow anyone working on the project to replicate the environment easily and ensure consistency across different setups.
