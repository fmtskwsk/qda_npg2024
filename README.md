# README

## Information of the paper
* authors  : Shunji Kotsuki, Fumitoshi Kawasaki and Masanao Ohashi
* title    : Quantum Data Assimilation : Solving Data Assimilation on Quantum Annealers
* year     : 2024

## Quantum data assimilation experiments
* You can generate data and figures used in the paper by executing these codes at your Linux/Unix environment.
* To execute these codes, the following preparations must be conducted prior to experiments.
      1. Obtain access token for Amplify Annealing Engine and D-Wave Quantum Annealer.
      2. Install Amplify SDK and the third-party dependency packages.
      3. Enter the token for Amplify Annealing Engine on line 51 of code "QA_SIM.py" and the token for D-Wave Quantum Annealer on line 51 of code "QA_PHY.py".
For more details, please refer to [here](https://amplify.fixstars.com/en/docs/quickstart.html)

## Note
* These codes are not guaranteed to output completely identical data and figures used in this study. This is because the results will depend on your execution environment and uncontrollable quantum effects.
* While one may see warnings during Code "make_dx_opt_rmse_time_SIM.py" execution, but it works correctly.
* Although it varies greatly depending on the environment, it would take quite a long time, about 9 hours, to execute all the code.