---
title: "MLflow x Hydra"
date: 2022-06-13T20:58:18+08:00
draft: false
summary: "A template for using both effectively in machine learning experiments"
tags: ["mlops"]
math: false
---

## A template for using both effectively in machine learning experiments

I like using [MLflow](https://mlflow.org/) for experiment tracking, and [Hydra](https://hydra.cc/) for experiment configuration management and hyperparameter tuning. MLflow enables logging and visualizing all relevant info for each machine learning model trained without much effort. Hydra drastically simplifies managing and running experiments involving lots of hyperparameters, without requiring much extra code. 

Although both libraries are functionally complementary, there are subtle interactions between them that can lead to unexpected gotchas that are barely mentioned on the internet. After a couple of projects, I eventually arrived at the annotated script template below for using MLflow and Hydra effectively in the same codebase. 

If you're looking for a tweaked and tested template integrating both libraries to kick start your work, you might find this post useful.

## Training script template

The sample config and training script template can be found at [tnwei/mlflow-x-hydra](https://github.com/tnwei/mlflow-x-hydra), reproduced below for copy-paste convenience. 

### `conf/train.yaml`

```yaml
# Metadata
# --------
expname: testrun
runname: # Left blank on purpose, to be specified in CLI

# Hyperparameters
# ---------------
n_epochs: 10
lr: 1e-5
batch_size: 32

# Hydra-specific config
# ---------------------
# The following stores single runs and multiruns (sweeps) in a hidden .hydra dir
hydra:
  run:
  ¦ dir: .hydra/${now:%Y-%m-%d_%H-%M-%S}
  ¦ # If this works for you why not
  ¦ # dir: .hydra/${expname}/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
  ¦ dir: .hydra/${now:%Y-%m-%d_%H-%M-%S}
  ¦ # If this works for you why not
  ¦ # dir: .hydra/${expname}/${now:%Y-%m-%d_%H-%M-%S}
  ¦ subdir: ${hydra.job.num}
```

### `main.py`

```python
"""
Example multirun command: python main.py -m runname=testrun lr=1e-4,5e-5,1e-5,5e-6
"""

# Import everything we need
# -------------------------
import sys
from pathlib import Path
import random
import matplotlib.pyplot as plt

# Import hydra and mlflow
# -----------------------
import hydra
import mlflow

# Obtain current working directory outside Hydra
# ----------------------------------------------
# Why: wd is changed temporarily for each Hydra run
# Original wd can still be retrieved at runtime with
# hydra.utils.get_original_cwd()
# But I prefer sticking to std lib when possible
wd = Path(__file__).parent.resolve()

# Define MLflow tracking and artifact storage URI
# -----------------------------------------------
# This is required whenever a new MLflow experiment is defined
# Do not change halfway through! Stick to one
# Migrating mlflow backend is non-trivial

# Default config: local backend + local artifact store
# Command for UI: mlflow ui
# Format for folder is `file://` + absolute file path, following file URI scheme
TRACKING_URI = f"file://{wd}/mlruns"  # This is default location
ARTIFACT_URI = TRACKING_URI

# Alternative config: sqlite backend + local artifact store
# Command for UI: mlflow ui --backend-store-uri sqlite:///mlruns.sqlite
# Format for sqlite is `sqlite:///` + absolute file path. Three slashes!
# TRACKING_URI = f"sqlite:///{wd}/mlruns.sqlite"
# ARTIFACT_URI = f"file://{wd}/mlruns"

# More options available but omitted for brevity
# See https://www.mlflow.org/docs/latest/tracking.html#how-runs-and-artifacts-are-recorded

mlflow.set_tracking_uri(TRACKING_URI)

# Main logic
# ----------
# Main training logic needs to be:
# (1) packaged into a function
# (2) wrapped with hydra.main() decorator
# (3) and called at the end of the script
# for Hydra to work
#
# Config path is dir to where the config is stored
# Config name is the name of the config file
@hydra.main(config_path="conf", config_name="train.yaml")
def main(cfg):
    # Hydra config parsing
    # --------------------
    # cfg is literally a dict of everything defined in the conf file
    # or passed from CLI
    # refer to conf.yaml for example
    print(cfg)

    # Unpack variables that will be used multiple times throughout the script
    # -----------------------------------------------------------------------
    # Why: This is to keep the script modular if migrating away from Hydra
    # Saves time if want to switch to some other runner e.g. Weights and Biases sweeps
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    batch_size = cfg.batch_size

    # Set MLFlow experiment
    # ---------------------
    # All runs recorded under the same exp name will show up on the same table in MLFlow
    # Create exp if not exists
    if mlflow.get_experiment_by_name(cfg.expname) is None:
        mlflow.create_experiment(
            name=cfg.expname, artifact_location=ARTIFACT_URI + "/" + cfg.expname
        )
    mlflow.set_experiment(cfg.expname)

    # Explicitly set new MLFlow run
    # -----------------------------
    # Explicitly set a new run:
    # Why 1: Running w/ hydra will pick up from a prev run
    # See https://stackoverflow.com/a/72605175/13095028
    # Why 2: Allows fine-grained control over Run object e.g. setting `run_name`
    # Note: Leaving run_id to be auto-generated, since the randomly generated UUID is convenient
    activerun = mlflow.start_run(run_name=cfg.runname)

    # Explicitly configure savedir
    # ----------------------------
    # Why: Create consistent savedir for all MLflow exps instead of using Hydra
    # Format: outputs/expname/runid/
    # Hydra's approach will have one savedir for each run
    # Use run ID instead of run name as latter isn't always defined
    savedir = wd / "outputs" / f"{cfg.expname}/{activerun.info.run_id}"
    if not savedir.exists():
        savedir.mkdir(parents=True)

    # Log hyperparameters
    # -------------------
    # Params can only be logged once, while metrics are logged over time

    # Log dict of parameters
    mlflow.log_params(cfg)

    # Log one parameter
    # Note: found logging the command that triggered the file to run to be useful
    mlflow.log_param("argv", " ".join(sys.argv))

    # Data loading and preprocessing
    # ------------------------------
    # ...
    # YOUR CODE HERE
    # ...

    # Main training loop
    # ------------------

    for epoch in range(n_epochs):
        # Model training
        # --------------
        # ...
        # YOUR CODE HERE
        # ...

        # Compute evaluation metrics
        # --------------------------
        print(f"Calculating metrics at step {epoch}/{n_epochs}")
        eval_metrics = {"acc": epoch / n_epochs}

        # Log metrics
        # -----------
        # Log dict of metrics
        mlflow.log_metrics(eval_metrics, step=epoch)

        # Log one metric
        mlflow.log_metric("luck", random.random(), step=epoch)

        # Save training outputs
        # ---------------------
        # Store artifacts by step in the defined savedir
        (savedir / f"{i:02d}.csv").touch()

        # Create a plot
        fig, ax = plt.subplots()
        ax.plot([0, 1], [2, 3])

        # If needed, log training outputs to MLflow artifact store
        # --------------------------------------------------------
        # Log one file
        mlflow.log_artifact(local_path=savedir / f"{i:02d}.csv")

        # Log the whole dir
        mlflow.log_artifacts(local_dir=savedir)

        # Log a figure
        mlflow.log_figure(fig, artifact_file=f"{i:02d}.png")

    # End MLflow run
    # --------------
    # Alternative: wrap code in `with mlflow.start_run() as run:`
    mlflow.end_run()


if __name__ == "__main__":
    main()
```

## Reasoning

Elaborating upon the more opinionated parts of the template below:

### Obtain current working directory outside Hydra

```python
# main.py
wd = Path(__file__).parent.resolve()
```

Hydra changes the work directory temporarily every time the script is executed. The idea is each script execution will have its outputs saved to a standalone folder. To refer to files based on the original working directory, we can either call `hydra.utils.get_original_cwd()`, or as I prefer it, save the original working directory using `pathlib` which is a standard library in Python. 

### Unpack variables that will be used multiple times throughout the script

```python
# main.py
n_epochs = cfg.n_epochs
lr = cfg.lr
batch_size = cfg.batch_size
```

All specified configs will be made available through the `cfg` argument passed to the `main()` function, as items in a dictionary. Instead of directly referring to these values, I assign them to independent variables at the start of the script. I prefer keeping the Hydra-specific code localized to just the head of the script. In the event  that I want to use another script runner like Weights and Biases sweeps instead, refactoring will just need tweaking a few lines. 

### Explicitly set new MLFlow run

```python
# main.py
activerun = mlflow.start_run(run_name=cfg.runname)
```

An MLflow Run is created explicitly in the script template. The reasons are two-fold:

1. If left to be automatically managed by MLflow, Hydra will consider all MLflow runs in the same Hydra multirun session to be the same. See [my answer to this Stackoverflow question](https://stackoverflow.com/a/72605175/13095028)
2. Explicit run creation allows greater control over the MLflow Run object. Example here is it allows us to set the `run_name`, which would be blank in the MLflow UI otherwise.

### Explicitly configure savedir

Out of the box, Hydra creates an `outputs/`directory to store the results of each execution, sorted by date. Results for each execution is saved in its own folder, using the current timestamp to the second as the folder name. Multiruns are stored in a separate `multirun` folder. This behaviour is convenient for integrating Hydra within existing code without much modification, but can be a bit tedious when it comes to accessing past results for future use. 

```python
# main.py
savedir = wd / "outputs" / f"{cfg.expname}/{activerun.info.run_id}"
if not savedir.exists():
    savedir.mkdir(parents=True)
```

```yaml
# conf/train.yaml
hydra:
  run:
    dir: .hydra/${now:%Y-%m-%d_%H-%M-%S}
  sweep:
    dir: .hydra/${now:%Y-%m-%d_%H-%M-%S}
    subdir: ${hydra.job.num}
```

In this script template, the output directory is explicitly configured to be `<cwd> /outputs/`, further sorted by experiment name and MLflow run ID. This approach consolidates all output into a single folder, compared to Hydra's default approach. Given that Hydra still creates output directories for each execution, the template redirects them to a `.hydra` directory which is out of sight.

I would have preferred using Hydra's configuration to determine the save location instead of setting it aside completely, but the save locations are determined upon execution, and Hydra does not accept variables created at runtime. Using the Hydra config would rule out using auto-generated names, requiring the user to manually provide a name for each script execution.

