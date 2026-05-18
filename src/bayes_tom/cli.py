import click
from pathlib import Path


def _load_config(path):
    if path.suffix == ".yaml":
        import yaml

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        raise NotImplementedError(f"Unable to read config type: {path.suffix}")

    return config_dict


@click.group
@click.version_option()
def main():
    pass


@main.command()
@click.argument("config")
def evaluate(config):
    """
    Submit a job
    """
    from bayes_tom.evaluation import run_partner_evaluation

    config_path = Path(config)

    config_dict = _load_config(config_path)

    run_partner_evaluation(config_dict, print_metrics=True)


@main.command()
@click.argument("config")
def train_children(config):
    """
    Train children models
    """
    from bayes_tom.teammate_generation.train_children import train_children

    config_path = Path(config)

    config_dict = _load_config(config_path)

    train_children(config_dict)


@main.command()
@click.argument("config")
def train_lbrdiv(config):
    """
    Train partner policies using LBRDiv.
    """
    from bayes_tom.teammate_generation.train_lbrdiv import run_lbrdiv

    config_path = Path(config)
    config_dict = _load_config(config_path)

    logger = _make_logger(config_dict)
    run_lbrdiv(config_dict, logger)


@main.command()
@click.argument("config")
def train_parents(config):
    """
    Train parent (CoMeDi) policies.
    """
    from bayes_tom.teammate_generation.train_parents import run_comedi

    config_path = Path(config)
    config_dict = _load_config(config_path)

    logger = _make_logger(config_dict)
    run_comedi(config_dict, logger)


def _make_logger(config: dict):
    """Return a wandb logger if configured, otherwise a no-op logger."""
    use_wandb = config.get("logger", {}).get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=config["logger"].get("project", "bayes-tom"),
            name=config["logger"].get("name", None),
            config=config,
        )
        return _WandbLogger()
    return _NoOpLogger()


class _WandbLogger:
    def log_item(self, name, value, train_step=None):
        import wandb
        wandb.log({name: value}, step=train_step)

    def commit(self):
        import wandb
        wandb.log({})

    def log_artifact(self, name, path, type_name):
        import wandb
        artifact = wandb.Artifact(name=name, type=type_name)
        artifact.add_dir(path)
        wandb.log_artifact(artifact)

    def log_xp_matrix(self, name, matrix):
        import wandb
        n = matrix.shape[0]
        data = [[int(i), int(j), float(matrix[i, j])] for i in range(n) for j in range(n)]
        table = wandb.Table(data=data, columns=["conf_id", "br_id", "return"])
        wandb.log({name: table})


class _NoOpLogger:
    def log_item(self, name, value, train_step=None):
        pass

    def commit(self):
        pass

    def log_artifact(self, name, path, type_name):
        pass

    def log_xp_matrix(self, name, matrix):
        pass