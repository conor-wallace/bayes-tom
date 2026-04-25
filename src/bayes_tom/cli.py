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

    run_partner_evaluation(config_dict)
