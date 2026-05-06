import os
from numbers import Number

import torch

from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback


class WandBLogger(Callback):
    def __init__(
        self,
        enabled: bool = False,
        project: str | None = None,
        entity: str | None = None,
        mode: str | None = None,
        log_every_n: int | None = None,
    ):
        self.enabled = enabled
        self.project = project
        self.entity = entity
        self.mode = mode
        self.log_every_n = log_every_n
        self._wandb = None

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        if not self.enabled or not distributed.is_rank0():
            return

        import wandb

        self._wandb = wandb
        config_job = self.config.job
        project = self.project or os.environ.get("WANDB_PROJECT") or config_job.project
        entity = self.entity or os.environ.get("WANDB_ENTITY") or None
        mode = self.mode or os.environ.get("WANDB_MODE") or "online"
        run_dir = os.environ.get("WANDB_DIR") or config_job.path_local

        os.makedirs(run_dir, exist_ok=True)
        wandb.init(
            project=project,
            entity=entity,
            group=config_job.group,
            name=config_job.name,
            dir=run_dir,
            mode=mode,
            resume="allow",
            config={
                "job_path": config_job.path,
                "job_path_local": config_job.path_local,
                "world_size": distributed.get_world_size(),
            },
        )

        config_yaml = os.path.join(config_job.path_local, "config.yaml")
        if os.path.exists(config_yaml):
            wandb.save(config_yaml, policy="now")
        log.info(f"WandB logging enabled: project={project}, group={config_job.group}, name={config_job.name}")

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if not self.enabled or not distributed.is_rank0():
            return

        every_n = self.log_every_n or self.config.trainer.logging_iter
        if iteration % every_n != 0:
            return

        if self._wandb is None:
            return

        metrics: dict[str, float | int] = {"train/loss": float(loss.detach().float().item())}
        for key, value in output_batch.items():
            scalar = self._to_scalar(value)
            if scalar is not None:
                metrics[f"train/{key}"] = scalar

        self._wandb.log(metrics, step=iteration)

    def on_app_end(self) -> None:
        if self.enabled and distributed.is_rank0() and self._wandb is not None and self._wandb.run is not None:
            self._wandb.finish()

    @staticmethod
    def _to_scalar(value) -> float | int | None:
        if isinstance(value, Number):
            return value
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return float(value.detach().float().item())
        return None
