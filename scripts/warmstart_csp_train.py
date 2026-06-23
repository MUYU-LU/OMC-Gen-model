import json
import logging
from pathlib import Path

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.cli import SaveConfigCallback

from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT, get_device
from mattergen.diffusion.run import AddConfigCallback, SimpleParser, maybe_instantiate

logger = logging.getLogger(__name__)


def init_lightningmodule_from_pretrained(cfg):
    ckpt_info = MatterGenCheckpointInfo(
        model_path=Path(hydra.utils.to_absolute_path(cfg.warmstart.model_path)),
        load_epoch=cfg.warmstart.load_epoch,
    )
    ckpt_path = ckpt_info.checkpoint_path
    logger.info(f"Warm-start loading from {ckpt_path}")

    lightning_module = hydra.utils.instantiate(cfg.lightning_module)
    checkpoint = torch.load(ckpt_path, map_location=get_device())
    pretrained = checkpoint["state_dict"]
    scratch = lightning_module.state_dict()

    matched = sorted(
        k for k in (scratch.keys() & pretrained.keys()) if scratch[k].shape == pretrained[k].shape
    )
    shape_mismatch = sorted(
        k for k in (scratch.keys() & pretrained.keys()) if scratch[k].shape != pretrained[k].shape
    )
    skipped = sorted(set(pretrained.keys()) - set(matched))
    missing = sorted(set(scratch.keys()) - set(matched))

    scratch.update((k, pretrained[k]) for k in matched)
    incompatible = lightning_module.load_state_dict(scratch, strict=False)
    return lightning_module, ckpt_path, matched, skipped, missing, incompatible, shape_mismatch


@hydra.main(config_path=str(MODELS_PROJECT_ROOT / "conf"), config_name="warmstart_csp", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    torch.set_float32_matmul_precision("high")
    trainer: pl.Trainer = maybe_instantiate(cfg.trainer, pl.Trainer)
    datamodule: pl.LightningDataModule = maybe_instantiate(cfg.data_module, pl.LightningDataModule)

    pl_module, ckpt_path, matched, skipped, missing, incompatible, shape_mismatch = init_lightningmodule_from_pretrained(cfg)

    config_as_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps({
        "warmstart_checkpoint": ckpt_path,
        "matched_keys": len(matched),
        "skipped_pretrained_keys": len(skipped),
        "missing_scratch_keys": len(missing),
        "shape_mismatch_keys": shape_mismatch,
        "incompatible_missing": list(incompatible.missing_keys),
        "incompatible_unexpected": list(incompatible.unexpected_keys),
    }, indent=2))

    trainer.callbacks.append(
        SaveConfigCallback(
            parser=SimpleParser(),
            config=config_as_dict,
            overwrite=True,
        )
    )
    trainer.callbacks.append(AddConfigCallback(config_as_dict))
    trainer.fit(model=pl_module, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    main()
