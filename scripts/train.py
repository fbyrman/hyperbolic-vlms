# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# Modified from github.com/facebookresearch/meru

"""
Train a MERU, HyCoCLIP, PHyCLIP or ARGENT model, or its Euclidean control, based on
parameters specified by a config file.
"""

import argparse
import multiprocessing as mp
import random
import socket
import time
from pathlib import Path

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if mp.get_start_method(allow_none=True) != "spawn":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.amp.grad_scaler import GradScaler

import gvlm.utils.distributed as dist
from gvlm.config import LazyConfig, LazyFactory
from gvlm.models import EuclideanHyCoCLIP, EuclideanPhyCLIP, HyCoCLIP, PHyCLIP
from gvlm.tokenizer import Tokenizer
from gvlm.utils.checkpointing import CheckpointManager
from gvlm.utils.directory import generate_output_dir_name
from gvlm.utils.timer import Timer
from gvlm.utils.worker import (
    evaluation_worker,
    tensorboard_worker,
)

parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--config", help="Path to a .py config file.")
parser.add_argument(
    "--output-dir",
    help="Write the run here instead of the directory named after the config's "
    "parameters. The iteration count is not one of those parameters, so a "
    "shorter rerun of a config resolves to the same directory as the full run "
    "and would overwrite it.",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Whether to resume training from `--output-dir`. This script will find "
    "the last saved checkpoint and resume training. It is user's responsibility "
    "to provide matching config file in `--config`.",
)
parser.add_argument(
    "--checkpoint-period", type=int, default=50000, help="Checkpoint saving period."
)
parser.add_argument(
    "--keep-recent",
    type=int,
    default=100,
    help="Number of recent checkpoints to keep on disk during training.",
)
parser.add_argument(
    "--eval-period",
    type=int,
    default=0,
    help="Evaluate every this many iterations, 0 to never. Needs the evaluation "
    "datasets set up (see gvlm/README.md) and should be a multiple of "
    "--checkpoint-period, since it evaluates the checkpoint written at that "
    "iteration.",
)
parser.add_argument(
    "--log-period",
    type=int,
    default=100,
    help="Log to stdout/tensorboard periodically (only main process).",
)
parser.add_argument(
    "--num-machines",
    type=int,
    default=1,
    help="Number of machines used in distributed training.",
)
parser.add_argument(
    "--num-gpus", type=int, default=0, help="Number of GPUs per machine."
)
parser.add_argument(
    "--machine-rank",
    type=int,
    default=0,
    help="Integer in [0, num_machines) to specifying machine ID.",
)
_random_port = random.randint(2000, 19999)
parser.add_argument(
    "--dist-url",
    default=f"tcp://127.0.0.1:{_random_port}",
    help="URL of the main process in distributed training, it defaults to "
    "localhost for single-machine training.",
)
parser.add_argument(
    "overrides", nargs="...", default=[], help="Config overrides (key-value pairs)."
)


def process_logging_value(log_data, prefix, name, value):
    """Process a single value or list of values with appropriate prefix and name."""
    if isinstance(value, list):
        # For lists, add suffix with index
        for i, v in enumerate(value):
            key = f"{prefix}/{name}_{i:03d}"

            if torch.is_tensor(v):
                scalar_value = v.detach().cpu().item()
            else:
                scalar_value = float(v)

            log_data["scalar"][key] = scalar_value
    else:
        # For non-lists, use name as is without suffix
        key = f"{prefix}/{name}"

        if torch.is_tensor(value):
            scalar_value = value.detach().cpu().item()
        else:
            scalar_value = float(value)

        log_data["scalar"][key] = scalar_value


def main(_A: argparse.Namespace):
    # -------------------------------------------------------------------------
    #   BASIC SETUP FOR TRAINING JOB.
    # -------------------------------------------------------------------------
    # Create a config object and perform common setup.
    _C = LazyConfig.load(_A.config)
    _C = LazyConfig.apply_overrides(_C, _A.overrides)

    output_dir = _A.output_dir or generate_output_dir_name(_C)

    # Handle config loading for resume
    output_dir = Path(output_dir)
    saved_config_yaml_path = output_dir / "config.yaml"

    if _A.resume and saved_config_yaml_path.exists():
        # Load from YAML and fix known issues
        logger.info(
            f"Resuming training. Loading saved config from {saved_config_yaml_path}"
        )
        try:
            # Load from YAML using LazyConfig
            _C = LazyConfig.load(str(saved_config_yaml_path))
            logger.info("Loaded and fixed YAML config successfully")

        except Exception as e:
            logger.warning(f"Failed to load YAML config: {e}")
            logger.info("Using current config file")

        # Apply command line overrides
        if _A.overrides:
            _C = LazyConfig.apply_overrides(_C, _A.overrides)
    elif _A.resume:
        logger.info("No saved config found, using current config file")

    # Create output directory early to ensure it exists for other processes
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if dist.is_main_process():
        # Set up tensorboard worker
        tensorboard_input_queue = mp.Queue()
        # daemon: a crash in main() must not leave live children keeping the
        # slurm job (and its GPU) allocated until the time limit.
        tensorboard_process = mp.Process(
            target=tensorboard_worker,
            args=(tensorboard_input_queue, output_dir),
            daemon=True,
        )
        tensorboard_process.start()

        # Set up evaluation worker, only if periodic evaluation was asked for.
        evaluation_input_queue = None
        if _A.eval_period > 0:
            evaluation_input_queue = mp.Queue()
            evaluation_process = mp.Process(
                target=evaluation_worker,
                args=(evaluation_input_queue, tensorboard_input_queue),
                daemon=True,
            )
            evaluation_process.start()

    # Get process rank and world size (assuming distributed is initialized).
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()

    if getattr(_C.train, "seed", None) is None:
        _C.train.seed = int(time.time())

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(_C.train.seed + RANK)
    np.random.seed(_C.train.seed + RANK)
    torch.manual_seed(_C.train.seed + RANK)
    torch.backends.cudnn.deterministic = _C.train.cudnn_deterministic
    torch.backends.cudnn.benchmark = _C.train.cudnn_benchmark

    # Save config in output directory.
    LazyConfig.save(_C, str(output_dir / "config.yaml"))

    # Create a logger for each process which writes to a separate log-file.
    logger.add(output_dir / f"log-rank{RANK}.txt", format="{time} {level} {message}")

    # Print process info, config and args.
    logger.info(f"Rank of current process: {RANK}. World size: {WORLD_SIZE}")
    logger.info(f"RANK {RANK} using random seed: {_C.train.seed + RANK}")
    logger.info(OmegaConf.to_yaml(_C))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    # -------------------------------------------------------------------------
    #   INSTANTIATE ALL OBJECTS FOR TRAINING.
    # -------------------------------------------------------------------------
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if _A.num_gpus != 0
        else torch.device("cpu")
    )
    dataloader = LazyFactory.build_dataloader(_C)
    tokenizer = Tokenizer()

    model = LazyFactory.build_model(_C, device)

    optimizer = LazyFactory.build_optimizer(_C, model)
    scheduler = LazyFactory.build_lr_scheduler(_C, optimizer)
    scaler = GradScaler(
        enabled=_C.train.amp,
    )

    checkpoint_manager = CheckpointManager(
        output_dir,
        keep_recent=_A.keep_recent,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    start_iteration = checkpoint_manager.resume() if _A.resume else 0

    # Create an iterator from dataloader to sample batches perpetually.
    dataloader_iter = iter(dataloader)
    timer = Timer(start_iteration + 1, total_iterations=_C.train.num_iterations)

    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(start_iteration + 1, _C.train.num_iterations + 1):
        data_time = time.perf_counter()
        batch = next(dataloader_iter)
        data_time = time.perf_counter() - data_time

        timer.tic()
        optimizer.zero_grad()
        # with amp.autocast(enabled=_C.train.amp):
        with torch.autocast(device_type="cuda", enabled=_C.train.amp):
            # Get image and text (tokens) from batch and pass through model.

            # DDP hides the class behind a wrapper, so unwrap before dispatching.
            actual_model = (
                model.module
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model
            )
            # The grounded models take boxes as separate arguments; ARGENT and
            # EuclideanARGENT come along as subclasses.
            if isinstance(
                actual_model,
                (HyCoCLIP, PHyCLIP, EuclideanHyCoCLIP, EuclideanPhyCLIP),
            ):
                tokens = tokenizer(batch["text"])
                box_tokens = tokenizer(batch["box_text"])
                output_dict = model(
                    batch["image"].to(device),
                    batch["box_image"].to(device),
                    tokens,
                    box_tokens,
                )
            else:
                # For CLIP and other models, check if use_boxes is enabled
                if getattr(actual_model, "use_boxes", False):
                    # Concatenate box data to increase training samples
                    # Concatenate images: [image, box_image]
                    all_images = torch.cat([batch["image"], batch["box_image"]], dim=0)

                    # Concatenate texts: [text, box_text]
                    all_texts = batch["text"] + batch["box_text"]
                    tokens = tokenizer(all_texts)

                    output_dict = model(all_images.to(device), tokens)
                else:
                    # Standard CLIP training without box data
                    tokens = tokenizer(batch["text"])
                    output_dict = model(batch["image"].to(device), tokens)

            loss = output_dict["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        timer.toc()

        # Log statistics to terminal and tensorboard.
        if iteration % _A.log_period == 0:
            timer_stats = (
                f"Iter {timer.iteration} | Time (sec): {data_time:.3f} data, "
                f"{timer.deltas[-1]:.3f} model | ETA: {timer.eta_hhmm}"
            )

            log_str = f"{timer_stats} [GPU {dist.gpu_mem_usage()} MB]"
            for key, value in output_dict["logging"].items():
                if not (isinstance(value, list) or isinstance(value, dict)):
                    log_str += f" [{key} {value:.3f}]"

            logger.info(log_str)

            if dist.is_main_process():
                log_data = {
                    "iteration": iteration,
                    "scalar": {
                        "lr": scheduler.get_last_lr()[0],
                        "amp_scale": scaler.get_scale(),
                    },
                    "type": "scalar",
                }

                for name, _value in output_dict["logging"].items():
                    prefix = (
                        "curv" if name == "curv" else "train"
                    )  # Use 'train' prefix for all except 'curv'
                    process_logging_value(log_data, prefix, name, _value)

                # Check data size before sending
                try:
                    # Use non-blocking put with timeout to avoid hanging
                    tensorboard_input_queue.put(log_data, timeout=5.0)
                    logger.debug(
                        f"Successfully sent log data to TensorBoard queue for iteration {iteration}"
                    )
                except Exception as e:
                    logger.error(f"Failed to send log data to TensorBoard queue: {e}")
                    logger.warning(
                        "Continuing training without TensorBoard logging for this iteration"
                    )

        # Save checkpoint to disk.
        if iteration % _A.checkpoint_period == 0 and dist.is_main_process():
            checkpoint_manager.step(iteration)

        # Send evaluation job. Kept off the checkpoint period on purpose: the
        # eval datasets are a separate setup and a missing one used to fail here
        # every time a checkpoint was written.
        if (
            _A.eval_period > 0
            and iteration % _A.eval_period == 0
            and dist.is_main_process()
        ):
            try:
                checkpoint_path = output_dir / f"checkpoint_{iteration:08d}.pth"
                yaml_config_path = output_dir / "config.yaml"

                evaluation_input_queue.put(
                    (str(checkpoint_path), str(yaml_config_path), iteration)
                )
                logger.info(f"Queued evaluation job for iteration {iteration}")
            except Exception as e:
                logger.error(f"Error in queueing evaluation job: {str(e)}")

    # Save the final checkpoint.
    if dist.is_main_process():
        checkpoint_manager.final_step(_C.train.num_iterations)

        # Send termination signal
        tensorboard_input_queue.put(None)
        tensorboard_process.join()

        if evaluation_input_queue is not None:
            evaluation_input_queue.put(None)
            evaluation_process.join()


if __name__ == "__main__":
    _A = parser.parse_args()
    if _A.eval_period > 0 and _A.eval_period % _A.checkpoint_period != 0:
        parser.error(
            f"--eval-period {_A.eval_period} must be a multiple of "
            f"--checkpoint-period {_A.checkpoint_period}, or the checkpoint it "
            "evaluates will not have been written"
        )
    if _A.num_gpus == 0:
        main(_A)
    else:
        # This will launch `main` and set appropriate CUDA device (GPU ID) as
        # per process (accessed in the beginning of `main`).
        # cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        # stdout = subprocess.check_output(cmd.split())
        # host_name = stdout.decode().splitlines()[0]
        # logger.info(f"Host name: {host_name}")
        # dist_url = f'tcp://{host_name}:{_random_port}'
        # logger.info(f"Distributed URL: {dist_url}")

        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)

        dist_url = f"tcp://{IPAddr}:{_random_port}"

        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus,
            machine_rank=_A.machine_rank,
            dist_url=dist_url,
            args=(_A,),
        )
