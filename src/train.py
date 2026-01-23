import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml
import math
import inspect

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import wandb
import pdb

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, loss_func,
               mode="decoder",
               lasso_guided_opt_lam=None,
               lasso_guided_opt_layer=-2,
               lasso_guided_opt_token=-1,
               w_star=None,
               use_moe=False):
    optimizer.zero_grad()
    
    # Determine what outputs we need
    need_hidden_states = (mode == "encoder" and lasso_guided_opt_lam is not None)
    try:
        sig = inspect.signature(model.forward)
        supports_aux = "return_aux_loss" in sig.parameters
    except (ValueError, TypeError):
        supports_aux = False
    need_aux_loss = use_moe and supports_aux
    
    if need_hidden_states and need_aux_loss:
        output, hidden_states, aux_loss = model(xs, ys, return_hidden_states=True, return_aux_loss=True)
    elif need_hidden_states:
        output, hidden_states = model(xs, ys, return_hidden_states=True)
        aux_loss = 0.0
    elif need_aux_loss:
        output, aux_loss = model(xs, ys, return_aux_loss=True)
    else:
        output = model(xs, ys)
        aux_loss = 0.0
    
    if mode == "decoder":
        loss = loss_func(output, ys)
        if need_aux_loss and isinstance(aux_loss, torch.Tensor):
            loss = loss + aux_loss
    elif mode == "encoder":
        # Predict on final token only in encoder mode
        loss = loss_func(output[:, -1:], ys[:, -1:])
        if lasso_guided_opt_lam is not None:
            B, N, d = xs.shape
            # compute loss between second-to-last layer, and true w_star which has shape Bxdx1
            w_star = w_star.to(xs.device)
            Bw, dw, _ = w_star.shape
            assert Bw == B and dw == d
            w_star = w_star.squeeze(2).view([B, 1, d])
            loss += lasso_guided_opt_lam * ((hidden_states[lasso_guided_opt_layer][:, :lasso_guided_opt_token, -d:] - w_star)**2).sum(dim=2).mean()
        # Add MoE auxiliary loss
        if use_moe and isinstance(aux_loss, torch.Tensor):
            loss = loss + aux_loss
    else:
        raise NotImplementedError
    loss.backward()
    optimizer.step()
    
    # Return aux_loss for logging
    aux_loss_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss
    return loss.detach().item(), output.detach(), aux_loss_val


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        if not args.training.optimizer_reset:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

        # refresh learning rate to the one specified by training args
        if args.training.learning_rate_override:
            for g in optimizer.param_groups:
                g['lr'] = args.training.learning_rate


    all_task_names = [t.name for t in args.training.tasks]

    n_dims = args.model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)

    all_task_samplers = [get_task_sampler(
        task.name,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **task.kwargs,
    ) for task in args.training.tasks]
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    # Device for data generation - generate directly on GPU to avoid CPU bottleneck
    device = torch.device('cuda')

    for i in pbar:

        # Loop over all tasks
        for (task_name, task_sampler) in zip(all_task_names, all_task_samplers):

            data_sampler_args = {}
            task_sampler_args = {}

            if "sparse" in task_name:
                task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
            if num_training_examples is not None:
                assert num_training_examples >= bsize
                seeds = sample_seeds(num_training_examples, bsize)
                data_sampler_args["seeds"] = seeds
                task_sampler_args["seeds"] = [s + 1 for s in seeds]

            # Generate data directly on GPU (key optimization)
            xs = data_sampler.sample_xs(
                curriculum.n_points,
                bsize,
                curriculum.n_dims_truncated,
                device=device,
                **data_sampler_args,
            )
            task = task_sampler(device=device, **task_sampler_args)
            ys = task.evaluate(xs)

            loss_func = task.get_training_metric()

            encoder_decoder_mode = "encoder" if args.model.family == "EncoderTF" else "decoder"
            moe_layers = getattr(args.model, "moe_layers", None)
            use_moe = getattr(args.model, 'use_moe', False) or (moe_layers is not None and len(moe_layers) > 0)

            if task_name == "sparse_linear_regression" and args.training.lasso_guided_opt:
                w_b = task.w_b
                if task.normalize_w:
                    w_b = w_b * task.scale / math.sqrt(task.sparsity)

                # xs, ys are already on GPU - no need for .cuda()
                loss, output, aux_loss = train_step(
                    model, xs, ys, optimizer, loss_func,
                    mode=encoder_decoder_mode,
                    lasso_guided_opt_lam=args.training.lasso_guided_opt_lam,
                    lasso_guided_opt_layer=args.training.lasso_guided_opt_layer,
                    lasso_guided_opt_token=args.training.lasso_guided_opt_token,
                    w_star=w_b,
                    use_moe=use_moe,
                )
            else:
                # xs, ys are already on GPU - no need for .cuda()
                loss, output, aux_loss = train_step(
                    model, xs, ys, optimizer, loss_func,
                    mode=encoder_decoder_mode,
                    use_moe=use_moe,
                )

            point_wise_tags = list(range(curriculum.n_points))
            point_wise_loss_func = task.get_metric()
            # ys is already on GPU
            point_wise_loss = point_wise_loss_func(output, ys).mean(dim=0)

            baseline_loss = (
                sum(
                    max(curriculum.n_dims_truncated - ii, 0)
                    for ii in range(curriculum.n_points)
                )
                / curriculum.n_points
            )

            if i % args.wandb.log_every_steps == 0 and not args.test_run:
                log_dict = {
                    f"{task_name}/overall_loss": loss,
                    f"{task_name}/excess_loss": loss / baseline_loss,
                    f"{task_name}/pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                }
                # Add MoE aux_loss if applicable
                if use_moe and aux_loss > 0:
                    log_dict[f"{task_name}/aux_loss"] = aux_loss
                wandb.log(log_dict, step=i)

            pbar.set_description(f"loss {loss}")

        # TASK FOR LOOP ENDS

        curriculum.update()

        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 1000
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.cuda()
    model.train()

    train(model, args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm", "EncoderTF", "llama_hf", "qwen_hf", "gemma_hf"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
