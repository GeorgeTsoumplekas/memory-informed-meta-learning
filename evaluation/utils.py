import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.inp import INP
from models.memory_inp import MemoryINP
import torch
from inp_config import Config as INPConfig
from memory_inp_config import Config as MemoryINPConfig
from models.loss import NLL
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import bootstrap
from sklearn.metrics import auc

EVAL_CONFIGS = {
    "test_num_z_samples": 32,
    "knowledge_dropout": 0,
    "batch_size": 25,
    "device": torch.device("cuda:{}".format(0)),
}


def _load_inp_model(config, save_dir, load_it="best"):
    print(save_dir)
    model = INP(config)
    model.to(config.device)
    model.eval()
    state_dict = torch.load(f"{save_dir}/model_{load_it}.pt")
    model.load_state_dict(state_dict)
    return model


def load_inp_model(save_dir, load_it="best"):
    config = INPConfig()
    config = config.from_toml(f"{save_dir}/config.toml")
    config.__dict__.update(EVAL_CONFIGS)
    model = _load_inp_model(config, save_dir, load_it)
    return model, config


def _load_memory_inp_model(config, save_dir, load_it="best"):
    print(save_dir)
    model = MemoryINP(config)
    model.to(config.device)
    model.eval()
    state_dict = torch.load(f"{save_dir}/model_{load_it}.pt")
    model.load_state_dict(state_dict)
    return model


def load_memory_inp_model(save_dir, load_it="best"):
    config = MemoryINPConfig()
    config = config.from_toml(f"{save_dir}/config.toml")
    config.__dict__.update(EVAL_CONFIGS)
    model = _load_memory_inp_model(config, save_dir, load_it)
    return model, config


def get_mask(k_type):
    if k_type == "a":
        mask = torch.tensor([1, 0, 0]).reshape(3, 1)
    elif k_type == "b":
        mask = torch.tensor([0, 1, 0]).reshape(3, 1)
    elif k_type == "c":
        mask = torch.tensor([0, 0, 1]).reshape(3, 1)
    elif k_type == "abc":
        mask = torch.tensor([1, 1, 1]).reshape(3, 1)
    elif k_type == "ab":
        mask = torch.tensor([1, 1, 0]).reshape(3, 1)
    elif k_type == "ac":
        mask = torch.tensor([1, 0, 1]).reshape(3, 1)
    elif k_type == "bc":
        mask = torch.tensor([0, 1, 1]).reshape(3, 1)

    elif k_type == "a1":
        mask = torch.tensor([1, 0, 0, 0, 0]).reshape(5, 1)
    elif k_type == "a2":
        mask = torch.tensor([0, 1, 0, 0, 0]).reshape(5, 1)
    elif k_type == "b1":
        mask = torch.tensor([0, 0, 1, 0, 0]).reshape(5, 1)
    elif k_type == "b2":
        mask = torch.tensor([0, 0, 0, 1, 0]).reshape(5, 1)
    elif k_type == "w":
        mask = torch.tensor([0, 0, 0, 0, 1]).reshape(5, 1)

    elif k_type == "raw":
        mask = None

    return mask


def plot_predictions(
    ax,
    i,
    outputs,
    x_context,
    y_context,
    x_target,
    extras,
    color="C0",
    plot_true=True,
    num_z_samples=5,
):
    mean = outputs[0].mean[:, i].cpu()
    stddev = outputs[0].stddev[:, i].cpu()

    for j in range(min(mean.shape[0], num_z_samples)):
        ax.plot(x_target[i].flatten().cpu(), mean[j].flatten(), color=color, alpha=0.8)
        ax.fill_between(
            x_target[i].flatten().cpu(),
            (mean[j] - stddev[j]).flatten(),
            (mean[j] + stddev[j]).flatten(),
            alpha=0.1,
            color=color,
        )
    ax.scatter(
        x_context[i].flatten().cpu(),
        y_context[i].flatten().cpu(),
        color="black",
        zorder=10,
        s=15,
    )
    if plot_true:
        ax.plot(
            extras["x"][i].flatten().cpu(),
            extras["y"][i].flatten().cpu(),
            color="black",
            linestyle="--",
            alpha=0.8,
        )
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def uniform_sampler(num_targets, num_context):
    return np.random.choice(list(range(num_targets)), num_context, replace=False)


def get_summary_df(
    model_dict,
    config_dict,
    data_loader,
    eval_type_ls,
    model_names,
    num_context_ls,
    sampler=uniform_sampler,
):
    # Evaluate the models on different knowledge types
    loss = NLL()

    losses = {}
    outputs_dict = {}
    data_knowledge = {}

    for model_name in model_names:
        losses[model_name] = {}
        outputs_dict[model_name] = {}
        for eval_type in eval_type_ls:
            losses[model_name][eval_type] = {}
            outputs_dict[model_name][eval_type] = {}
            for num_context in num_context_ls:
                losses[model_name][eval_type][num_context] = []
                outputs_dict[model_name][eval_type][num_context] = []

    knowledge_ls = []
    y_target_ls = []
    extras_ls = []

    for model_name in model_names:
        model, config = model_dict[model_name], config_dict[model_name]

        for batch in data_loader:
            (x_context, y_context), (x_target, y_target), knowledge, extras = batch
            knowledge_ls.append(knowledge)
            y_target_ls.append(y_target)
            extras_ls.append(extras)
            x_context = x_context.to(config.device)
            y_context = y_context.to(config.device)
            x_target = x_target.to(config.device)
            y_target = y_target.to(config.device)

            sample_idx = sampler(x_target.shape[1], max(num_context_ls))

            for _ in range(3):
                for num_context in num_context_ls:
                    x_context = x_target[:, sample_idx[:num_context], :]
                    y_context = y_target[:, sample_idx[:num_context], :]

                    for eval_type in eval_type_ls:
                        with torch.no_grad():
                            if eval_type == "raw":
                                outputs = model(
                                    x_context,
                                    y_context,
                                    x_target,
                                    y_target=y_target,
                                    knowledge=None,
                                )
                            elif config.use_knowledge:
                                if eval_type == "informed":
                                    outputs = model(
                                        x_context,
                                        y_context,
                                        x_target,
                                        y_target=y_target,
                                        knowledge=knowledge,
                                    )
                                else:
                                    mask = get_mask(eval_type)
                                    outputs = model(
                                        x_context,
                                        y_context,
                                        x_target,
                                        y_target=y_target,
                                        knowledge=knowledge * mask,
                                    )
                            else:
                                continue
                            outputs = tuple(
                                [
                                    o.cpu() if isinstance(o, torch.Tensor) else o
                                    for o in outputs
                                ]
                            )
                            loss_value = loss.get_loss(
                                outputs[0], outputs[1], outputs[2], outputs[3], y_target
                            )
                            losses[model_name][eval_type][num_context].append(
                                loss_value
                            )
                            outputs_dict[model_name][eval_type][num_context].append(
                                {
                                    "outputs": outputs,
                                    "x_context": x_context.cpu(),
                                    "y_context": y_context.cpu(),
                                    "x_target": x_target.cpu(),
                                    "y_target": y_target.cpu(),
                                    "knowledge": knowledge,
                                }
                            )

    loss_summary = {}
    for model_name in model_names:
        loss_summary[model_name] = {}
        for eval_type in eval_type_ls:
            loss_summary[model_name][eval_type] = {}
            for num_context in num_context_ls:
                loss_summary[model_name][eval_type][num_context] = {}
                loss_values = losses[model_name][eval_type][num_context]
                if len(loss_values) == 0:
                    loss_summary[model_name][eval_type][num_context]["mean"] = np.nan
                    loss_summary[model_name][eval_type][num_context]["std"] = np.nan
                else:
                    loss_values = torch.concat(loss_values, dim=0)
                    loss_summary[model_name][eval_type][num_context]["median"] = (
                        torch.median(loss_values).item()
                    )
                    loss_summary[model_name][eval_type][num_context]["mean"] = (
                        torch.mean(loss_values).item()
                    )
                    loss_summary[model_name][eval_type][num_context]["std"] = torch.std(
                        loss_values
                    ).item()

    summary_df = pd.DataFrame()
    for model_name in model_names:
        for eval_type in eval_type_ls:
            df = pd.DataFrame().from_dict(
                loss_summary[model_name][eval_type], orient="index"
            )
            df["num_context"] = df.index
            df["model_name"] = model_name
            df["eval_type"] = eval_type
            summary_df = pd.concat([summary_df, df])

    return summary_df, losses, outputs_dict


def get_uncertainties(
    outputs_dict, num_context_ls, knowledge_type_ls, model_name="INP", n_batches=-1
):
    if n_batches == -1:
        n_batches = len(
            outputs_dict[model_name][knowledge_type_ls[0]][num_context_ls[0]]
        )

    uncertainties = {}
    for num_context in num_context_ls:
        uncertainties[num_context] = {}
        for eval_type in knowledge_type_ls:
            uncertainties[num_context][eval_type] = {}
            for batch_idx in range(n_batches):
                uncertainties[num_context][eval_type][batch_idx] = []

    for num_context in num_context_ls:
        for eval_type in knowledge_type_ls:
            print("num_context:", num_context, "eval_type:", eval_type)
            for batch_idx in tqdm(range(n_batches)):
                mu = (
                    outputs_dict[model_name][eval_type][num_context][batch_idx][
                        "outputs"
                    ][0]
                    .mean[:, :, :]
                    .cpu()
                )  # [num_z_samples, bs, num_targets, 1]
                sigma = (
                    outputs_dict[model_name][eval_type][num_context][batch_idx][
                        "outputs"
                    ][0]
                    .stddev[:, :, :]
                    .cpu()
                )  # [num_z_samples, bs, num_targets, 1]

                # predictive uncertainty = MC estimate of int p(y|x, D) log p(y | x, D)dy
                dy = torch.linspace(-4, 4, 120)
                # p(dy | x, D) = N(dy ; mu, sigma^2)

                p_y = np.exp(-0.5 * (dy - mu) ** 2 / sigma**2) / np.sqrt(
                    2 * np.pi * sigma**2
                )
                p_y = p_y.mean(axis=0)

                p_U = -torch.sum(p_y * torch.log(p_y + 1e-8), axis=-1)

                # aleoric uncertainty = entropy of the normal with sigma^2
                a_U = (0.5 * np.log(2 * np.pi * np.e * sigma**2)).mean(axis=0).squeeze()

                # epistemic Uncertainty = predictive Uncertainty - aleatoric Uncertainty
                e_U = p_U - a_U

                uncertainties[num_context][eval_type][batch_idx] = {
                    "aleatoric": a_U,
                    "epistemic": e_U,
                    "predictive": p_U,
                }

    return uncertainties


def get_auc_summary(losses, model_name, eval_type_ls, num_context_ls):
    auc_summary = {}
    auc_values = {}
    improvement = {}

    for eval_type in eval_type_ls:
        base = (
            -torch.stack(
                [
                    torch.concat(losses[model_name]["raw"][num_context])
                    for num_context in num_context_ls
                ]
            )
            .cpu()
            .numpy()
        )
        informed = (
            -torch.stack(
                [
                    torch.concat(losses[model_name][eval_type][num_context])
                    for num_context in num_context_ls
                ]
            )
            .cpu()
            .numpy()
        )
        N = base.shape[-1]
        # estimate the area under the curve with the trapezoidal rule
        base_auc = np.array([auc(num_context_ls, base[:, i]) for i in range(N)])
        informed_auc = np.array([auc(num_context_ls, informed[:, i]) for i in range(N)])
        improvement[eval_type] = (
            [
                ((informed[i, :] - base[i, :]) / -base[i, :]).mean()
                for i in range(len(num_context_ls))
            ],
            [
                bootstrap(
                    ((informed[i, :] - base[i, :]) / -base[i, :],),
                    np.mean,
                    confidence_level=0.9,
                ).standard_error
                for i in range(len(num_context_ls))
            ],
        )
        auc_values[eval_type] = (informed_auc - base_auc) / -base_auc
        auc_summary[eval_type] = (
            np.mean(auc_values[eval_type]),
            bootstrap(
                (auc_values[eval_type],), np.mean, confidence_level=0.9
            ).standard_error,
        )

    return auc_summary, improvement


def combine_dictionaries(dict1_path, dict2_path, merged_dict_path):

    with open(dict1_path, "r") as f:
        dict1 = json.load(f)

    with open(dict2_path, "r") as f:
        dict2 = json.load(f)

    # Convert third level keys to integers in both dictionaries
    for d in [dict1, dict2]:
        for key1 in d:
            for key2 in d[key1]:
                d[key1][key2] = {int(k): v for k, v in d[key1][key2].items()}

    merged_dict = dict1.copy()

    for key1 in dict2:
        if key1 not in merged_dict:
            merged_dict[key1] = dict2[key1]
        else:
            for key2 in dict2[key1]:
                if key2 not in merged_dict[key1]:
                    merged_dict[key1][key2] = dict2[key1][key2]
                else:
                    merged_dict[key1][key2].update(dict2[key1][key2])

    with open(merged_dict_path, "w") as f:
        json.dump(merged_dict, f)

    # Convert the loaded losses back to tensors
    for model_name in merged_dict:
        for eval_type in merged_dict[model_name]:
            for num_context in merged_dict[model_name][eval_type]:
                # Convert list to tensor
                merged_dict[model_name][eval_type][num_context] = [
                    torch.tensor(loss) for loss in merged_dict[model_name][eval_type][num_context]
                ]

    return merged_dict
