import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import wandb
import yaml


class Recorder:

    def __init__(self, cfg):
        self.cfg = cfg
        name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.dir = os.path.join("logs", name)
        os.makedirs(self.dir)
        self.model_dir = os.path.join(self.dir, "nn")
        os.mkdir(self.model_dir)
        self.writer = SummaryWriter(os.path.join(self.dir, "summaries"))
        if self.cfg["runner"]["use_wandb"]:
            wandb.init(
                project=self.cfg["basic"]["task"],
                dir=self.dir,
                name=name,
                notes=self.cfg["basic"]["description"],
                config=self.cfg,
            )

        self.episode_statistics = {}
        self.last_episode = {}
        self.last_episode["steps"] = []
        self.episode_steps = None
        self._term_cause_lists = {"contact": [], "vel": [], "height": [], "episode_timeout": []}
        self._term_primary_codes = []
        self._trunc_cmd_resample_means = []

        with open(os.path.join(self.dir, "config.yaml"), "w") as file:
            yaml.dump(self.cfg, file)

    def record_episode_statistics(
        self,
        done,
        ep_info,
        it,
        write_record=False,
        term_causes=None,
        term_primary=None,
        trunc_cmd_resample_mean=None,
    ):
        if self.episode_steps is None:
            self.episode_steps = torch.zeros_like(done, dtype=int)
        else:
            self.episode_steps += 1
        for val in self.episode_steps[done]:
            self.last_episode["steps"].append(val.item())
        self.episode_steps[done] = 0

        for key, value in ep_info.items():
            if self.episode_statistics.get(key) is None:
                self.episode_statistics[key] = torch.zeros_like(value)
            self.episode_statistics[key] += value
            if self.last_episode.get(key) is None:
                self.last_episode[key] = []
            for done_value in self.episode_statistics[key][done]:
                self.last_episode[key].append(done_value.item())
            self.episode_statistics[key][done] = 0

        log_term = self.cfg["runner"].get("log_termination_reasons", True)
        if log_term and term_causes is not None:
            for key, value in term_causes.items():
                for val in value[done]:
                    self._term_cause_lists[key].append(val.item())
        if log_term and term_primary is not None:
            for val in term_primary[done]:
                self._term_primary_codes.append(val.item())
        if log_term and trunc_cmd_resample_mean is not None:
            self._trunc_cmd_resample_means.append(float(trunc_cmd_resample_mean))

        if write_record:
            for key in self.last_episode.keys():
                path = ("" if key == "steps" or key == "reward" else "episode/") + key
                value = self._mean(self.last_episode[key])
                self.writer.add_scalar(path, value, it)
                if self.cfg["runner"]["use_wandb"]:
                    wandb.log({path: value}, step=it)
                self.last_episode[key].clear()

            if self.cfg["runner"].get("log_termination_reasons", True):
                for key, lst in self._term_cause_lists.items():
                    path = "termination/frac_{}".format(key)
                    v = self._mean(lst)
                    self.writer.add_scalar(path, v, it)
                    if self.cfg["runner"]["use_wandb"]:
                        wandb.log({path: v}, step=it)
                    lst.clear()
                codes = self._term_primary_codes
                primary_names = [(1, "contact"), (2, "vel"), (3, "height"), (4, "episode_timeout")]
                if len(codes) > 0:
                    n = len(codes)
                    for code, name in primary_names:
                        frac = sum(1 for c in codes if c == code) / n
                        path = "termination/primary/{}".format(name)
                        self.writer.add_scalar(path, frac, it)
                        if self.cfg["runner"]["use_wandb"]:
                            wandb.log({path: frac}, step=it)
                else:
                    for _, name in primary_names:
                        path = "termination/primary/{}".format(name)
                        self.writer.add_scalar(path, 0.0, it)
                        if self.cfg["runner"]["use_wandb"]:
                            wandb.log({path: 0.0}, step=it)
                self._term_primary_codes.clear()
                if len(self._trunc_cmd_resample_means) > 0:
                    tv = sum(self._trunc_cmd_resample_means) / len(self._trunc_cmd_resample_means)
                    path = "truncation/frac_cmd_resample"
                    self.writer.add_scalar(path, tv, it)
                    if self.cfg["runner"]["use_wandb"]:
                        wandb.log({path: tv}, step=it)
                    self._trunc_cmd_resample_means.clear()

    def record_statistics(self, statistics, it):
        for key, value in statistics.items():
            self.writer.add_scalar(key, float(value), it)
            if self.cfg["runner"]["use_wandb"]:
                wandb.log({key: float(value)}, step=it)

    def save(self, model_dict, it):
        path = os.path.join(self.model_dir, "model_{}.pth".format(it))
        print("Saving model to {}".format(path))
        torch.save(model_dict, path)

    def _mean(self, data):
        if len(data) == 0:
            return 0.0
        else:
            return sum(data) / len(data)
