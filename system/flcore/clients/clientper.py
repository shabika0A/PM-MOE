# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
import numpy as np
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from flcore.clients.clientbase import Client
from flcore.trainmodel.moe.moe import ToPMoE


class clientPer(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = max(1, np.random.randint(1, max_local_epochs // 2 + 1))

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def set_parameters(self, model):
        self.model.to(self.device)
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.to(self.device).clone()


class PMOE_clientPer(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.moe_fine_tuning_epochs = args.moe_fine_tuning_epochs
        self.trained_experts = None  # set later by server
        self.is_moe_finetune = False
        self.lock_experts = args.lock_experts

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = max(1, np.random.randint(1, max_local_epochs // 2 + 1))

        for _ in range(max_local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += time.time() - start_time

    def set_parameters(self, model):
        self.model.to(self.device)
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.to(self.device).clone()

    def set_moe_experts(self, fintuned_heads):
        self.trained_experts = fintuned_heads

    # after personalized pretrain, use MoE control heads for finetuning
    def moe_finetune(self):
        assert self.trained_experts is not None, "Call set_moe_experts() before moe_finetune()."

        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)
        self.model.train()
        self.is_moe_finetune = True

        # attach MoE (top-k) head
        self.model.moe = ToPMoE(
            trained_experts=self.trained_experts,
            gate_input_dim=self.trained_experts[0].in_features,
            args=self.args,
        ).to(self.device)

        # freeze base, unfreeze gate; optionally un/lock experts
        for p in self.model.parameters():
            p.requires_grad = False

        for p in self.model.moe.gating.parameters():
            p.requires_grad = True

        if self.lock_experts == 1:  # 0 -- lock, 1 -- unlock
            for p in self.model.moe.experts.parameters():
                p.requires_grad = True

        # optimizer for the (unfrozen) parameters
        self.moe_opt = torch.optim.SGD(filter(lambda q: q.requires_grad, self.model.parameters()),
                                       lr=self.learning_rate)

        for _ in range(self.moe_fine_tuning_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # representation from base
                rep = self.model.base(x)

                # optionally pass client id to gating (if used)
                try:
                    if hasattr(self.model, "moe") and hasattr(self.model.moe, "args"):
                        self.model.moe.args.id = int(getattr(self, "id", 0))
                except Exception:
                    pass

                output = self.model.moe(rep)
                loss = self.loss(output, y)

                self.moe_opt.zero_grad(set_to_none=True)
                loss.backward()
                self.moe_opt.step()

        self.train_time_cost["total_cost"] += time.time() - start_time

    def test_metrics(self):
        testloader = self.load_test_data()

        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if self.is_moe_finetune:
                    rep = self.model.base(x)
                    output = self.model.moe(rep)
                else:
                    output = self.model(x)

                preds = torch.argmax(output, dim=1)
                test_acc += (preds == y).sum().item()
                test_num += y.size(0)

                y_prob.append(output.detach().float().cpu().numpy())

                nc = self.num_classes + (1 if self.num_classes == 2 else 0)
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()

        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0.0

        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if self.is_moe_finetune:
                    rep = self.model.base(x)
                    output = self.model.moe(rep)
                else:
                    output = self.model(x)

                loss = self.loss(output, y)
                n = y.size(0)
                train_num += n
                losses += loss.item() * n

        return losses, train_num
