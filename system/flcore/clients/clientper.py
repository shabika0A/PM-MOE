# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.trainmodel.moe.moe import ToPMoE
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
class clientPer(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to('cpu')
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to('cpu')
                y = y.to('cpu')
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
            self.model.to(self.device)
            old_param.data = new_param.data.clone()


class PMOE_clientPer(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.args = args
        self.moe_fine_tuning_epochs = args.moe_fine_tuning_epochs
        self.trained_experts = None  # moe experts
        self.is_moe_finetune = False
        self.lock_experts = args.lock_experts

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to('cpu')
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to('cpu')
                y = y.to('cpu')
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
            self.model.to(self.device)
            old_param.data = new_param.data.clone()
    
    
    def set_moe_experts(self, fintuned_heads):
        self.trained_experts = fintuned_heads
        
     # after personalized finetune use moe control heads
    def moe_finetune(self):
        # === grad preamble (auto) ===
        import torch
        self.model.train()
        # freeze all, then unfreeze MoE + head
        for p in self.model.parameters():
            p.requires_grad = False
        moe_names = []
        for name, p in self.model.named_parameters():
            if any(k in name.lower() for k in ('moe','expert','experts','gate','router','fc','classifier','head')):
                p.requires_grad = True
                moe_names.append(name)
        # if nothing matched, unfreeze parameters directly under self.model.moe (if present)
        if not moe_names and hasattr(self.model, 'moe'):
            for name, p in self.model.moe.named_parameters():
                p.requires_grad = True
                moe_names.append('moe.'+name)
        # rebuild optimizer from trainable params
        _moe_lr = float(getattr(getattr(self, 'args', self), 'moe_lr', getattr(self, 'learning_rate', 0.005)))
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        assert trainable, 'No trainable parameters for finetune'
        self.optimizer = torch.optim.SGD(trainable, lr=_moe_lr, momentum=0.9)
        # pass current client id into MoE args if available
        try:
            if hasattr(self.model, 'moe') and hasattr(self.model.moe, 'args'):
                self.model.moe.args.id = int(getattr(self, 'id', 0))
        except Exception:
            pass
        printed_check = False
        # Enable grads for MoE/FC; rebuild optimizer with moe_lr
        import torch
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = False
        for name, p in self.model.named_parameters():
            if ('moe' in name) or ('gate' in name) or ('experts' in name) or ('fc' in name) or ('head' in name):
                p.requires_grad = True
        _moe_lr = float(getattr(getattr(self, 'args', self), 'moe_lr', getattr(self, 'learning_rate', 0.005)))
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        assert trainable, 'No trainable parameters for finetune'
        self.optimizer = torch.optim.SGD(trainable, lr=_moe_lr, momentum=0.9)
        # Ensure MoE sees current client id
        try:
            if hasattr(self.model, 'moe') and hasattr(self.model.moe, 'args'):
                self.model.moe.args.id = int(getattr(self, 'id', 0))
        except Exception:
            pass
        # Enable grads for MoE/FC; rebuild optimizer with moe_lr
        import torch
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad = False
        for name, p in self.model.named_parameters():
            if ('moe' in name) or ('gate' in name) or ('experts' in name) or ('fc' in name) or ('head' in name):
                p.requires_grad = True
        _moe_lr = float(getattr(getattr(self, 'args', self), 'moe_lr', getattr(self, 'learning_rate', 0.005)))
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        assert trainable, 'No trainable parameters for finetune'
        self.optimizer = torch.optim.SGD(trainable, lr=_moe_lr, momentum=0.9)
        # Ensure MoE sees current client id
        try:
            if hasattr(self.model, 'moe') and hasattr(self.model.moe, 'args'):
                self.model.moe.args.id = int(getattr(self, 'id', 0))
        except Exception:
            pass
        trainloader = self.load_train_data()
        start_time = time.time()
        
        self.model.train() # MoeHeadSpilt
        
        self.is_moe_finetune = True
        
        assert self.trained_experts is not None
        
        self.model.moe = ToPMoE(trained_experts = self.trained_experts,
                                gate_input_dim=self.trained_experts[0].in_features, 
                                args = self.args).to(self.args.device)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
            # moe 
            for param in self.model.moe.gating.parameters():
                param.requires_grad = True
                
            # expert 
            if self.lock_experts == 1: # 0--lock  1--unlock
                for param in self.model.moe.experts.parameters():
                    param.requires_grad = True
        
        # reset optimi
        self.moe_opt= torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.args.moe_fine_tuning_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to('cpu')
                y = y.to('cpu')
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                                # ensure MoE knows the current client id
                try:
                    if hasattr(self.model, 'moe') and hasattr(self.model.moe, 'args'):
                        self.model.moe.args.id = int(getattr(self, 'id', 0))
                except Exception:
                    pass
                output = self.model.moe(rep)
                loss = self.loss(output, y)
                self.moe_opt.zero_grad()
                assert hasattr(loss, 'backward') and getattr(loss, 'requires_grad', False), 'loss must be a Tensor with grad'

                assert hasattr(loss, 'backward') and getattr(loss, 'requires_grad', False), 'loss must be a Tensor with grad'

                # === grad sanity (auto) ===

                if not printed_check:

                    req = getattr(loss, 'requires_grad', None)

                    any_trainable = any(p.requires_grad for p in self.model.parameters())

                    any_moe = any(p.requires_grad for n,p in self.model.named_parameters() if 'moe' in n.lower() or 'expert' in n.lower())

                    print('DBG finetune:', 'loss.req', req, 'any_trainable', any_trainable, 'any_moe', any_moe, 'n_trn', sum(int(p.requires_grad) for p in self.model.parameters()))

                    printed_check = True

                assert getattr(loss, 'requires_grad', False), 'loss must be a Tensor with grad'

                loss.backward()
                self.moe_opt.step()
                
        
        self.train_time_cost['total_cost'] += time.time() - start_time
    
    
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to('cpu')
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to('cpu')
                y = y.to('cpu')
                
                if self.is_moe_finetune == True:
                    # print("self.is_moe_finetune=", self.is_moe_finetune)
                    rep = self.model.base(x)
                    output = self.model.moe(rep)
                else:
                    output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to('cpu')
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to('cpu')
                y = y.to('cpu')
                
                if self.is_moe_finetune == True:
                    # print("self.is_moe_finetune=", self.is_moe_finetune)
                    rep = self.model.base(x)
                    output = self.model.moe(rep)
                else:
                    output = self.model(x)
                    
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    
