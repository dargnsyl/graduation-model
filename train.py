from typing import overload
import torch
from numpy.lib import save
from util import Logger, accuracy, TotalMeter
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from util.prepossess import mixup_criterion, mixup_data
from util.loss import mixup_cluster_loss
from sklearn.metrics import roc_auc_score, confusion_matrix
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicTrain:

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = train_config['epochs']
        self.optimizers = optimizers
        self.best_acc = 0
        self.best_model = None
        self.best_acc_val = 0
        self.best_auc_val = 0
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.group_loss = train_config['group_loss']

        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']
        self.topo_reg_loss_weight = train_config.get('topo_reg_loss_weight', 1.0)
        self.additive_reg_weight = train_config.get("additive_reg_weight", 1e-3)
        self.moe_balance_loss_weight = train_config.get("moe_balance_loss_weight", 1e-2)
        self.moe_entropy_loss_weight = train_config.get("moe_entropy_loss_weight", 1e-3)

        self.save_path = log_folder
        self.save_model_path = Path("save_model")
        self.run_time_tag = None

        self.save_learnable_graph = True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy, \
            self.val_accuracy, self.test_accuracy, self.edges_num = [
            TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()

    def train_per_epoch(self, optimizer):

        self.model.train()

        for data_in, pearson, label, _ in self.train_dataloader:

            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)
##Mixup 数据增强
            inputs, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 1, device)
            targets_a = targets_a.long()
            targets_b = targets_b.long()

            output, learnable_matrix, edge_variance = self.model(inputs, nodes)#output[16,2]，learnable_matrix就是论文中的矩阵A[16,200,200]

            loss = 2 * mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            if self.group_loss:
                loss += mixup_cluster_loss(learnable_matrix,
                                           targets_a, targets_b, lam)

            if self.sparsity_loss:
                sparsity_loss = self.sparsity_loss_weight * \
                                torch.norm(learnable_matrix, p=1)
                loss += sparsity_loss

            topo_reg = getattr(self.model, 'topo_reg_loss', None)
            if topo_reg is not None:
                loss += self.topo_reg_loss_weight * topo_reg

            add_reg = getattr(self.model, "additive_kernel_loss", None)
            if add_reg is not None:
                loss += self.additive_reg_weight * add_reg

            moe_balance = getattr(self.model, "moe_balance_loss", None)
            if moe_balance is not None:
                loss += self.moe_balance_loss_weight * moe_balance

            moe_entropy = getattr(self.model, "moe_entropy_loss", None)
            if moe_entropy is not None:
                loss -= self.moe_entropy_loss_weight * moe_entropy

            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            self.edges_num.update_with_weight(edge_variance, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for data_in, pearson, label, _ in dataloader:
            label = label.long()
            data_in, pearson, label = data_in.to(
                device), pearson.to(device), label.to(device)
            output, _, _ = self.model(data_in, pearson)

            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)
        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        con_matrix = confusion_matrix(labels, result)
        return [auc] + list(metric), con_matrix

    def generate_save_learnable_matrix(self):
        learable_matrixs = []

        labels = []

        for data_in, nodes, label, _ in self.test_dataloader:
            label = label.long()
            data_in, nodes, label = data_in.to(
                device), nodes.to(device), label.to(device)
            _, learable_matrix, _ = self.model(data_in, nodes)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results, txt):

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)
        with open(self.save_path / "training_info.txt", 'a', encoding='utf-8') as f:
            f.write(txt)

        local_model_name = f"model_{self.best_acc:.3f}%.pt"
        torch.save(self.best_model.state_dict(), self.save_path / local_model_name)

        self.save_model_path.mkdir(exist_ok=True, parents=True)
        run_tag = self.run_time_tag if self.run_time_tag is not None else datetime.now().strftime("%m-%d-%H-%M-%S")
        central_model_name = f"{run_tag}_{self.best_acc:.3f}%.pt"
        torch.save(self.best_model.state_dict(), self.save_model_path / central_model_name)

    def train(self):
        training_process = []
        txt = ''
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.reset_meters()                                  #重置计量�?
            self.train_per_epoch(self.optimizers[0])
            val_result, _ = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result, con_matrix = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            if self.best_acc <= self.test_accuracy.avg:
                self.best_acc = self.test_accuracy.avg
                self.best_model = self.model

            if (con_matrix[0][0] + con_matrix[1][0]) != 0:
                SEN = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[1][0])
            else:
                SEN = 0

            if (con_matrix[1][1] + con_matrix[0][1]) != 0:
                SPE = con_matrix[1][1] / (con_matrix[1][1] + con_matrix[0][1])
            else:
                SPE = 0

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train ACC:{self.train_accuracy.avg: .3f}%',
                f'Val ACC:{self.val_accuracy.avg: .2f}%',
                f'Val AUC:{val_result[0]:.2f}',
                f'Test ACC:{self.test_accuracy.avg: .2f}%',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test SEN:{SEN:.4f}',
                f'Test SPE:{SPE:.4f}',
                f'Test F1:{test_result[-4]:.4f}',
            ]))

            txt += f'Epoch[{epoch}/{self.epochs}] '+f'Train Loss:{self.train_loss.avg: .3f} '+f'Train ACC:{self.train_accuracy.avg: .3f}% '+f'Val ACC:{self.val_accuracy.avg: .3f}% '+ f'Val AUC:{val_result[0]:.3f} '+f'Test ACC:{self.test_accuracy.avg: .3f}% '+f'Test AUC:{test_result[0]:.4f} '+f'Test SEN:{SEN:.4f} '+f'Test SPE:{SPE:.4f} '+f'Test F1:{test_result[-4]:.4f}'+'\n'

            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result + test_result)
        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        self.run_time_tag = date_time
        self.save_path = self.save_path/Path(f"{self.best_acc: .3f}%_{date_time}")
        self.logger.info(" | ".join([
            f'Best_ACC[{self.best_acc}]'
        ]))
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process, txt)

