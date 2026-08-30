# import torch
from FLAlgorithms.users.userPerMFL import UserPerMFL
from utils.model_utils import read_data, read_user_data
from tqdm import trange
import copy
import os
import random
import h5py
import numpy as np
import torch
from datetime import date
import pandas as pd


# Implementation for PerMFL Server

class PerMFL():
    def __init__(self, 
                device, 
                dataset, 
                algorithm, 
                model, 
                batch_size, 
                alpha,
                beta, 
                lamda, 
                gamma,
                eta, 
                num_glob_iters, 
                local_epochs, 
                team_epochs,
                optimizer, 
                num_users, 
                model_name, 
                exp_no, 
                num_teams, 
                tot_users, 
                num_labels, 
                analysis,
                group_division,
                p_teams,
                cluster_cfg=None,
                lamda_team=None,
                weighted_agg=0):
        
        
        self.tot_group_samples = [ [] for _ in range(num_teams) ]
        self.tau = [ [] for _ in range(num_teams) ]
        self.model_name = model_name
        self.old_global_model = copy.deepcopy(model)
        self.w_bar = copy.deepcopy(model)
        self.theta_bar = copy.deepcopy(model)
        self.model = copy.deepcopy(model)  # initialize global model
        
        self.team = []
        for i in range(0,num_teams):
            self.team.append(copy.deepcopy(model))
        
        self.users = [[] for _ in range(num_teams)]
        self.participated_devices = []
        
        self.device_train_acc = [[] for _ in range(num_teams)]
        self.device_test_acc = [[] for _ in range(num_teams)]
        self.device_train_loss = [[] for _ in range(num_teams)]
        self.device_test_loss = [[] for _ in range(num_teams)]

        self.team_train_acc = [[] for _ in range(num_teams)]
        self.team_test_acc = [[] for _ in range(num_teams)]
        self.team_train_loss = [[] for _ in range(num_teams)]
        self.team_test_loss = [[] for _ in range(num_teams)]


        self.global_f1 = []
        self.per_f1 = []
        self.global_recall = []; self.per_recall = []
        self.global_fpr = []; self.per_fpr = []
        self.global_train_acc = []
        self.global_test_acc = [] 
        self.global_train_loss = []
        self.global_test_loss = [] 

        self.avg_per_train_acc = []
        self.avg_per_train_loss = []        
        self.avg_per_test_acc = []        
        self.avg_per_test_loss = []

        self.device = device
        self.dataset = dataset
        
        self.batch_size = batch_size
        
        self.total_train_samples = 0
        
        self.num_users = num_users
       
        # global iteration, team iterations, local iterations
        self.num_glob_iters = num_glob_iters
        self.team_iters = team_epochs
        self.local_epochs = local_epochs
    
        # hyper parameters

        self.beta = beta
        self.lamda = lamda
        # lamda does double duty in PerMFL: it is the device->team proximal
        # pull in the device update AND the team's weight on its members'
        # average in the team update. Those want opposite values -- low lamda
        # personalises well but leaves the global model advancing 0.135% per
        # round, high lamda trains the global model but collapses PM.
        # lamda_team decouples the second role. Defaults to lamda, so the
        # shipped behaviour is unchanged unless --lamda_team is given.
        self.lamda_team = lamda if lamda_team is None else lamda_team
        self.weighted_agg = weighted_agg
        self.gamma = gamma
        self.eta = eta
        self.alpha = alpha  # learning rate
        
        self.algorithm = algorithm
        self.group = num_teams
        
        self.exp_no = exp_no
        self.analysis = analysis

        self.p_teams = p_teams
        self.team_list = list(range(num_teams))
        

        
        # read_data(dataset): this function divide the dataset into non-iid manner.
        # into n number of clients.
        #  returns a list of list named data.
        """  data[0] has clients
          data[1] has clients in 2 different groups.
          data[2] training data of users
          data[3] test data of users
        print("dataset :", dataset)
        """
        # group_division 3 means derived teams. No loader can do that: teams are
        # assigned before training, so no client updates exist yet. Start from
        # the random partition and let the round loop recluster.
        self.cluster_cfg = cluster_cfg or {}
        self.derive_teams = (group_division == 3)
        self.recluster_log = []
        data = read_data(dataset, tot_users, num_labels, num_teams,
                         1 if group_division == 3 else group_division)
        if dataset == "Cicids":
            # day-domain of each client, ground truth for the ARI report
            from utils import cicids as _c
            self.cluster_cfg.setdefault("truth", getattr(_c.read_cicids_data, "client_domain", None))

        # this function divide the dataset into non-iid manner in to n clients
        # total_users = len(data[1][0])
        self.tot_users=tot_users
        self.groups = data[1]
        self.model_initializer()
        
        for grp in range(len(data[1])):
            total_users = len(data[1][grp])
            self.group_sample = 0
            for i in range(total_users):
                id, train, test = read_user_data(data[1][grp][i], data, dataset)
                user = UserPerMFL(device, id, train, test, model, model_name,
                                batch_size, alpha, beta, lamda, local_epochs, dataset)
                
                self.users[grp].append(user)
                self.total_train_samples += user.train_samples
                self.group_sample += user.train_samples
            self.tot_group_samples[grp] = self.group_sample
            print("Number of users in group [", grp, "] / total users:", total_users)
            print("Finished creating user server and team objects")
        
        for grp in range(num_teams):
            self.tau[grp] = self.tot_group_samples[grp]/self.total_train_samples

        print("tau :", self.tau)

        """
        Model initializing
        1. self.model = global model
        2. self.old_global_model = previous copy of global model
        3. self.w_bar = aggregated model before global update
        4. self.theta_bar = aggregated team model before team update
        5. self.team : list of team update
        """
        
    def _pooled_f1(self):
        """Macro F1 for the global model and for the personal models.

        Both pooled: every client's confusion matrix is summed before the F1
        is taken, so the score is over one combined test set rather than an
        average of per-client scores. Per-client averaging flatters
        personalisation, since each device is graded only on the classes it
        holds.
        """
        from FLAlgorithms.metrics import macro_f1, num_classes_of
        C = num_classes_of(self.model)
        cm_g = np.zeros((C, C), dtype=np.int64)
        cm_p = np.zeros((C, C), dtype=np.int64)
        for grp in range(self.group):
            for u in self.users[grp]:
                cm_g += u.confusion(self.model.parameters(), C)
        for u in (self.participated_devices or
                  [u for grp in range(self.group) for u in self.users[grp]]):
            cm_p += u.confusion_personalized(C)
        return macro_f1(cm_g)[0], macro_f1(cm_p)[0], cm_g, cm_p

    def per_client_report(self):
        """Per-client and per-team accuracy and macro F1 at the end of training.

        Pooled figures hide the structure here: a Monday client holding only
        BENIGN scores perfectly and masks clients that hold attacks.
        """
        from FLAlgorithms.metrics import macro_f1, num_classes_of
        C = num_classes_of(self.model)
        rows, team_rows = [], []
        for grp in range(self.group):
            cm_team = np.zeros((C, C), dtype=np.int64)
            for u in self.users[grp]:
                cm = u.confusion_personalized(C)
                cm_team += cm
                acc = np.trace(cm) / max(cm.sum(), 1)
                rows.append((int(u.id), grp, acc, macro_f1(cm)[0],
                             int(cm.sum()), int((cm.sum(1) > 0).sum())))
            team_rows.append((grp, np.trace(cm_team) / max(cm_team.sum(), 1),
                              macro_f1(cm_team)[0], int(cm_team.sum())))
        print("\n  per-client (personal model)")
        print("   %4s %5s %9s %9s %8s %8s" % ("cli", "team", "acc", "macroF1", "test_n", "classes"))
        for cid, g, a, f, n, k in sorted(rows):
            print("   %4d %5d %9.4f %9.4f %8d %8d" % (cid, g, a, f, n, k))
        acc = np.array([r[2] for r in rows]); f1 = np.array([r[3] for r in rows])
        print("   client acc  mean %.4f  peak %.4f  min %.4f" % (acc.mean(), acc.max(), acc.min()))
        print("   client F1   mean %.4f  peak %.4f  min %.4f" % (f1.mean(), f1.max(), f1.min()))
        print("  per-team (pooled within team)")
        for g, a, f, n in team_rows:
            print("   team %d  acc %.4f  macroF1 %.4f  n %d" % (g, a, f, n))
        ta = np.array([r[1] for r in team_rows]); tf = np.array([r[2] for r in team_rows])
        print("   team acc    mean %.4f  peak %.4f" % (ta.mean(), ta.max()))
        print("   team F1     mean %.4f  peak %.4f" % (tf.mean(), tf.max()))
        gf = np.array(self.global_f1); pf = np.array(self.per_f1)
        if len(gf):
            print("  pooled over rounds: GM F1 final %.4f peak %.4f | PM F1 final %.4f peak %.4f"
                  % (gf[-1], gf.max(), pf[-1], pf.max()))
        ga = np.array(self.global_test_acc); pa = np.array(self.avg_per_test_acc)
        if len(ga):
            print("  pooled over rounds: GM acc final %.4f peak %.4f | PM acc final %.4f peak %.4f"
                  % (ga[-1], ga.max(), pa[-1], pa.max()))
        return rows, team_rows

    def _recompute_tau(self):
        self.total_train_samples = 0
        for grp in range(self.group):
            self.tot_group_samples[grp] = sum(u.train_samples for u in self.users[grp])
            self.total_train_samples += self.tot_group_samples[grp]
        for grp in range(self.group):
            self.tau[grp] = self.tot_group_samples[grp] / self.total_train_samples

    def maybe_recluster(self, rnd):
        """CFMD-i trigger + MCTC formation. No-op unless --group_division 3."""
        if not self.derive_teams:
            return
        cfg = self.cluster_cfg
        if rnd < cfg.get("start", 1):
            return
        from FLAlgorithms.clustering.team_former import (
            client_signal, should_recluster, form_teams, agreement)

        flat, owner = [], []
        for grp in range(self.group):
            for u in self.users[grp]:
                flat.append(client_signal(u, self.team[grp], cfg.get("signal", "residual")))
                owner.append(u)

        fire, mx, mean = should_recluster(flat, cfg.get("eps_lo", float("inf")),
                                          cfg.get("eps_hi", 0.0))
        if not fire:
            self.recluster_log.append((rnd, False, mx, mean, None))
            return

        teams = form_teams(flat, self.group, pca_dim=cfg.get("pca_dim", 8))
        self.users = [[owner[i] for i in t] for t in teams]
        self._recompute_tau()

        ari = None
        truth = cfg.get("truth")
        if truth is not None:
            # teams index into `owner`, so ground truth must be reordered to
            # owner order. user.id is the client index for Emnist10 and Cicids.
            try:
                ari = agreement(teams, [truth[int(u.id)] for u in owner])
            except (IndexError, ValueError, TypeError):
                ari = None
        self.recluster_log.append((rnd, True, mx, mean, ari))
        if cfg.get("every", 1) == 0:      # form teams once, then hold fixed
            self.derive_teams = False
        print("  [recluster] round %d  max=%.4f mean=%.4f  sizes=%s  ARI=%s"
              % (rnd, mx, mean, [len(t) for t in teams],
                 "n/a" if ari is None else round(ari, 4)))

    def model_initializer(self):
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio

    def send_parameters(self, grp):
        assert (self.users[grp] is not None and len(self.users[grp]) > 0)
        # print(type(self.model))
        # print("line 178:", self.selected_users)
        for user in self.users[grp]:
            # print(user)
            user.set_parameters(self.team[grp])
         
    def add_parameters(self, user, ratio):

        for theta_bar_param, user_param in zip(self.theta_bar.parameters(), user.get_parameters()):
            theta_bar_param.data = theta_bar_param.data + user_param.data * ratio

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.theta_bar.parameters():
            param.data = torch.zeros_like(param.data)
        # print(self.theta_bar.parameters())
        total_train = 0
        # if(self.num_users = self.to)
        # print("line 115 :",self.selected_users)
        for user in self.selected_users:
            
            # print("user.train_samples :",user.train_samples)
            total_train += user.train_samples
            # print(user.train_samples)
        # print(total_train)
        # input("press")
        for user in self.selected_users:
            # Uniform is what Algorithm 1 line 8 specifies. Sample-weighted is
            # what FedAvg and every FL-IDS method surveyed (ClusterFed,
            # Fed-ANIDS, Stones From Other Hills) uses. --weighted_agg 1
            # selects the latter.
            if self.weighted_agg:
                self.add_parameters(user, user.train_samples / max(total_train, 1))
            else:
                self.add_parameters(user, 1 / self.num_users)

    def set_old_global_parameter(self):
        for old_param, global_param in zip(self.old_global_model.parameters(), self.model.parameters()):
            old_param.data = global_param.data.clone()

    def set_team_parameter(self, i):  # w^(t,0) = x^t
        for team_param, global_param in zip(self.team[i].parameters(), self.old_global_model.parameters()):
            team_param.data = global_param.data.clone()

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def select_users(self, round, num_users, grp):
        # selects num_clients clients weighted by number of samples from possible_clients
        # self.selected_users = []
        
        if num_users == len(self.users[grp]):
            # print("All users are selected")
            # print(self.users[grp])
            return self.users[grp]
        elif  num_users < len(self.users[grp]):
         
            # self.selected_users =  random.sample(self.users[grp], num_users)
            # return self.selected_users
            num_users = min(num_users, len(self.users[grp]))
            np.random.seed(round)
            return np.random.choice(self.users[grp], num_users, replace=False)  # , p=pk)

        else: 
            print("num_users :",num_users)
            print(" size of user per group :",len(self.users[grp]))
            assert (self.selected_users > len(self.users))
            # print("number of selected users are greater than total users")
        
        # 
    # define function for personalized agegatation.
    """def personalized_update_parameters(self, user, ratio):
        # only argegate the local_weight_update
        for theta_bar_param, user_param in zip(self.theta_bar.parameters(), user.local_weight_updated):
            theta_bar_param.data = theta_bar_param.data + user_param.data.clone() * ratio
    """

    #  global_param = x^t
    #  w_team = w^t
    def personalized_team_aggregate_parameters(self, grp):
        assert (self.users is not None and len(self.users) > 0)

        for team_param, old_glob_param, theta_bar_param in zip(self.team[grp].parameters(),
                                                                self.old_global_model.parameters(),
                                                                self.theta_bar.parameters()):
            lt = self.lamda_team
            team_param.data = (1 - self.eta * lt - self.eta * self.gamma) * team_param.data + self.eta * self.gamma * \
                            old_glob_param.data + lt * self.eta * theta_bar_param.data

    """
    calculate w_bar
    """
    def server_level_aggregate_parameters(self):
        # for param in self.w_bar.parameters():
        # param.data = torch.zeros_like(param.data)

        # aggregate average model with previous model using parameter beta

        for param in self.w_bar.parameters():
            param.data = torch.zeros_like(param.data)
        ratio = float(1/self.group)
        for grp in range(len(self.team)):
            if self.weighted_agg:
                ratio = self.tau[grp]        # sample-proportional, already computed
            
        #    for param, team_param in zip(self.w_bar.parameters(), self.team[grp].parameters()):
        #        param.data = param.data + self.tau[grp] * team_param.data
            for param, team_param in zip(self.w_bar.parameters(), self.team[grp].parameters()):
                param.data = param.data + ratio * team_param.data
        
    """
     compute global model
    """
    def global_update(self):

        for pre_param, param, w_param in zip(self.old_global_model.parameters(), self.model.parameters(), self.w_bar.parameters()):
            param.data = (1 - self.beta * self.gamma) * pre_param.data + self.beta * self.gamma * w_param.data

    # Save loss, accurancy to h5 file
    def save_results(self):
        today = date.today()
        d1 = today.strftime("%d_%m_%Y")
        alg = self.dataset + "_" + self.algorithm
        print("exp_no ", self.exp_no)
        #alg = "team_per_round" + "_" + str(self.p_teams) + "_" +  str(self.num_users) +  "_" + str(self.model_name) + "_" + "exp_no_" + str(self.exp_no) + "_TE_" + str(self.team_iters)
        #alg = "num_teams_" + str(self.group) + "_num_teams_" + str(self.p_teams) + "_num_users_" + str(self.num_users) + "_TE_" + str(self.team_iters)
        alg = str(self.exp_no) + "_lamdba_" + str(self.lamda) + "_gamma_" + str(self.gamma) +  "_beta_" + str(self.beta) + "_model_" + str(self.model_name) + "_dataset_" +str(self.dataset)
        
        #+ "_" + "alpha_" + str(self.alpha) + "_" + "beta_" + str(
        #    self.beta) + "_" + "lambda_" + str(self.lamda) + "_" + "gamma_" + str(self.gamma) +  "_" + "eta_" + str(self.eta) + \
        #    "_" + "num_teams_" + str(self.group) + "_" +  str(self.batch_size) + "b" + "_" + str(self.num_glob_iters) + "GE" + "_" + str(self.team_iters) + "TE" + \
        #     "_" + str(self.local_epochs) + "LE" + '_' + d1
        print(alg)
        # if self.algorithm == "pFedMe":
        #    alg = alg + "_" + str(self.alpha)
        # alg = alg + "_" + str(self.times)
        # Create a directory named "new_dir"
        directory_name = self.algorithm + "/" + self.dataset + "/" + str(self.model_name)  + "/" + self.analysis + "/" + str(self.p_teams)
        # Check if the directory already exists
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)

        with h5py.File("./results/"+ directory_name + "/" + '{}.h5'.format(alg, self.local_epochs), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('alpha', data=self.alpha) 
            hf.create_dataset('beta', data=self.beta)
            hf.create_dataset('gamma', data=self.gamma)
            hf.create_dataset('lambda', data=self.lamda)
            # defect 21: these two decided the experiment and were never recorded
            hf.create_dataset('lambda_team', data=self.lamda_team)
            hf.create_dataset('weighted_agg', data=int(self.weighted_agg))
            hf.create_dataset('tot_users', data=int(sum(len(g) for g in self.users)))
            hf.create_dataset('num_teams', data=int(self.group))
            hf.create_dataset('p_teams', data=int(self.p_teams))
            hf.create_dataset('num_users_per_round', data=int(self.num_users))
            hf.create_dataset('eta', data=self.eta)
            # hf.create_dataset('tau', data=self.tau)
            hf.create_dataset('global_rounds', data=self.num_glob_iters)
            hf.create_dataset('team_rounds', data=self.team_iters)
            hf.create_dataset('local_rounds', data=self.local_epochs)
            
            hf.create_dataset('per_train_accuracy', data=self.avg_per_train_acc)
            hf.create_dataset('per_train_loss', data=self.avg_per_train_loss)
            hf.create_dataset('per_test_accuracy', data=self.avg_per_test_acc)
            hf.create_dataset('per_test_loss', data=self.avg_per_test_loss)

            hf.create_dataset('global_train_accuracy', data=self.global_train_acc)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_test_accuracy', data=self.global_test_acc)
            hf.create_dataset('global_test_loss', data=self.global_test_loss)
            hf.create_dataset('global_macro_f1', data=self.global_f1)
            hf.create_dataset('per_macro_f1', data=self.per_f1)
            hf.create_dataset('global_macro_recall', data=self.global_recall)
            hf.create_dataset('per_macro_recall', data=self.per_recall)
            hf.create_dataset('global_macro_fpr', data=self.global_fpr)
            hf.create_dataset('per_macro_fpr', data=self.per_fpr)
            # final-round confusion matrices, [true, predicted]
            if getattr(self, 'last_cm_g', None) is not None:
                hf.create_dataset('cm_global', data=self.last_cm_g)
                hf.create_dataset('cm_personal', data=self.last_cm_p)

            # hf.create_dataset('team_train_accuracy', data=self.global_train_acc)
            # hf.create_dataset('team_train_loss', data=self.global_train_loss)
            # hf.create_dataset('team_test_accuracy', data=self.global_test_acc)
            # hf.create_dataset('team_test_loss', data=self.global_test_loss)
            
            
            hf.close()

    def test_server(self):
        num_samples = []
        tot_correct = []
        losses = []
        user = []

        for grp in range(self.group):
            user += self.users[grp]
        
        for c in user:
            ct, ls,  ns = c.test(self.model.parameters())
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(ls)
            # ids = [c.id for c in user]
        return num_samples, tot_correct, losses

    def test_server_level_aggregation(self):
        num_samples = []
        tot_correct = []
        losses = []
        user = []

        for grp in range(2):
            user += self.users[grp]
        #  print(user)
        for c in user:
            ct, ns = c.test(self.w_bar.parameters())
            # print("+_+_+_+_")
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            # :w
            # print("user", c.id, "accuracy:", ((ct * 1.0)/ns)*100)
            ids = [c.id for c in user]
        # print(tot_correct)
        # print(num_samples)
        # print("accuracy :", np.sum(tot_correct) / np.sum(num_samples))

        return ids, num_samples, tot_correct
    

    def test_thetabar(self, grp):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        i = 0
        for c in self.users[grp]:
            ct, ns = c.test_thetabar(self.theta_bar.parameters())
            tot_correct.append(ct * 1.0)
            
            num_samples.append(ns)

            # print("user id : ", i, "accuracy :", ct / ns)
            i += 1
        ids = [c.id for c in self.users[grp]]

        return ids, num_samples, tot_correct


    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for grp in range(self.group):
            for c in self.users[grp]:
                ct, cl, ns = c.train_error_and_loss(self.model.parameters())
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
                losses.append(cl * 1.0)

       #  ids = [c.id for c in self.users[grp]]
        
        return num_samples, tot_correct, losses

    def test_personalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        tot_loss = []
        i = 0
        for c in self.participated_devices:
            ct, ls, ns = c.test_personalized_model()
            tot_correct.append(ct * 1.0)
            tot_loss.append(ls * 1.0)
            num_samples.append(ns)
            i += 1
        print("Line 360 server base num_participated_device_test :",i)
        # ids = [c.id for c in self.users[grp]]

        return num_samples, tot_correct, tot_loss

    
    def test_personalized_team_model(self, grp):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        tot_loss = []
        i = 0
        for c in self.users[grp]:
            ct, ls, ns = c.test_personalized_team_model(self.team[grp].parameters())
            
            tot_correct.append(ct * 1.0)
            tot_loss.append(ls * 1.0)
            num_samples.append(ns)

            # print("user id : ", i, "accuracy :", ct / ns)
            i += 1
        ids = [c.id for c in self.users[grp]]

        return ids, num_samples, tot_correct, tot_loss

    def train_error_and_loss_personalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.participated_devices:
            ct, cl, ns = c.train_error_and_loss_personalized_model()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        # ids = [c.id for c in self.users[grp]]
        # groups = [c.group for c in self.clients]

        return num_samples, tot_correct, losses

    def train_error_and_loss_personalized_team_model(self, grp):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users[grp]:
            ct, cl, ns = c.train_error_and_loss_personalized_team_model(self.team[grp].parameters())
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.users[grp]]
        # groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def client_level_evaluate(self, grp):
        # print("stats test")
        stats = self.test_thetabar(grp)
        thetabar_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        if grp == 0:
            self.client_team_0_test_acc.append(thetabar_acc)
        elif grp == 1:
            self.client_team_1_test_acc.append(thetabar_acc)
        else:
            print("Number of group exceeds two")

        print("client level Accurancy: ", thetabar_acc)


    """
    Global model evaluation

    """
    def evaluate(self):

        per_device_train_acc = 0
        per_device_train_loss = 0
        per_device_test_acc = 0
        per_device_test_loss = 0

        
        eval_device =  self.evaluate_personalized_model()
        # eval_team =  self.evaluate_personalized_team_model(grp)
        # print(eval_device)
        # per_device_train_acc += float(eval_device[0])
        # per_device_train_loss += float(eval_device[1])
        # per_device_test_acc += float(eval_device[2])
        # per_device_test_loss += float(eval_device[3])

        """
        eval_device[0] = train accuracy
        eval_device[1] = train loss
        eval_device[2] = test accuracy
        eval_device[3] = test loss
        
        """

        print("Average Personal Train Accuracy: ", eval_device[0])
        print("Average Personal Train Loss: ", eval_device[1])
        print("Average Personal Test Accuracy: ", eval_device[2])
        print("Average Personal Test Loss: ", eval_device[3])
        
        self.avg_per_train_acc.append(eval_device[0])
        self.avg_per_train_loss.append(eval_device[1])
        self.avg_per_test_acc.append(eval_device[2])
        self.avg_per_test_loss.append(eval_device[3])
        


        # print("stats test")
        stats_test = self.test_server()
        # print(stats)
        # print("stats train")
        stats_train = self.train_error_and_loss()
        test_acc = np.sum(stats_test[1]) * 1.0 / np.sum(stats_test[0])
        train_acc = np.sum(stats_train[1]) * 1.0 / np.sum(stats_train[0])
        test_loss = sum([x * y for (x, y) in zip(stats_test[0], stats_test[2])]).item() / np.sum(stats_test[0])
        train_loss = sum([x * y for (x, y) in zip(stats_train[0], stats_train[2])]).item() / np.sum(stats_train[0])


        self.global_train_acc.append(train_acc)
        self.global_train_loss.append(train_loss)
        self.global_test_acc.append(test_acc)
        self.global_test_loss.append(test_loss)


        
        
        gf1, pf1, cmg, cmp_ = self._pooled_f1()
        # keep the last matrices so save_results can persist them; defect B1
        self.last_cm_g, self.last_cm_p = cmg, cmp_
        self.global_f1.append(gf1)
        self.per_f1.append(pf1)
        from FLAlgorithms.metrics import full_report
        rg, rp = full_report(cmg), full_report(cmp_)
        self.global_recall.append(rg["macro_recall"]); self.per_recall.append(rp["macro_recall"])
        self.global_fpr.append(rg["macro_fpr"]); self.per_fpr.append(rp["macro_fpr"])
        print("Global macro F1: %.4f   Personal macro F1: %.4f" % (gf1, pf1))
        print("Global macro recall: %.4f  FPR: %.4f   Personal macro recall: %.4f  FPR: %.4f"
              % (rg["macro_recall"], rg["macro_fpr"], rp["macro_recall"], rp["macro_fpr"]))
        print("Global Trainning Accurancy: ", train_acc)
        print("Global Trainning Loss: ", train_loss)
        print("Global test accurancy: ", test_acc)
        print("Global test_loss:",test_loss)
        
    """
    Personalized model evaluation

    """
    
    def evaluate_personalized_model(self):
        stats_test = self.test_personalized_model()
        stats_train = self.train_error_and_loss_personalized_model()
        test_acc = np.sum(stats_test[1]) * 1.0 / np.sum(stats_test[0])
        train_acc = np.sum(stats_train[1]) * 1.0 / np.sum(stats_train[0])
        train_loss = sum([x * y for (x, y) in zip(stats_train[0], stats_train[2])]).item() / np.sum(stats_train[0])
        test_loss = sum([x * y for (x, y) in zip(stats_test[0], stats_test[2])]).item() / np.sum(stats_test[0])
        
        
        # self.device_train_acc.append(train_acc)
        # self.device_train_loss.append(train_loss)
        # self.device_test_acc.append(test_acc)
        # self.device_test_loss.append(test_loss)
        
        # print("Average Personalized Trainning Accurancy: ", train_acc)
        # print("Average Personalized Trainning Loss: ", train_loss)
        # print("Average Personalized Testing Accurancy: ", test_acc)
        # print("Average Personalized Testing Loss: ", test_loss)

        return train_acc, train_loss, test_acc, test_loss

    """
    Personalized team model evaluation

    """
    def evaluate_personalized_team_model(self, grp):
        stats_test = self.test_personalized_team_model(grp)
        stats_train = self.train_error_and_loss_personalized_team_model(grp)
        test_acc = np.sum(stats_test[2]) * 1.0 / np.sum(stats_test[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        test_loss = sum([x * y for (x, y) in zip(stats_test[1], stats_test[3])]).item() / np.sum(stats_test[1])
        
        self.team_train_acc[grp].append(train_acc)
        self.team_train_loss[grp].append(train_loss)
        self.team_test_acc[grp].append(test_acc)
        self.team_test_loss[grp].append(test_loss)
        
        return train_acc, train_loss, test_acc, test_loss

    def evaluate_server_aggregation(self):
        stats = self.test_server_level_aggregation()
        glob_acc = np.sum(stats[2]) * 1.0 / np.sum(stats[1])
        
        self.global_test_acc.append(glob_acc)
        print("Server aggregation Test Accurancy: ", glob_acc)
        # print("Average Global Trainning Accurancy: ", train_acc)
        # print("Average Global Trainning Loss: ", train_loss)

    def plot_results(self):

        data_loss = {
        #'per_train_loss' : self.avg_per_train_loss[:500:50],
        'per_test_loss' : self.avg_per_test_loss[:500:50],
        #'global_train_loss' : self.global_train_loss[:500:50],
        'global_test_loss' : self.global_test_loss[:500:50]
        }

        data_acc = {
        #'per_train_acc' : self.avg_per_train_loss[:500:50],
        'per_test_acc' : self.avg_per_test_acc[:500:50],
        #'global_train_acc' : self.global_train_loss[:500:50],
        'global_test_acc' : self.global_test_acc[:500:50]
        }




        df_l = pd.DataFrame(data_loss)
        df_a = pd.DataFrame(data_acc)

       
        print(df_l)
        print(df_a)





    def train(self):

        for iters in trange(self.num_glob_iters):  # t = 1.... T-1
            list_devices = []
            self.set_old_global_parameter()  
            print("line 634:",self.team_list)  # x^(t-1) = x^t
            self.sub_group = random.sample(self.team_list, self.p_teams)
            #bprint("line 635 self.sub_group :", self.sub_group)
            for grp in (self.sub_group):  # for group 0 and 1
                # print("---Computing parameters for group :[", grp, "]")
                self.set_team_parameter(grp)

                
                for team_iter in range(self.team_iters):  # k = 1 .. K-1
                    # print("---exp_no[",self.exp_no,"] Global[",iters,"]---Team[", grp, "] rounds: ", team_iter + 1, " ---")

                    # send all parameter to the users of the group

                    

                    self.selected_users = self.select_users(team_iter, self.num_users, grp)
                    # print("line 676:",self.selected_users)
                    self.send_parameters(grp)
                    for user in self.selected_users:
                        # print("[", user, "]")
                        list_devices.append(user)
                        
                        user.train(self.local_epochs, self.team[grp].parameters())
                
                    self.aggregate_parameters()  # computing theta_bar done

                    self.personalized_team_aggregate_parameters(grp) #done 

                    # self.evaluate_personalized_team_model(grp)  #done
            
            self.participated_devices = list(set(list_devices))
            
            
            print(len(self.participated_devices))

            self.server_level_aggregate_parameters() # done
            
            self.global_update() #done

            self.evaluate()

            self.maybe_recluster(iters)
            
        self.per_client_report()
        self.save_results()
        self.plot_results()
        # self.save_model()
