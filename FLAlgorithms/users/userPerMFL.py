import os
import torch
import torch.nn as nn
from FLAlgorithms.optimizers.fedoptimizer import pFedMeOptimizer
from FLAlgorithms.users.userbase import User
from tqdm import trange
import copy

# Implementation for pFedMe clients

class UserPerMFL(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, model_name, 
                batch_size, alpha, beta, lamda, local_epochs, dataset):
        super().__init__(device, numeric_id, train_data, test_data, model, batch_size=batch_size,
                         alpha=alpha, beta=beta, lamda=lamda, local_epochs=local_epochs)
        # print("model[1] :", model[1])
        # input("press :")
        if (model_name == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
            # print("model name :", model_name)
        elif model_name == "cnn" and dataset in ["FMnist", "Cifar100"]:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
            # self.loss = nn.CrossEntropyLoss()

        # Class-weighted loss, off by default so earlier runs stay comparable.
        # CLASS_WEIGHTS=1 sets each class weight to the inverse of its frequency
        # in this client's own training labels, normalised to mean one. Classes
        # absent from the client get weight zero, which is what NLLLoss and
        # CrossEntropyLoss both expect for an unseen class.
        #
        # Rationale: the loss optimises unweighted accuracy on an 81.7 per cent
        # benign corpus while macro F1 weights every class equally. The
        # objective and the reported metric disagree, which is a candidate
        # explanation for the global model collapsing to BENIGN.
        if int(os.environ.get("CLASS_WEIGHTS", "0")):
            n_cls = self.model(next(iter(self.trainloaderfull))[0][:1]).shape[1]
            counts = torch.zeros(n_cls)
            for _, y in self.trainloaderfull:
                counts += torch.bincount(y.view(-1).long(), minlength=n_cls).float()
            w = torch.where(counts > 0, counts.sum() / (counts * (counts > 0).sum()),
                            torch.zeros_like(counts))
            self.loss = type(self.loss)(weight=w.to(device))

        # self.K = K
        self.alpha = alpha
        self.optimizer = pFedMeOptimizer(self.model.parameters(), alpha=self.alpha, lamda=self.lamda)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs, team_model):
        team_model_list = copy.deepcopy(list(team_model))
        for epoch in range(0, epochs):  # local update
            # print(" at training :: Epoch [", epoch, "]")
            self.model.train()
            X, y = self.get_next_train_batch()
            # print("X :", X, "y :", y)

            # K = 30 # K is number of personalized steps
            #  for i in range(self.K):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            # print(loss)
            loss.backward()

            # personalized_model_bar is the weights that are generated after pFedMe operations in the client (lower)
            # level

            self.personalized_model_bar, _ = self.optimizer.step(team_model_list)

            # update local weight after finding aproximate theta. Copy the personalized_model_bar to the local model
            # of each user

            for new_param, localweight in zip(self.personalized_model_bar, self.local_model):
                # localweight.data = localweight.data - self.lamda * self.alpha * (
                #             localweight.data - new_param.data)
                localweight.data = new_param.data
        # update local model as local_weight_upated
        
        self.update_parameters(self.local_model)
