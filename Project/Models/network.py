import statistics

from ..Utils.utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network:
    def __init__(self, Net, lat_weights, name, lead_time):
        self.Net = Net
        self.name = name
        self.lead_time = lead_time
        self.Net_optimizer = optim.AdamW(self.Net.parameters(), lr=5e-4, weight_decay=1e-5)
        self.Net_linearLR, self.Net_cos_annLR = lr_schedulers(self.Net_optimizer)
        self.low_year_train, self.max_year_train = low_year_train, max_year_train
        self.low_hour_train, self.max_hour_train = low_hour_train, max_hour_train
        self.low_year_val, self.max_year_val = low_year_val, max_year_val
        self.low_hour_val, self.max_hour_val = low_hour_val, max_hour_val
        self.lat_weights = lat_weights
        self.done = False
        self.count = 0
        self.loss = None
        self.validation_losses = []
        self.previous_validation_losses = self.validation_losses
        self.pred, self.targ = [], []

    def pre_steps(self):
        if not self.done:
            self.count = 0
            self.loss = None
            self.validation_losses = []
            self.previous_validation_losses = self.validation_losses

    def train_step(self, input_list, constants):
        # input_list = [data, six_hours_ago, twelve_hours_ago, target]
        if not self.done:
            self.loss = check(self.Net, input_list[0], input_list[1], input_list[2], input_list[3], constants,
                              self.lat_weights)

    def val_step(self, input_list, constants):
        # input_list = [data, six_hours_ago, twelve_hours_ago, target]
        if not self.done:
            loss = check(self.Net, input_list[0], input_list[1], input_list[2], input_list[3], constants,
                         self.lat_weights)
            self.validation_losses.append(loss.item())

    def post_steps(self, epoch):
        # Optimization
        if not self.done:
            loss = self.loss
            self.Net_optimizer.zero_grad()
            loss.mean().backward()
            self.Net_optimizer.step()
            # Learning Rate update
            # Linear warmup schedule if 'epoch < 5' - Cosine-annealing warmup schedule 'else'
            if epoch < 5:
                self.Net_linearLR.step()
                self.Net_optimizer.defaults['lr'] = self.Net_linearLR.get_last_lr()
            else:
                self.Net_cos_annLR.step()
                self.Net_optimizer.defaults['lr'] = self.Net_cos_annLR.get_last_lr()

    def stopping(self):
        if self.previous_validation_losses is None:
            self.count, self.done = 0, False
        else:
            self.count, self.done = EarlyStopping(statistics.mean(self.validation_losses),
                                                  statistics.mean(self.previous_validation_losses), self.count)
        if self.done:
            print("{}_{} is done".format(self.name, self.lead_time))

    def eval_step(self, data, six_hours_ago, twelve_hours_ago, target, constants):
        p = self.Net(data, six_hours_ago, twelve_hours_ago, target, constants, for_test=1)
        self.pred.append(p.detach().numpy()), self.targ.append(target.detach().numpy())
