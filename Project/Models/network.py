import torch
import torch.optim as optim
from metrics import loss_function


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


low_year_train, max_year_train = 0, 6
low_hour_train, max_hour_train = 0, 8753
low_year_val, max_year_val = 0, 0
low_hour_val, max_hour_val = 0, 4379


def EarlyStopping(curr_monitor, old_monitor, count, patience=5, min_delta=0):
    stop = False
    if (curr_monitor - old_monitor) <= min_delta:
        count += 1
        if count > patience:
            stop = True
            return count, stop
    count = 0
    return count, stop


def lr_schedulers(Net_optimizer):
    Net_linearLR = optim.lr_scheduler.LinearLR(Net_optimizer, total_iters=5)
    Net_cos_annLR = optim.lr_scheduler.CosineAnnealingLR(Net_optimizer, T_max=45, eta_min=3.75e-4)
    return Net_linearLR, Net_cos_annLR


def check(Net, Net_func_in: list, time: list, latitude_weights):
    # Net_func_in = [train_data, utils.low_year_train, utils.max_year_train, lead_time]
    pred_Net, target_Net = Net(Net_func_in[0], Net_func_in[1], Net_func_in[2], Net_func_in[3], time)
    loss = loss_function(pred_Net, target_Net, latitude_weights)
    return loss


class Network:
    def __init__(self, Net, lead_time, train_data, validation_data, lat_weights, name):
        self.Net = Net
        self.name = name
        self.Net_optimizer = optim.AdamW(self.Net.parameters(), lr=5e-4, weight_decay=1e-5)
        self.Net_linearLR, self.Net_cos_annLR = lr_schedulers(self.Net_optimizer)
        self.low_year_train, self.max_year_train = low_year_train, max_year_train
        self.low_year_val, self.max_year_val = low_year_val, max_year_val
        self.lead_time = lead_time
        self.train_data = train_data
        self.validation_data = validation_data
        self.lat_weights = lat_weights
        self.done = False
        self.count = 0
        self.losses = []

    def reset(self):
        self.done = False
        self.count = 0
        self.losses = []

    def pre_steps(self):
        self.losses = []

    def step(self, time):
        if not self.done:
            self.losses.append(check(self.Net, [self.train_data, self.low_year_train, self.max_year_train,
                                                self.lead_time], time, self.lat_weights))

    def post_steps(self, time, epoch):
        # Optimization + EarlyStopping
        if not self.done:
            prev_loss = check(self.Net, [self.validation_data, self.low_year_val, self.max_year_val, 6], time,
                              self.lat_weights)
            # Optimization
            loss = torch.FloatTensor(self.losses).to(device)
            loss.requires_grad = True
            self.Net_optimizer.zero_grad()
            loss.mean().backward()
            self.Net_optimizer.step()
            # EarlyStopping
            curr_loss = check(self.Net, [self.validation_data, self.low_year_val, self.max_year_val, self.lead_time],
                              time, self.lat_weights)
            self.count, self.done = EarlyStopping(curr_loss, prev_loss, self.count)
            if self.done:
                print("{}_{} is done".format(self.name, self.lead_time))

        # Learning Rate update
        # Linear warmup schedule 'if epoch < 5' - Cosine-annealing warmup schedule 'else'
        if not self.done:
            if epoch < 5:
                self.Net_linearLR.step()
            else:
                self.Net_cos_annLR.step()

        return self.done
