from .Models.network import Network
from .Models.baselines import *
from .Models.ResNet import ResNet
from .Models.UNet import UNet
from .Models.ViT import ViT
from .utils.data_processing import *
from .utils.utils import *
from .utils.metrics import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Forecasting(nn.Module):
    def __init__(self, constants_set, train_data, validation_data, batch_size=128,
                 res_params=list[128, 28], u_params=list[64, 2], vit_params=list[8, 4, 2, 128]):
        super(Forecasting, self).__init__()
        self.device = device
        self.num_channels = 141
        self.out_channels = 3
        self.batch_size = None
        self.img_size = (self.height, self.width) = (32, 64)
        self.val_dim = len(validation_data)
        self.processor = PreProcessing()
        self.latitude_coordinates = latitude_coordinates
        self.latitude_weights = latitude_weighting_function(self.latitude_coordinates)
        self.constants = constants_set
        self.validation_data = validation_data
        train_years = max_year_train + 1 - low_year_train
        train_6 = [train_data[12:((8760*train_years) - 6)], train_data[6:((8760*train_years) - 12)],
                   train_data[0:((8760*train_years) - 18)], train_data[18:(8760*train_years)]]
        train_24 = [train_data[12:((8760*train_years) - 24)], train_data[6:((8760*train_years) - 30)],
                    train_data[0:((8760*train_years) - 36)], train_data[36:(8760*train_years)]]
        train_72 = [train_data[12:((8760*train_years) - 72)], train_data[6:((8760*train_years) - 78)],
                    train_data[0:((8760*train_years) - 84)], train_data[84:(8760*train_years)]]
        train_120 = [train_data[12:((8760*train_years) - 120)], train_data[6:((8760*train_years) - 126)],
                     train_data[0:((8760*train_years) - 132)], train_data[132:(8760*train_years)]]
        train_240 = [train_data[12:((8760*train_years) - 240)], train_data[6:((8760*train_years) - 246)],
                     train_data[0:((8760*train_years) - 252)], train_data[252:(8760*train_years)]]
        train_set = CustomDataset(train_6, train_24, train_72, train_120, train_240)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_6 = [validation_data[12:4374], validation_data[6:4368], validation_data[0:4362], validation_data[18:4380]]
        val_24 = [validation_data[12:4356], validation_data[6:4350], validation_data[0:4344], validation_data[36:4380]]
        val_72 = [validation_data[12:4308], validation_data[6:4302], validation_data[0:4296], validation_data[84:4380]]
        val_120 = [validation_data[12:4260], validation_data[6:4254], validation_data[0:4248],
                   validation_data[132:4380]]
        val_240 = [validation_data[12:4140], validation_data[6:4134], validation_data[0:4128],
                   validation_data[252:4380]]
        validation_set = CustomDataset(val_6, val_24, val_72, val_120, val_240)
        self.validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True)

        # Baselines
        self.Baseline_6 = Network(
            Baseline(self.num_channels,
                     self.out_channels,
                     self.processor),
            self.latitude_weights,
            'LinearRegression',
            6)
        self.Baseline_24 = Network(
            Baseline(self.num_channels,
                     self.out_channels,
                     self.processor),
            self.latitude_weights,
            'LinearRegression',
            24)
        self.Baseline_72 = Network(
            Baseline(self.num_channels,
                     self.out_channels,
                     self.processor),
            self.latitude_weights,
            'LinearRegression',
            72)
        self.Baseline_120 = Network(
            Baseline(self.num_channels,
                     self.out_channels,
                     self.processor),
            self.latitude_weights,
            'LinearRegression',
            120)
        self.Baseline_240 = Network(
            Baseline(self.num_channels,
                     self.out_channels,
                     self.processor),
            self.latitude_weights,
            'LinearRegression',
            240)
        # ResNets
        self.ResNet_6 = Network(
            ResNet(self.num_channels,
                   self.out_channels,
                   self.processor,
                   hidden_channels=res_params[0],
                   num_blocks=res_params[1]),
            self.latitude_weights,
            'ResNet',
            6)
        self.ResNet_24 = Network(
            ResNet(self.num_channels,
                   self.out_channels,
                   self.processor,
                   hidden_channels=res_params[0],
                   num_blocks=res_params[1]),
            self.latitude_weights,
            'ResNet',
            24)
        self.ResNet_72 = Network(
            ResNet(self.num_channels,
                   self.out_channels,
                   self.processor,
                   hidden_channels=res_params[0],
                   num_blocks=res_params[1]),
            self.latitude_weights,
            'ResNet',
            72)
        self.ResNet_120 = Network(
            ResNet(self.num_channels,
                   self.out_channels,
                   self.processor,
                   hidden_channels=res_params[0],
                   num_blocks=res_params[1]),
            self.latitude_weights,
            'ResNet',
            120)
        self.ResNet_240 = Network(
            ResNet(self.num_channels,
                   self.out_channels,
                   self.processor,
                   hidden_channels=res_params[0],
                   num_blocks=res_params[1]),
            self.latitude_weights,
            'ResNet',
            240)
        # UNets
        self.UNet_6 = Network(
            UNet(self.num_channels,
                 self.out_channels,
                 self.processor,
                 hidden_channels=u_params[0],
                 blocks=u_params[1]),
            self.latitude_weights,
            'UNet',
            6)
        self.UNet_24 = Network(
            UNet(self.num_channels,
                 self.out_channels,
                 self.processor,
                 hidden_channels=u_params[0],
                 blocks=u_params[1]),
            self.latitude_weights,
            'UNet',
            24)
        self.UNet_72 = Network(
            UNet(self.num_channels,
                 self.out_channels,
                 self.processor,
                 hidden_channels=u_params[0],
                 blocks=u_params[1]),
            self.latitude_weights,
            'UNet',
            72)
        self.UNet_120 = Network(
            UNet(self.num_channels,
                 self.out_channels,
                 self.processor,
                 hidden_channels=u_params[0],
                 blocks=u_params[1]),
            self.latitude_weights,
            'UNet',
            120)
        self.UNet_240 = Network(
            UNet(self.num_channels,
                 self.out_channels,
                 self.processor,
                 hidden_channels=u_params[0],
                 blocks=u_params[1]),
            self.latitude_weights,
            'UNet',
            240)
        # ViTs
        self.ViT_6 = Network(
            ViT(self.num_channels,
                self.out_channels,
                self.img_size,
                self.processor,
                embedding_dim=vit_params[3],
                depth=vit_params[0],
                num_heads=vit_params[1],
                prediction_depth=vit_params[2],
                hidden_dimension=vit_params[3]),
            self.latitude_weights,
            'ViT', 6)
        self.ViT_24 = Network(
            ViT(self.num_channels,
                self.out_channels,
                self.img_size,
                self.processor,
                embedding_dim=vit_params[3],
                depth=vit_params[0],
                num_heads=vit_params[1],
                prediction_depth=vit_params[2],
                hidden_dimension=vit_params[3]),
            self.latitude_weights,
            'ViT', 24)
        self.ViT_72 = Network(
            ViT(self.num_channels,
                self.out_channels,
                self.img_size,
                self.processor,
                embedding_dim=vit_params[3],
                depth=vit_params[0],
                num_heads=vit_params[1],
                prediction_depth=vit_params[2],
                hidden_dimension=vit_params[3]),
            self.latitude_weights,
            'ViT', 72)
        self.ViT_120 = Network(
            ViT(self.num_channels,
                self.out_channels,
                self.img_size,
                self.processor,
                embedding_dim=vit_params[3],
                depth=vit_params[0],
                num_heads=vit_params[1],
                prediction_depth=vit_params[2],
                hidden_dimension=vit_params[3]),
            self.latitude_weights,
            'ViT', 120)
        self.ViT_240 = Network(
            ViT(self.num_channels,
                self.out_channels,
                self.img_size,
                self.processor,
                embedding_dim=vit_params[3],
                depth=vit_params[0],
                num_heads=vit_params[1],
                prediction_depth=vit_params[2],
                hidden_dimension=vit_params[3]),
            self.latitude_weights,
            'ViT', 240)

    def train_forecasters(self, epochs=50):
        print("Start Training")
        for epoch in range(epochs):
            print("Epoch ", epoch)
            self.ResNet_6.pre_steps(), self.ResNet_24.pre_steps(), self.ResNet_72.pre_steps(),\
                self.ResNet_120.pre_steps(), self.ResNet_240.pre_steps()
            self.UNet_6.pre_steps(), self.UNet_24.pre_steps(), self.UNet_72.pre_steps(), self.UNet_120.pre_steps(),\
                self.UNet_240.pre_steps()
            self.ViT_6.pre_steps(), self.ViT_24.pre_steps(), self.ViT_72.pre_steps(), self.ViT_120.pre_steps(),\
                self.ViT_240.pre_steps()

            # New Part with Batch
            print("   Train")
            for num, (train_6, train_24, train_72, train_120, train_240) in enumerate(self.train_loader):
                # Train
                printProgressAction('    Batch', num)
                batch_dim = len(train_6[0])
                train_constants = np.array((self.constants,) * batch_dim)

                # Baselines
                self.Baseline_6.train_step(train_6, train_constants), \
                    self.Baseline_24.train_step(train_24, train_constants), \
                    self.Baseline_72.train_step(train_72, train_constants), \
                    self.Baseline_120.train_step(train_120, train_constants), \
                    self.Baseline_240.train_step(train_240, train_constants)
                # ResNets
                self.ResNet_6.train_step(train_6, train_constants),\
                    self.ResNet_24.train_step(train_24, train_constants),\
                    self.ResNet_72.train_step(train_72, train_constants),\
                    self.ResNet_120.train_step(train_120, train_constants),\
                    self.ResNet_240.train_step(train_240, train_constants)
                # UNets
                self.UNet_6.train_step(train_6, train_constants),\
                    self.UNet_24.train_step(train_24, train_constants),\
                    self.UNet_72.train_step(train_72, train_constants),\
                    self.UNet_120.train_step(train_120, train_constants),\
                    self.UNet_240.train_step(train_240, train_constants)
                # ViTs
                self.ViT_6.train_step(train_6, train_constants),\
                    self.ViT_24.train_step(train_24, train_constants),\
                    self.ViT_72.train_step(train_72, train_constants),\
                    self.ViT_120.train_step(train_120, train_constants),\
                    self.ViT_240.train_step(train_240, train_constants)
                # Optimization
                # ResNets
                self.ResNet_6.post_steps(epoch), self.ResNet_24.post_steps(epoch), self.ResNet_72.post_steps(epoch),\
                    self.ResNet_120.post_steps(epoch), self.ResNet_240.post_steps(epoch)
                # UNets
                self.UNet_6.post_steps(epoch), self.UNet_24.post_steps(epoch), self.UNet_72.post_steps(epoch),\
                    self.UNet_120.post_steps(epoch), self.UNet_240.post_steps(epoch)
                # ViTs
                self.ViT_6.post_steps(epoch), self.ViT_24.post_steps(epoch), self.ViT_72.post_steps(epoch),\
                    self.ViT_120.post_steps(epoch), self.ViT_240.post_steps(epoch)
            # Validation
            print("\n   Validation")
            for num, (val_6, val_24, val_72, val_120, val_240) in enumerate(self.validation_loader):
                printProgressAction('    Batch', num)
                batch_dim = len(val_6[0])
                val_constants = np.array((self.constants,) * batch_dim)
                # Baselines
                self.LinReg_Baseline_6.val_step(val_6, val_constants),\
                    self.LinReg_Baseline_24.val_step(val_24, val_constants),\
                    self.LinReg_Baseline_72.val_step(val_72, val_constants),\
                    self.LinReg_Baseline_120.val_step(val_120, val_constants),\
                    self.LinReg_Baseline_240.val_step(val_240, val_constants)
                # ResNets
                self.ResNet_6.val_step(val_6, val_constants), self.ResNet_24.val_step(val_24, val_constants),\
                    self.ResNet_72.val_step(val_72, val_constants), self.ResNet_120.val_step(val_120, val_constants),\
                    self.ResNet_240.val_step(val_240, val_constants)
                # UNets
                self.UNet_6.val_step(val_6, val_constants), self.UNet_24.val_step(val_24, val_constants),\
                    self.UNet_72.val_step(val_72, val_constants), self.UNet_120.val_step(val_120, val_constants),\
                    self.UNet_240.val_step(val_240, val_constants)
                # ViTs
                self.ViT_6.val_step(val_6, val_constants), self.ViT_24.val_step(val_24, val_constants),\
                    self.ViT_72.val_step(val_72, val_constants), self.ViT_120.val_step(val_120, val_constants),\
                    self.ViT_240.val_step(val_240, val_constants)

            # EarlyStopping
            # Baselines
            self.LinReg_Baseline_6.stopping(), self.LinReg_Baseline_24.stopping(), self.LinReg_Baseline_72.stopping(),\
                self.LinReg_Baseline_120.stopping(), self.LinReg_Baseline_240.stopping()
            # ResNets
            self.ResNet_6.stopping(), self.ResNet_24.stopping(), self.ResNet_72.stopping(), self.ResNet_120.stopping(),\
                self.ResNet_240.stopping()
            # UNets
            self.UNet_6.stopping(), self.UNet_24.stopping(), self.UNet_72.stopping(), self.UNet_120.stopping(),\
                self.UNet_240.stopping()
            # ViTs
            self.ViT_6.stopping(), self.ViT_24.stopping(), self.ViT_72.stopping(), self.ViT_120.stopping(),\
                self.ViT_240.stopping()

            if all([self.ResNet_6.done, self.ResNet_24.done, self.ResNet_72.done, self.ResNet_120.done,
                    self.ResNet_240.done, self.UNet_6.done, self.UNet_24.done, self.UNet_72.done, self.UNet_120.done,
                    self.UNet_240.done, self.ViT_6.done, self.ViT_24.done, self.ViT_72.done, self.ViT_120.done,
                    self.ViT_240.done]):
                print("Stopped prematurely due to EarlyStopping")
                break
        print("\nEnd Training")

    def evaluate_forecasters(self, test_loader, constants_set):
        print("Start Evaluation")
        for num, (test_6, test_24, test_72, test_120, test_240) in enumerate(test_loader):
            if len(test_6[0]) != 128:
                break
            printProgressAction('    Batch', num)
            batch_dim = len(test_6[0])
            test_constants = np.array((constants_set,) * batch_dim)
            # Baselines
            self.Baseline_6.eval_step(test_6[0], test_6[1], test_6[2], test_6[3][:, [4, 9, 33], :, :],
                                      test_constants), \
                self.Baseline_24.eval_step(test_24[0], test_24[1], test_24[2], test_24[3][:, [4, 9, 33], :, :],
                                           test_constants), \
                self.Baseline_72.eval_step(test_72[0], test_72[1], test_72[2], test_72[3][:, [4, 9, 33], :, :],
                                           test_constants), \
                self.Baseline_120.eval_step(test_120[0], test_120[1], test_120[2], test_120[3][:, [4, 9, 33], :, :],
                                            test_constants), \
                self.Baseline_240.eval_step(test_240[0], test_240[1], test_240[2], test_240[3][:, [4, 9, 33], :, :],
                                            test_constants)
            # ResNets
            self.ResNet_6.eval_step(test_6[0], test_6[1], test_6[2], test_6[3][:, [4, 9, 33], :, :], test_constants),\
                self.ResNet_24.eval_step(test_24[0], test_24[1], test_24[2], test_24[3][:, [4, 9, 33], :, :],
                                         test_constants),\
                self.ResNet_72.eval_step(test_72[0], test_72[1], test_72[2], test_72[3][:, [4, 9, 33], :, :],
                                         test_constants),\
                self.ResNet_120.eval_step(test_120[0], test_120[1], test_120[2], test_120[3][:, [4, 9, 33], :, :],
                                          test_constants),\
                self.ResNet_240.eval_step(test_240[0], test_240[1], test_240[2], test_240[3][:, [4, 9, 33], :, :],
                                          test_constants)
            # UNets
            self.UNet_6.eval_step(test_6[0], test_6[1], test_6[2], test_6[3][:, [4, 9, 33], :, :], test_constants),\
                self.UNet_24.eval_step(test_24[0], test_24[1], test_24[2], test_24[3][:, [4, 9, 33], :, :],
                                       test_constants),\
                self.UNet_72.eval_step(test_72[0], test_72[1], test_72[2], test_72[3][:, [4, 9, 33], :, :],
                                       test_constants),\
                self.UNet_120.eval_step(test_120[0], test_120[1], test_120[2], test_120[3][:, [4, 9, 33], :, :],
                                        test_constants),\
                self.UNet_240.eval_step(test_240[0], test_240[1], test_240[2], test_240[3][:, [4, 9, 33], :, :],
                                        test_constants)
            # ViTs
            self.ViT_6.eval_step(test_6[0], test_6[1], test_6[2], test_6[3][:, [4, 9, 33], :, :],
                                 test_constants),\
                self.ViT_24.eval_step(test_24[0], test_24[1], test_24[2], test_24[3][:, [4, 9, 33], :, :],
                                      test_constants),\
                self.ViT_72.eval_step(test_72[0], test_72[1], test_72[2], test_72[3][:, [4, 9, 33], :, :],
                                      test_constants),\
                self.ViT_120.eval_step(test_120[0], test_120[1], test_120[2], test_120[3][:, [4, 9, 33], :, :],
                                       test_constants),\
                self.ViT_240.eval_step(test_240[0], test_240[1], test_240[2], test_240[3][:, [4, 9, 33], :, :],
                                       test_constants)

        # Baselines
        rmse_LinReg_Baseline_6, acc_LinReg_Baseline_6 = compute_eval(self.Baseline_6.pred,
                                                                     self.Baseline_6.targ, self.latitude_weights,
                                                                     self.Baseline_6.Net.set_climatology)
        rmse_LinReg_Baseline_24, acc_LinReg_Baseline_24 = compute_eval(self.Baseline_24.pred,
                                                                       self.Baseline_24.targ,
                                                                       self.latitude_weights,
                                                                       self.Baseline_24.Net.set_climatology)
        rmse_LinReg_Baseline_72, acc_LinReg_Baseline_72 = compute_eval(self.Baseline_72.pred,
                                                                       self.Baseline_72.targ,
                                                                       self.latitude_weights,
                                                                       self.Baseline_72.Net.set_climatology)
        rmse_LinReg_Baseline_120, acc_LinReg_Baseline_120 = compute_eval(self.Baseline_120.pred,
                                                                         self.Baseline_120.targ,
                                                                         self.latitude_weights,
                                                                         self.Baseline_120.Net.set_climatology)
        rmse_LinReg_Baseline_240, acc_LinReg_Baseline_240 = compute_eval(self.Baseline_240.pred,
                                                                         self.Baseline_240.targ,
                                                                         self.latitude_weights,
                                                                         self.Baseline_240.Net.set_climatology)
        # ResNets
        rmse_ResNet_6, acc_ResNet_6 = compute_eval(self.ResNet_6.pred, self.ResNet_6.targ, self.latitude_weights,
                                                   self.ResNet_6.Net.set_climatology)
        rmse_ResNet_24, acc_ResNet_24 = compute_eval(self.ResNet_24.pred, self.ResNet_24.targ, self.latitude_weights,
                                                     self.ResNet_24.Net.set_climatology)
        rmse_ResNet_72, acc_ResNet_72 = compute_eval(self.ResNet_72.pred, self.ResNet_72.targ, self.latitude_weights,
                                                     self.ResNet_72.Net.set_climatology)
        rmse_ResNet_120, acc_ResNet_120 = compute_eval(self.ResNet_120.pred, self.ResNet_120.targ,
                                                       self.latitude_weights, self.ResNet_120.Net.set_climatology)
        rmse_ResNet_240, acc_ResNet_240 = compute_eval(self.ResNet_240.pred, self.ResNet_240.targ,
                                                       self.latitude_weights, self.ResNet_240.Net.set_climatology)
        # UNets
        rmse_UNet_6, acc_UNet_6 = compute_eval(self.UNet_6.pred, self.UNet_6.targ, self.latitude_weights,
                                               self.UNet_6.Net.set_climatology)
        rmse_UNet_24, acc_UNet_24 = compute_eval(self.UNet_24.pred, self.UNet_24.targ, self.latitude_weights,
                                                 self.UNet_24.Net.set_climatology)
        rmse_UNet_72, acc_UNet_72 = compute_eval(self.UNet_72.pred, self.UNet_72.targ, self.latitude_weights,
                                                 self.UNet_72.Net.set_climatology)
        rmse_UNet_120, acc_UNet_120 = compute_eval(self.UNet_120.pred, self.UNet_120.targ, self.latitude_weights,
                                                   self.UNet_120.Net.set_climatology)
        rmse_UNet_240, acc_UNet_240 = compute_eval(self.UNet_240.pred, self.UNet_240.targ, self.latitude_weights,
                                                   self.UNet_240.Net.set_climatology)
        # ViT
        rmse_ViT_6, acc_ViT_6 = compute_eval(self.ViT_6.pred, self.ViT_6.targ, self.latitude_weights,
                                             self.ViT_6.Net.set_climatology)
        rmse_ViT_24, acc_ViT_24 = compute_eval(self.ViT_24.pred, self.ViT_24.targ, self.latitude_weights,
                                               self.ViT_24.Net.set_climatology)
        rmse_ViT_72, acc_ViT_72 = compute_eval(self.ViT_72.pred, self.ViT_72.targ, self.latitude_weights,
                                               self.ViT_72.Net.set_climatology)
        rmse_ViT_120, acc_ViT_120 = compute_eval(self.ViT_120.pred, self.ViT_120.targ, self.latitude_weights,
                                                 self.ViT_120.Net.set_climatology)
        rmse_ViT_240, acc_ViT_240 = compute_eval(self.ViT_240.pred, self.ViT_240.targ, self.latitude_weights,
                                                 self.ViT_240.Net.set_climatology)

        linreg_baseline_rmse = create_list([rmse_LinReg_Baseline_6, rmse_LinReg_Baseline_24, rmse_LinReg_Baseline_72,
                                            rmse_LinReg_Baseline_120, rmse_LinReg_Baseline_240])
        linreg_baseline_acc = create_list([acc_LinReg_Baseline_6, acc_LinReg_Baseline_24, acc_LinReg_Baseline_72,
                                           acc_LinReg_Baseline_120, acc_LinReg_Baseline_240])
        resnet_rmse = create_list([rmse_ResNet_6, rmse_ResNet_24, rmse_ResNet_72, rmse_ResNet_120, rmse_ResNet_240])
        resnet_acc = create_list([acc_ResNet_6, acc_ResNet_24, acc_ResNet_72, acc_ResNet_120, acc_ResNet_240])
        unet_rmse = create_list([rmse_UNet_6, rmse_UNet_24, rmse_UNet_72, rmse_UNet_120, rmse_UNet_240])
        unet_acc = create_list([acc_UNet_6, acc_UNet_24, acc_UNet_72, acc_UNet_120, acc_UNet_240])
        vit_rmse = create_list([rmse_ViT_6, rmse_ViT_24, rmse_ViT_72, rmse_ViT_120, rmse_ViT_240])
        vit_acc = create_list([acc_ViT_6, acc_ViT_24, acc_ViT_72, acc_ViT_120, acc_ViT_240])

        plot('t2m', linreg_baseline_rmse[0], linreg_baseline_acc[0], resnet_rmse[0], resnet_acc[0], unet_rmse[0],
             unet_acc[0], vit_rmse[0], vit_acc[0])
        plot('Z500', linreg_baseline_rmse[1], linreg_baseline_acc[1], resnet_rmse[1], resnet_acc[1], unet_rmse[1],
             unet_acc[1], vit_rmse[1], vit_acc[1])
        plot('T850', linreg_baseline_rmse[2], linreg_baseline_acc[2], resnet_rmse[2], resnet_acc[2], unet_rmse[2],
             unet_acc[2], vit_rmse[2], vit_acc[2])

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self):
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
