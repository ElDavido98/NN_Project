from forecasting import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res_params = [32, 4]     # [128, 28]
u_params = [16, 1]       # [64, 2]
vit_params = [2, 2, 1, 32]     # [8, 4, 2, 128]

constants_set = define_sets('const')
train_set = define_sets('train')
validation_set, test_set = define_sets('val_test')
forecasters = Forecasting(constants_set, train_set, validation_set,
                          res_params=res_params, u_params=u_params, vit_params=vit_params).to(device)
forecasters.train_forecasters(epochs=1)
forecasters.save()
