from forecasting import *
from utils import *
from metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

res_params = [32, 4]     # [128, 28]
u_params = [16, 1]       # [64, 2]
vit_params = [2, 2, 1, 32]     # [8, 4, 2, 128]

constants_set = define_sets('const')
train_set = define_sets('train')
validation_set, test_set = define_sets('val_test')
latitude_weights = latitude_weighting_function(latitude_coordinates)
forecasters = Forecasting(constants_set, train_set, validation_set,
                          res_params=res_params, u_params=u_params, vit_params=vit_params).to(device)

t_6 = [test_set[12:4374], test_set[6:4368], test_set[0:4362], test_set[18:4380]]
t_24 = [test_set[12:4356], test_set[6:4350], test_set[0:4344], test_set[36:4380]]
t_72 = [test_set[12:4308], test_set[6:4302], test_set[0:4296], test_set[84:4380]]
t_120 = [test_set[12:4260], test_set[6:4254], test_set[0:4248], test_set[132:4380]]
t_240 = [test_set[12:4140], test_set[6:4134], test_set[0:4128], test_set[252:4380]]
t = CustomDataset(t_6, t_24, t_72, t_120, t_240)
test_loader = torch.utils.data.DataLoader(t, batch_size=128, shuffle=True)

forecasters.load()
forecasters.evaluate_forecasters(test_loader, constants_set)

"""pred_ResNet_6, pred_ResNet_24, pred_ResNet_72, pred_ResNet_120, pred_ResNet_240 = ([], ) * 5
targ_ResNet_6, targ_ResNet_24, targ_ResNet_72, targ_ResNet_120, targ_ResNet_240 = ([], ) * 5
pred_UNet_6, pred_UNet_24, pred_UNet_72, pred_UNet_120, pred_UNet_240 = ([], ) * 5
targ_UNet_6, targ_UNet_24, targ_UNet_72, targ_UNet_120, targ_UNet_240 = ([], ) * 5
pred_ViT_6, pred_ViT_24, pred_ViT_72, pred_ViT_120, pred_ViT_240 = ([], ) * 5
targ_ViT_6, targ_ViT_24, targ_ViT_72, targ_ViT_120, targ_ViT_240 = ([], ) * 5

print("Start Evaluation")
for num, (test_6, test_24, test_72, test_120, test_240) in enumerate(test_loader):
    if len(test_6[0]) != 128:
        break
    printProgressAction('    Batch', num)
    batch_dim = len(test_6[0])
    test_constants = np.array((constants_set,) * batch_dim)
    p = forecasters.ResNet_6.Net(test_6[0], test_6[1], test_6[2], test_6[3][:, [4, 9, 33], :, :], test_constants)
    pred_ResNet_6.append(p.detach().numpy()), targ_ResNet_6.append(test_6[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.ResNet_24.Net(test_24[0], test_24[1], test_24[2], test_24[3][:, [4, 9, 33], :, :], test_constants)
    pred_ResNet_24.append(p.detach().numpy()), targ_ResNet_24.append(test_24[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.ResNet_72.Net(test_72[0], test_72[1], test_72[2], test_72[3][:, [4, 9, 33], :, :], test_constants)
    pred_ResNet_72.append(p.detach().numpy()), targ_ResNet_72.append(test_72[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.ResNet_120.Net(test_120[0], test_120[1], test_120[2], test_120[3][:, [4, 9, 33], :, :], test_constants)
    pred_ResNet_120.append(p.detach().numpy()), targ_ResNet_120.append(test_120[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.ResNet_240.Net(test_240[0], test_240[1], test_240[2], test_240[3][:, [4, 9, 33], :, :], test_constants)
    pred_ResNet_240.append(p.detach().numpy()), targ_ResNet_240.append(test_240[3][:, [4, 9, 33], :, :].detach().numpy())
    # UNets
    p = forecasters.UNet_6.Net(test_6[0], test_6[1], test_6[2], test_6[3][:, [4, 9, 33], :, :], test_constants)
    pred_UNet_6.append(p.detach().numpy()), targ_UNet_6.append(test_6[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.UNet_24.Net(test_24[0], test_24[1], test_24[2], test_24[3][:, [4, 9, 33], :, :], test_constants)
    pred_UNet_24.append(p.detach().numpy()), targ_UNet_24.append(test_24[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.UNet_72.Net(test_72[0], test_72[1], test_72[2], test_72[3][:, [4, 9, 33], :, :], test_constants)
    pred_UNet_72.append(p.detach().numpy()), targ_UNet_72.append(test_72[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.UNet_120.Net(test_120[0], test_120[1], test_120[2], test_120[3][:, [4, 9, 33], :, :], test_constants)
    pred_UNet_120.append(p.detach().numpy()), targ_UNet_120.append(test_120[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.UNet_240.Net(test_240[0], test_240[1], test_240[2], test_240[3][:, [4, 9, 33], :, :], test_constants)
    pred_UNet_240.append(p.detach().numpy()), targ_UNet_240.append(test_240[3][:, [4, 9, 33], :, :].detach().numpy())
    # ViT
    p = forecasters.ViT_6.Net(test_6[0], test_6[1], test_6[2], test_6[3][:, [4, 9, 33], :, :], test_constants)
    pred_ViT_6.append(p.detach().numpy()), targ_ViT_6.append(test_6[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.ViT_24.Net(test_24[0], test_24[1], test_24[2], test_24[3][:, [4, 9, 33], :, :], test_constants)
    pred_ViT_24.append(p.detach().numpy()), targ_ViT_24.append(test_24[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.ViT_72.Net(test_72[0], test_72[1], test_72[2], test_72[3][:, [4, 9, 33], :, :], test_constants)
    pred_ViT_72.append(p.detach().numpy()), targ_ViT_72.append(test_72[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.ViT_120.Net(test_120[0], test_120[1], test_120[2], test_120[3][:, [4, 9, 33], :, :], test_constants)
    pred_ViT_120.append(p.detach().numpy()), targ_ViT_120.append(test_120[3][:, [4, 9, 33], :, :].detach().numpy())
    p = forecasters.ViT_240.Net(test_240[0], test_240[1], test_240[2], test_240[3][:, [4, 9, 33], :, :], test_constants)
    pred_ViT_240.append(p.detach().numpy()), targ_ViT_240.append(test_240[3][:, [4, 9, 33], :, :].detach().numpy())


# ResNets
rmse_ResNet_6, acc_ResNet_6 = compute_eval(pred_ResNet_6, targ_ResNet_6, latitude_weights,
                                           forecasters.ResNet_6.Net.set_climatology)
rmse_ResNet_24, acc_ResNet_24 = compute_eval(pred_ResNet_24, targ_ResNet_24, latitude_weights,
                                             forecasters.ResNet_24.Net.set_climatology)
rmse_ResNet_72, acc_ResNet_72 = compute_eval(pred_ResNet_72, targ_ResNet_72, latitude_weights,
                                             forecasters.ResNet_72.Net.set_climatology)
rmse_ResNet_120, acc_ResNet_120 = compute_eval(pred_ResNet_120, targ_ResNet_120, latitude_weights,
                                               forecasters.ResNet_120.Net.set_climatology)
rmse_ResNet_240, acc_ResNet_240 = compute_eval(pred_ResNet_240, targ_ResNet_240, latitude_weights,
                                               forecasters.ResNet_240.Net.set_climatology)
# UNets
rmse_UNet_6, acc_UNet_6 = compute_eval(pred_UNet_6, targ_UNet_6, latitude_weights,
                                       forecasters.UNet_6.Net.set_climatology)
rmse_UNet_24, acc_UNet_24 = compute_eval(pred_UNet_24, targ_UNet_24, latitude_weights,
                                         forecasters.UNet_24.Net.set_climatology)
rmse_UNet_72, acc_UNet_72 = compute_eval(pred_UNet_72, targ_UNet_72, latitude_weights,
                                         forecasters.UNet_72.Net.set_climatology)
rmse_UNet_120, acc_UNet_120 = compute_eval(pred_UNet_120, targ_UNet_120, latitude_weights,
                                           forecasters.UNet_120.Net.set_climatology)
rmse_UNet_240, acc_UNet_240 = compute_eval(pred_UNet_240, targ_UNet_240, latitude_weights,
                                           forecasters.UNet_240.Net.set_climatology)
# ViT
rmse_ViT_6, acc_ViT_6 = compute_eval(pred_ViT_6, targ_ViT_6, latitude_weights,
                                     forecasters.ViT_6.Net.set_climatology)
rmse_ViT_24, acc_ViT_24 = compute_eval(pred_ViT_24, targ_ViT_24, latitude_weights,
                                       forecasters.ViT_24.Net.set_climatology)
rmse_ViT_72, acc_ViT_72 = compute_eval(pred_ViT_72, targ_ViT_72, latitude_weights,
                                       forecasters.ViT_72.Net.set_climatology)
rmse_ViT_120, acc_ViT_120 = compute_eval(pred_ViT_120, targ_ViT_120, latitude_weights,
                                         forecasters.ViT_120.Net.set_climatology)
rmse_ViT_240, acc_ViT_240 = compute_eval(pred_ViT_240, targ_ViT_240, latitude_weights,
                                         forecasters.ViT_240.Net.set_climatology)

resnet_rmse = create_list([rmse_ResNet_6, rmse_ResNet_24, rmse_ResNet_72, rmse_ResNet_120, rmse_ResNet_240])
resnet_acc = create_list([acc_ResNet_6, acc_ResNet_24, acc_ResNet_72, acc_ResNet_120, acc_ResNet_240])
unet_rmse = create_list([rmse_UNet_6, rmse_UNet_24, rmse_UNet_72, rmse_UNet_120, rmse_UNet_240])
unet_acc = create_list([acc_UNet_6, acc_UNet_24, acc_UNet_72, acc_UNet_120, acc_UNet_240])
vit_rmse = create_list([rmse_ViT_6, rmse_ViT_24, rmse_ViT_72, rmse_ViT_120, rmse_ViT_240])
vit_acc = create_list([acc_ViT_6, acc_ViT_24, acc_ViT_72, acc_ViT_120, acc_ViT_240])

plot('t2m', resnet_rmse[0], resnet_acc[0], unet_rmse[0], unet_acc[0], vit_rmse[0], vit_acc[0])
plot('Z500', resnet_rmse[1], resnet_acc[1], unet_rmse[1], unet_acc[1], vit_rmse[1], vit_acc[1])
plot('T850', resnet_rmse[2], resnet_acc[2], unet_rmse[2], unet_acc[2], vit_rmse[2], vit_acc[2])
"""