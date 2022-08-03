import torch
import torch.nn
import torch.nn.functional as F
from models import GroupSort
torch.manual_seed(0)


def vos_update(model, vos_dict):
    sum_temp = 0
    for index in range(vos_dict["num_classes"]):
        sum_temp += vos_dict["number_dict"][index]
    lr_reg_loss = torch.zeros(1).cuda()[0]
    xe_outlier = torch.zeros(1).cuda()[0]
    gauss_nll_loss = torch.zeros(1).cuda()[0]

    if sum_temp == vos_dict["num_classes"] * vos_dict["sample_number"] and vos_dict["epoch"] < model.start_epoch:
        # maintaining an ID data queue for each class.
        target_numpy = vos_dict["target"].cpu().data.numpy()
        for index in range(len(vos_dict["target"])):
            dict_key = target_numpy[index]
            vos_dict["data_dict"][dict_key] = torch.cat((vos_dict["data_dict"][dict_key][1:],
                                                         vos_dict["output"][index].detach().view(1, -1)), 0)
    elif sum_temp == vos_dict["num_classes"] * vos_dict["sample_number"] and vos_dict["epoch"] >= model.start_epoch:
        target_numpy = vos_dict["target"].cpu().data.numpy()
        for index in range(len(vos_dict["target"])):
            dict_key = target_numpy[index]
            vos_dict["data_dict"][dict_key] = torch.cat((vos_dict["data_dict"][dict_key][1:],
                                                         vos_dict["output"][index].detach().view(1, -1)), 0)
        # the covariance finder needs the data to be centered.
        for index in range(vos_dict["num_classes"]):
            if index == 0:
                x_var = vos_dict["data_dict"][index] - vos_dict["data_dict"][index].mean(0)
                mean_embed_id = vos_dict["data_dict"][index].mean(0).view(1, -1)
            else:
                x_var = torch.cat((x_var, vos_dict["data_dict"][index] - vos_dict["data_dict"][index].mean(0)), 0)
                mean_embed_id = torch.cat((mean_embed_id,
                                           vos_dict["data_dict"][index].mean(0).view(1, -1)), 0)

        ## add the variance.
        temp_precision = torch.mm(x_var.t(), x_var) / len(x_var)
        temp_precision += 0.0001 * model.eye_matrix

        for index in range(vos_dict["num_classes"]):
            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embed_id[index], covariance_matrix=temp_precision)
            negative_samples = new_dis.rsample((vos_dict['sample_from'],))
            prob_density = new_dis.log_prob(negative_samples)
            # breakpoint()
            # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
            # keep the data in the low density area.
            cur_samples, index_prob = torch.topk(-prob_density, vos_dict["select"])
            if index == 0:
                ood_samples = negative_samples[index_prob]
            else:
                ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)

        if len(ood_samples) != 0:
            # add some gaussian noise
            gs_smp1 = GroupSort(2, 0)(ood_samples)
            #gs_smp2 = GroupSort(5, 0)(ood_samples)
            predictions_iid = vos_dict['pred']
            energy_score_for_fg = model.log_sum_exp(predictions_iid, 1)

            ood_samples = torch.cat((ood_samples, gs_smp1), 0)#, gs_smp2), 0) #
            #ood_samples = GroupSort(2, 0)(vos_dict["output"])
            predictions_ood = model.fc(ood_samples)
            energy_score_for_bg = model.log_sum_exp(predictions_ood, 1)

            input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)

            input_for_lr_2 = torch.cat((predictions_iid, predictions_ood), 0)
            labels_for_lr = torch.cat((torch.ones(len(vos_dict['output'])).cuda(),
                                       torch.zeros(len(ood_samples)).cuda()), -1)

            lbl2 = torch.cat((vos_dict['target'], torch.ones(len(ood_samples)).cuda() * vos_dict["num_classes"]), -1)

            #crit 1
            criterion = torch.nn.CrossEntropyLoss()
            #out_1 = torch.nn.LeakyReLU(inplace=True)(input_for_lr)
            out_1 = model.logistic_regression(input_for_lr.view(-1, 1))
            lr_reg_loss = criterion(out_1, labels_for_lr.long())

            #out_2=model.logistic_regression((gg-input_for_lr).view(-1, 1))

            pred = torch.argmax(input_for_lr_2, 1)
            run_means = [model.vos_means[t] for t in pred]
            run_means = torch.stack(run_means).cuda()
            run_stds = [model.vos_stds[t] for t in pred]
            run_stds = torch.stack(run_stds).cuda()
            # min weight on ood samples. and lower weight of b

            # crit 2
            weight_crit2 = torch.ones(vos_dict["num_classes"] + 1).cuda()
            weight_crit2[-1] = 1 / vos_dict["bs"]
            criterion2 = torch.nn.CrossEntropyLoss(weight=weight_crit2)
            #run_norms = torch.distributions.normal.Normal(run_means, run_stds)
            #out_2 = (1 - run_norms.cdf(input_for_lr))*2 - 1
            #out_2 = torch.nn.LeakyReLU()(out_2)
            out_s = torch.softmax(out_1, 1)
            shifted_means = run_means + model.ood_mean - input_for_lr
            out_j = (shifted_means * out_s[:, 0]).unsqueeze(1)#(run_means * out_2).unsqueeze(1)#(torch.pow(run_means + run_stds, 2) / torch.pow(input_for_lr, 2)).unsqueeze(1)
            #out_j = torch.nn.Tanh()(out_j).unsqueeze(1)
            #s = torch.softmax(input_for_lr_2, 1)
            output = torch.cat((input_for_lr_2, out_j), 1)
            # output = (run_means - run_stds).unsqueeze(1) - input_for_lr_2
            #residual = model.logistic_regression2(output)
            # output2 = torch.nn.LeakyReLU(inplace=True)(output2)
            # residual = model.logistic_regression3(output2)
            #residual = torch.nn.LeakyReLU(inplace=True)(residual)
            #catted = torch.cat((input_for_lr_2, torch.zeros_like(out_j).cuda()), 1)
            #output3 = catted + residual
            xe_outlier = criterion2(output, lbl2.long())
            # BCE = torch.nn.BCEWithLogitsLoss()
            # inv = (~labels_for_lr.bool()).float()
            # lr_reg_loss = BCE(output[:128, -1], inv[:128])
            #
            # #
            # if vos_dict["epoch"] >= model.start_epoch + 1:
            #     cls_oh = F.one_hot(vos_dict["target"])
            #     energy_fg_oh = energy_score_for_fg.unsqueeze(1) * cls_oh
            #     energy_fg_oh_m = energy_fg_oh.sum(0) / (cls_oh.sum(0) + 1e-6)
            #     # energy_fg_oh_std = torch.sqrt((((energy_fg_oh_m*cls_oh-energy_fg_oh)**2).sum(0)+1e-6)/(cls_oh.sum(0)))
            #     # mm = energy_fg_oh_m.mean().detach().repeat(energy_fg_oh_m.shape[0], 1).reshape(-1)
            #     if energy_fg_oh_m.shape[0] == 10:
            #         MSE = torch.nn.MSELoss()
            #         gauss_nll_loss = MSE(energy_fg_oh_m, torch.abs(model.vos_means).detach()) * 0.01
            #         # torch.max(torch.tensor(12), torch.abs(model.vos_means.detach())))
            #         # gauss_nll_loss = torch.max(gauss_nll_loss, squeeze - model.vos_stds.detach()) * 0.01
            #         gauss_nll_loss += MSE(energy_score_for_bg, (model.vos_means - model.vos_stds * 3).detach()) * 0.001
            #         # gauss_nll_loss += MSE(energy_score_for_fg.mean(), (model.vos_mean + model.vos_std * 2).detach()) * 0.005


            #var = torch.max((energy_fg_oh_m * cls_oh - energy_fg_oh) ** 2, 1)[0]

            #r = [torch.distributions.normal.Normal(c.detach(), s.detach()) for c, s in zip(model.vos_mean, model.vos_std)]
            # dis_cls = [torch.tensor([model.vos_mean[i].detach(), model.vos_std[i].detach()]) for i in vos_dict["target"]]
            # gauss_cls = torch.stack(dis_cls).cuda()
            #
            # dis_fg = torch.tensor([model.vos_mean.mean(0).detach(), model.vos_std.mean(0).detach()])
            # gauss_fg = dis_fg.repeat(energy_score_for_fg.shape[0], 1).cuda()
            #
            # dis_bg = torch.tensor([model.ood_mean, model.ood_std])
            # gauss_bg = dis_bg.repeat(energy_score_for_bg.shape[0], 1).cuda()

            # g_nll_crit = torch.nn.GaussianNLLLoss(reduction='mean')
            # gauss_nll_loss = g_nll_crit(gauss_cls[:, 0], energy_score_for_fg,  gauss_cls[:, 1])
            # #gauss_nll_loss += g_nll_crit(gauss_fg[:, 0], energy_score_for_fg,  gauss_fg[:, 1])
            # if model.ood_mean > torch.tensor(0) and not model.ood_std.isnan():
            #     gauss_nll_loss += g_nll_crit(gauss_bg[:, 0], energy_score_for_bg, gauss_bg[:, 1])
            #     gauss_nll_loss /= 2
            # else:
            #     gauss_nll_loss /= 1
            #kl_true_bg = curr_dist.rsample(energy_score_for_bg.shape)
            #kl_loss_bg = 0#torch.log(1 + kl_crit(-energy_score_for_bg, kl_true_bg))


    else:
        target_numpy = vos_dict['target'].cpu().data.numpy()
        for index in range(len(vos_dict['target'])):
            dict_key = target_numpy[index]
            if vos_dict['number_dict'][dict_key] < vos_dict['sample_number']:
                vos_dict['data_dict'][dict_key][vos_dict['number_dict'][dict_key]] = vos_dict['output'][index].detach()
                vos_dict['number_dict'][dict_key] += 1
    return lr_reg_loss, xe_outlier, gauss_nll_loss

def hist_train_samples(model):
    import matplotlib.pyplot as plt
    import numpy as np
    ad = []
    for i,x in enumerate(model.training_outputs):
        d = x.detach().cpu().numpy()
        plt.hist(d, bins=50)
        plt.show()
        plt.title(f'm: {i}')
        ad.append(d)
    plt.hist(np.concatenate(ad), bins=100)
    plt.title('all')
    plt.show()
