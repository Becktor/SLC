import torch
import torch.nn
import torch.nn.functional as F

torch.manual_seed(0)


def vos_update(model, vos_dict):
    sum_temp = 0
    for index in range(vos_dict["num_classes"]):
        sum_temp += vos_dict["number_dict"][index]
    lr_reg_loss = torch.zeros(1).cuda()[0]
    gauss_nll_loss = torch.zeros(1).cuda()[0]
    kl_loss_bg = torch.zeros(1).cuda()[0]
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
            energy_score_for_fg = model.log_sum_exp(vos_dict['pred'], 1)
            #clm = model.idd_mean.cuda() + model.idd_std.cuda() * 2
            #energy_score_for_fg = torch.clamp_max(energy_score_for_fg, clm)
            predictions_ood = model.fc(ood_samples)
            energy_score_for_bg = model.log_sum_exp(predictions_ood, 1)

            input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
            labels_for_lr = torch.cat((torch.ones(len(vos_dict['output'])).cuda(),
                                       torch.zeros(len(ood_samples)).cuda()), -1)

            criterion = torch.nn.CrossEntropyLoss()#torch.tensor([1, 1-labels_for_lr.float().mean()]))
            criterion.cuda()
            output1 = model.logistic_regression(input_for_lr.view(-1, 1))

            lr_reg_loss = criterion(output1, labels_for_lr.long())
            #
            if vos_dict["epoch"] >= model.start_epoch + 1:
                cls_oh = F.one_hot(vos_dict["target"])
                energy_fg_oh = energy_score_for_fg.unsqueeze(1)*cls_oh
                energy_fg_oh_m = energy_fg_oh.sum(0)/(cls_oh.sum(0) + 1e-6)
                #energy_fg_oh_std = torch.sqrt((((energy_fg_oh_m*cls_oh-energy_fg_oh)**2).sum(0)+1e-6)/(cls_oh.sum(0)))
                mm = energy_fg_oh_m.mean().detach().repeat(energy_fg_oh_m.shape[0], 1).reshape(-1)
                if energy_fg_oh_m.shape[0] == 8:
                    MSE = torch.nn.MSELoss()
                    squeeze = MSE(energy_fg_oh_m, mm)
                    gauss_nll_loss = torch.max(gauss_nll_loss, squeeze - model.vos_std.detach())
                    gauss_nll_loss += MSE(energy_fg_oh_m, model.vos_means.detach()) * 0.1
                    gauss_nll_loss += MSE(energy_score_for_bg.mean(), (model.vos_mean - model.vos_std * 3).detach()) * 0.1
                    gauss_nll_loss /= 2

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
    return lr_reg_loss, gauss_nll_loss, kl_loss_bg
