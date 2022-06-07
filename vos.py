import torch
import torch.nn.functional as F
torch.manual_seed(0)


def log_sum_exp(value, weight_energy, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(
            F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def vos_update(model, vos_dict):
    sum_temp = 0
    for index in range(vos_dict["num_classes"]):
        sum_temp += vos_dict["number_dict"][index]
    lr_reg_loss = torch.zeros(1).cuda()[0]
    if sum_temp == vos_dict["num_classes"] * vos_dict["sample_number"] and vos_dict["epoch"] < vos_dict["start_epoch"]:
        # maintaining an ID data queue for each class.
        target_numpy = vos_dict["target"].cpu().data.numpy()
        for index in range(len(vos_dict["target"])):
            dict_key = target_numpy[index]
            vos_dict["data_dict"][dict_key] = torch.cat((vos_dict["data_dict"][dict_key][1:],
                                                         vos_dict["output"][index].detach().view(1, -1)), 0)
    elif sum_temp == vos_dict["num_classes"] * vos_dict["sample_number"] and vos_dict["epoch"] >= vos_dict[
        "start_epoch"]:
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
        temp_precision += 0.0001 * vos_dict["eye_matrix"]

        for index in range(vos_dict["num_classes"]):
            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embed_id[index], covariance_matrix=temp_precision)
            negative_samples = new_dis.rsample((vos_dict['sample_from'],))
            prob_density = new_dis.log_prob(negative_samples)
            # breakpoint()
            # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
            # keep the data in the low density area.
            cur_samples, index_prob = torch.topk(- prob_density, vos_dict["select"])
            if index == 0:
                ood_samples = negative_samples[index_prob]
            else:
                ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
        if len(ood_samples) != 0:
            # add some gaussian noise
            energy_score_for_fg = log_sum_exp(vos_dict['pred'], vos_dict['weight_energy'], 1)
            predictions_ood = model.fc(ood_samples)
            energy_score_for_bg = log_sum_exp(predictions_ood, vos_dict['weight_energy'], 1)

            input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
            labels_for_lr = torch.cat((torch.ones(len(vos_dict['output'])).cuda(),
                                       torch.zeros(len(ood_samples)).cuda()), -1)

            criterion = torch.nn.CrossEntropyLoss()
            output1 = vos_dict["logistic_regression"](input_for_lr.view(-1, 1))
            lr_reg_loss = criterion(output1, labels_for_lr.long())
    else:
        target_numpy = vos_dict['target'].cpu().data.numpy()
        for index in range(len(vos_dict['target'])):
            dict_key = target_numpy[index]
            if vos_dict['number_dict'][dict_key] < vos_dict['sample_number']:
                vos_dict['data_dict'][dict_key][vos_dict['number_dict'][dict_key]] = vos_dict['output'][index].detach()
                vos_dict['number_dict'][dict_key] += 1
    return lr_reg_loss

