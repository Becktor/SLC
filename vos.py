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

            #ood_samples = torch.cat((ood_samples,  GroupSort(2, 0)(negative_samples[index_prob])), 0)

        if len(ood_samples) != 0:
            # add some gaussian noise
            #gs_smp1 = GroupSort(2, 0)(ood_samples)
            #gs_smp2 = GroupSort(5, 0)(ood_samples)
            predictions_iid = vos_dict['pred']
            energy_score_for_fg = model.log_sum_exp(predictions_iid, 1)

            #ood_samples = torch.cat((ood_samples, gs_smp1), 0)#, gs_smp2), 0) #
            #ood_samples = GroupSort(2, 0)(vos_dict["output"])
            predictions_ood = model.fc(ood_samples)
            energy_score_for_bg = model.log_sum_exp(predictions_ood, 1)

            input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)

            input_for_lr_2 = torch.cat((predictions_iid, predictions_ood), 0)
            labels_for_lr = torch.cat((torch.ones(len(vos_dict['output'])).cuda(),
                                       torch.zeros(len(ood_samples)).cuda()), -1)

            # lbl2 = torch.cat((vos_dict['target'], torch.ones(len(ood_samples)).cuda() * vos_dict["num_classes"]), -1)

            #crit 1
            # criterion = torch.nn.CrossEntropyLoss()
            #out_1 = torch.nn.LeakyReLU(inplace=True)(input_for_lr)
            #out_1 = model.logistic_regression(input_for_lr.view(-1, 1))
            #lr_reg_loss = criterion(out_1, labels_for_lr.long())

            #out_2=model.logistic_regression((gg-input_for_lr).view(-1, 1))

            pred = torch.argmax(input_for_lr_2, 1)
            run_means = [model.vos_means[t] for t in pred]
            run_means = torch.stack(run_means).cuda()
            run_stds = [model.vos_stds[t] for t in pred]
            run_stds = torch.stack(run_stds).cuda()
            # min weight on ood samples. and lower weight of b

            # crit 2
            #weight_crit2 = torch.ones(vos_dict["num_classes"] + 1).cuda()
            #weight_crit2[-1] = 1/128
            #criterion2 = torch.nn.CrossEntropyLoss(weight=weight_crit2)

            clamped_inp = torch.clamp(input_for_lr, min=0.00001)
            shifted_means = torch.log(run_means.mean() / clamped_inp)
            # #xe = shifted_means * model.ood_mean
            # #out_j = xe.unsqueeze(1)
            # #output = torch.cat((input_for_lr_2, out_j), 1)
            # #xe_outlier = criterion2(output, lbl2.long())
            inv = (~labels_for_lr.bool()).float()
            # #out_1 = model.logistic_regression(shifted_means.view(-1, 1))
            criterion = torch.nn.BCEWithLogitsLoss()
            lr_reg_loss = criterion(shifted_means, inv)

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
    for i, x in enumerate(model.training_outputs):
        d = x.detach().cpu().numpy()
        plt.hist(d, bins=50)
        plt.show()
        plt.title(f'm: {i}')
        ad.append(d)
    plt.hist(np.concatenate(ad), bins=100)
    plt.title('all')
    plt.show()
