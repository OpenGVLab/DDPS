import numpy as np
import torch
import math

from .misc import extract, log_add_exp, log_1_min_a, index_to_log_onehot, sample_categorical, log_onehot_to_index


def cos_alpha_schedule(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999, exp=3):
    att = np.arange(0, time_step)
    att = (np.cos((att + time_step) * math.pi * 0.5 / time_step) + 1)**exp
    att = att * (att_1 - att_T) + att_T
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]

    ctt = np.arange(0, time_step)
    ctt = (np.cos((ctt + time_step) * math.pi * 0.5 / time_step) + 1)**exp
    ctt = ctt * (ctt_1 - ctt_T) + ctt_T
    ctt = np.concatenate(([0], ctt))

    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


def alpha_schedule(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


def alpha_schedule_torch(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = torch.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = torch.cat((torch.tensor([1]), att))
    at = att[1:] / att[:-1]
    bt = (1 - at) / N
    att = torch.cat((att[1:], torch.tensor([1])))
    btt = (1 - att) / N
    return at, bt, att, btt


def cos_alpha_schedule_torch(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999, exp=2):
    att = torch.arange(0, time_step)
    att = (torch.cos((att + time_step) * math.pi * 0.5 / time_step) + 1)**exp
    att = att * (att_1 - att_T) + att_T
    att = torch.cat((torch.tensor([1]), att))
    at = att[1:] / att[:-1]
    bt = (1 - at) / N
    att = torch.cat((att[1:], torch.tensor([1])))
    btt = (1 - att) / N
    return at, bt, att, btt


def q_pred(x_start, t, num_timesteps, num_classes, log_cumprod_at, log_cumprod_bt):           # q(xt|x0)
    # log_x_start can be onehot or not
    log_x_start = index_to_log_onehot(x_start, num_classes)
    t = (t + (num_timesteps + 1)) % (num_timesteps + 1)
    log_cumprod_at = extract(log_cumprod_at, t, log_x_start.shape)         # at~
    log_cumprod_bt = extract(log_cumprod_bt, t, log_x_start.shape)         # bt~
    log_probs = torch.cat([
        log_add_exp(log_x_start + log_cumprod_at, log_cumprod_bt)
    ], dim=1)

    sample_logits = sample_categorical(log_probs)
    return sample_logits


def q_pred_log(log_x_start, t, num_timesteps, log_cumprod_at, log_cumprod_bt):       # q(xt|x0)
    t = (t + (num_timesteps + 1)) % (num_timesteps + 1)
    log_cumprod_at = extract(log_cumprod_at, t, log_x_start.shape)         # at~
    log_cumprod_bt = extract(log_cumprod_bt, t, log_x_start.shape)         # bt~
    log_probs = torch.cat([
        log_add_exp(log_x_start + log_cumprod_at, log_cumprod_bt)
    ], dim=1)
    return log_probs


def q_pred_log_one_step(log_x_start, t, log_at, log_bt):
    log_at = extract(log_at, t, log_x_start.shape)         # at~
    log_bt = extract(log_bt, t, log_x_start.shape)         # bt~
    log_probs_one_step = torch.cat([
        log_add_exp(log_x_start + log_at, log_bt)
    ], dim=1)
    return log_probs_one_step


def q_posterior(x_start, x_t, t, num_timesteps, num_classes, log_cumprod_at, log_cumprod_bt, log_at, log_bt):       
    log_x_start = index_to_log_onehot(x_start, num_classes)
    log_x_t = index_to_log_onehot(x_t, num_classes)
     
    # compute q(x_t|x_0)
    log_xt_given_x_start = q_pred_log(log_x_start, t, num_timesteps, log_cumprod_at, log_cumprod_bt)     
    
    # compute q(x_t|x_t-1,x_0) = q(x_t|x_t-1)
    # [1] Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions     
    # the following actually compute q(x_t+1|x_t), but [1] says it is the same as q(x_t|x_t-1)
    # see the appendix of [1]               
    log_xt_given_x_t_minus_1 = q_pred_log_one_step(log_x_t, t, log_at, log_bt)     
    
    # compute q(x_t-1|x_0)
    log_xt_minus_1_given_x_start = q_pred_log(log_x_start, t-1, num_timesteps, log_cumprod_at, log_cumprod_bt)
    
    log_EV_xtmin_given_xt_given_xstart = log_xt_given_x_t_minus_1 + log_xt_minus_1_given_x_start - log_xt_given_x_start
    
    log_probs = torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    sample_logits = sample_categorical(log_probs)
    return sample_logits



def q_posterior_log(log_x_start, log_x_t, t, num_timesteps, num_classes, log_cumprod_at, log_cumprod_bt, log_at, log_bt):       
    # compute q(x_t|x_0)
    log_xt_given_x_start = q_pred_log(log_x_start, t, num_timesteps, log_cumprod_at, log_cumprod_bt)     
    
    # compute q(x_t|x_t-1,x_0) = q(x_t|x_t-1)
    # [1] Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions     
    # the following actually compute q(x_t+1|x_t), but [1] says it is the same as q(x_t|x_t-1)
    # see the appendix of [1]               
    log_xt_given_x_t_minus_1 = q_pred_log_one_step(log_x_t, t, log_at, log_bt)     
    
    # compute q(x_t-1|x_0)
    log_xt_minus_1_given_x_start = q_pred_log(log_x_start, t-1, num_timesteps, log_cumprod_at, log_cumprod_bt)
    
    log_EV_xtmin_given_xt_given_xstart = log_xt_given_x_t_minus_1 + log_xt_minus_1_given_x_start - log_xt_given_x_start
    
    log_probs = torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    return log_probs