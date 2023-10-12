# simulation plots

from math import log,exp
from numpy.random import default_rng
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from tqdm import tqdm

import action_value

g = default_rng()

action_base_colours = {'ignore':'#0B18C2', 'control':'#309B50', 'control^2':'#EADF02', 'monitor':'#FF580E'}

def no_mon_action (n, dn,**params):
    Q_no_mon = action_value.Q_interventions(n,dn,**params)
    best_a = min(Q_no_mon, key=Q_no_mon.__getitem__).split(',')[0]
    return best_a

def general_action (n,dn,**params):
    Q = action_value.Q_all (n,dn,**params)
    best_a = min(Q, key=Q.__getitem__).split(',')[0]
    return best_a

# update state sequences - 
#   - for each policy: dict of lists of dicts (true_n, n, dn, action, reward to date)
#   - for ground truth: list of actual growth
# according to dict of actions
# first action in None, action in list is the action that "got you there"

def update_seqs (policy_seqs, growth_seq, actions, **params):
    next_growth = g.normal (loc=params['r'], scale=params['dr'])
    growth_seq.append(next_growth)

    for pol in policy_seqs:
        a = actions[pol]
        prev_step = policy_seqs[pol][-1]

        if a in params['interventions']:
            new_true_state = prev_step['true_n'] + log(1-params['interventions'][a][1]) + next_growth
            new_n = prev_step['n'] + log(1-params['interventions'][a][1]) + params['r']
            new_dn = prev_step['dn'] + params['dr']
            cost = params['interventions'][a][0]

        else:
            new_true_state = prev_step['true_n'] + log(1-params['mon_dict'][a][2]) + next_growth
            new_n = g.normal (loc=new_true_state, scale=params['mon_dict'][a][1])
            new_dn = action_value.new_dn(prev_step['dn'], params['mon_dict'][a][1]) + params['dr']
            cost = params['mon_dict'][a][0]

        rew = exp(prev_step['true_n']) + cost

        policy_seqs[pol].append(dict(true_n=new_true_state,
                                     n=new_n,
                                     dn=new_dn,
                                     action=a,
                                     rew_sofar=prev_step['rew_sofar'] + rew))

    return (policy_seqs, growth_seq)

# run the simulation
def run_sim (init_true_n, init_n, init_dn, seq_len=None, **params):

    init_policy_state = dict(true_n=init_true_n,
                             n=init_n,
                             dn=init_dn,
                             action=None,
                             rew_sofar=0)
    growth_seq = []
    policy_seqs = {'naive':[init_policy_state], 'full':[init_policy_state]}

    if seq_len is None:
        t = params['num_steps']
    else:
        t = seq_len

    for _ in range(t):

        naive_a = no_mon_action(n=policy_seqs['naive'][-1]['n'], dn=policy_seqs['naive'][-1]['dn'], **params)
        full_a = general_action(n=policy_seqs['full'][-1]['n'], dn=policy_seqs['full'][-1]['dn'], **params)
        actions = {'naive':naive_a, 'full':full_a}

        policy_seqs, growth_seq = update_seqs (policy_seqs, growth_seq, actions, **params)

        if seq_len is None: params['num_steps'] -= 1

    final_rewards = {pol: seq[-1]['rew_sofar'] + exp(seq[-1]['true_n']) for pol,seq in policy_seqs.items() }

    return policy_seqs, final_rewards

# plot a histogram of rewards over many simulations
# compare the naive and full model
def plot_batch (init_n, init_dn, num_sims, **params):

    returns = {'naive':[], 'full':[]}

    for _ in tqdm(range(num_sims)):
        init_true_n = g.normal(init_n, init_dn)
        pol_seqs, rews = run_sim(init_true_n, init_n, init_dn, **params)
        for pol in rews: returns[pol].append(rews[pol])

    mean_returns = {pol:sum(rets)/num_sims for pol,rets in returns.items()}
    print(pd.Series(mean_returns))

    fig,ax = plt.subplots()
    pols, rs = zip(*returns.items())

    hist, bins = np.histogram(rs)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

    ax.set_xscale('log')
    ax.hist(rs, bins=logbins, label=pols, density=True, log=True)

    ax.set_xlabel('reward (log scale)')
    ax.set_ylabel('density')
    ax.set_title(f'Simulated policies over {num_steps} steps, starting with mu={init_mu} and dn={init_dn}')

    ax.legend()
    plt.show()

# run simulations for a range of quantiles
def run_representatives (init_n, init_dn, qs, seq_len=None, **params):

    for q in qs:

        init_true_n = norm.ppf(q, loc=init_n, scale=init_dn)
        pol_seqs, rews = run_sim(init_true_n, init_n, init_dn, seq_len=seq_len, **params)

        # print(general_action(-1,1.5,**params))

        print('')
        print(f'Ground truth N = {exp(init_true_n):.2f} at quantile {q:.1f}')

        for pol,seq in pol_seqs.items():
            print(pol)
            print(pd.DataFrame(seq))
            print('---------------')

        print(pd.Series(rews))

# plot the timeseries sequence
def plot_seq (seq):

    action_alpha = 0.3
    action_colours = {a:action_base_colours[a] + hex(int(256 * action_alpha))[2:] for a in action_base_colours}

    action_seq = list(seq['action'])[1:]
    
    fig,ax = plt.subplots()

    seq['mu'] = seq['n'] + seq['dn']**2
    seq['lower_bound'] = seq['n']-seq['dn']
    seq['upper_bound'] = seq['n']+seq['dn']
    lower_n = seq.loc[:,['true_n', 'lower_bound']].min().min()
    upper_n = seq.loc[:,['true_n', 'upper_bound']].max().max()

    expand = 1.1
    m = (lower_n + upper_n) / 2
    width = upper_n - lower_n
    lower = m - expand * width / 2
    upper = m + expand * width / 2

    ax.set_ybound(lower,upper)
    ax.set_xbound(0,len(action_seq))

    for i,a in enumerate(action_seq):
        ax.add_patch(mpatches.Rectangle((i,lower), width=1, height=upper-lower, color=action_colours[a]))

    true_line = ax.plot(seq['true_n'], 'ko-', label='true n')[0]
    est_line = ax.plot(seq['n'], 'ko--', label='estimated n')[0]
    ax.fill_between(x=range(len(seq)), y1=seq['lower_bound'], y2=seq['upper_bound'], color='k', alpha=.1)
    line_legend = ax.legend(handles=[true_line, est_line], bbox_to_anchor=(1., .5), loc='lower left')

    # create a patch (proxy artist) for every color
    patches = [ mpatches.Patch(color=action_colours[k], label=k) for k in action_colours if k in action_seq]
    # put those patched as legend-handles into the legend
    plt.gca().add_artist(line_legend)
    plt.legend(handles=patches, title='actions taken', bbox_to_anchor=(1., .5), loc='upper left')
    
    ax.set_xlabel('time steps')
    ax.set_ylabel('abundance (log scale)')
    ax.set_title('Simulated time series: fire ant case study')

    plt.tight_layout()
    plt.show()

# plot the expected payoff as the quantile varies
def plot_payoffs (init_mu, init_dn, num_sims = 1, num_qs=50, **params):
    init_n = log(init_mu) - init_dn**2/2
    qs = np.linspace(0,1,num_qs, endpoint=False)

    rews = {'naive':[], 'full':[]}

    for q in tqdm(qs):
        
        cum_returns = {'naive':0, 'full':0}

        for _ in range(num_sims):
            init_true_n = norm.ppf(q, loc=init_n, scale=init_dn)
            _, cur_rews = run_sim(init_true_n, init_n, init_dn, **params)

            for pol,rew in cur_rews.items():
                cum_returns[pol] += rew

        for pol,rew in cum_returns.items():
            rews[pol].append(rew/num_sims)

    fig,ax = plt.subplots()

    for pol in rews:
        ax.plot(qs, rews[pol], label=pol)

    ax.legend()
    ax.set_title(f'expected payoffs, init_mu={init_mu:.2f} init_dn={init_dn:.2f} over {params["num_steps"]} steps')
    ax.set_xlabel('initial true abundance (quantile)')
    ax.set_ylabel('final cost')
    plt.show()


if __name__ == '__main__':

    ce = 4 # cost
    rho = 0.6 # efficacy

    more_interventions = {'ignore':(0.0,0.0), 'control':(ce,rho), 'control^2':(2*ce,1-(1-rho)**2)}

    cm = 1 # cost
    err_mon = 0.1 # std error in the monitoring process

    one_mon = {'monitor':(cm,err_mon,0)}

    r = 1.2 # growth rate
    dr = 0.05 # uncertainty in growth rate

    gamma = .9 # discount
    num_steps = 5 # horizon, plot involves this many "choices"

    params = dict(interventions=more_interventions,
                    mon_dict=one_mon,
                    gamma=gamma,
                    r=r,
                    dr=dr,
                    num_steps=num_steps,
                    waiting_steps=None)

    params['seq_len'] = 5

    init_dn = 1.8
    init_mu = 0.3
    init_n = log(init_mu) - init_dn**2/2


    fa_init_dn = 2
    fa_init_mu = .25
    fa_init_n = log(fa_init_mu) - fa_init_dn**2/2
    fire_ant_params = {'gamma': 0.9, 'num_steps': 2, 'detail': None, 'r': 2.82, 'dr': 0.015, 
                    'interventions': {'ignore': (0.0, 0.0), 'control': (31.82244784482759, 0.9989054351771577)}, 
                    'mon_dict': {'monitor': (6.204010721551725, 0, 0)}, 'seq_len':5}


    # q = .8
    # init_true_n = norm.ppf(q, loc=init_n, scale=init_dn)
    # seq = run_sim(init_true_n, init_n, init_dn, **params)[0]['full']
    # plot_seq(pd.DataFrame(seq))


    num_sims = 1000

    plot_batch(init_n, init_dn, num_sims, **params)

    run_representatives(init_n, init_dn, np.arange(0,1,.1), **params)

    plot_payoffs(init_mu, init_dn, num_sims=100, num_qs=100, **params)


    run_representatives(fa_init_n, fa_init_dn, np.arange(0,1,.1), **fire_ant_params)

    q = .8
    fa_init_true_n = norm.ppf(q, loc=fa_init_n, scale=fa_init_dn)
    seq = run_sim(fa_init_true_n, fa_init_n, fa_init_dn, **fire_ant_params)[0]['full']
    plot_seq(pd.DataFrame(seq))

    plot_payoffs(fa_init_mu, fa_init_dn, num_sims=1000, num_qs=100, **fire_ant_params)

    from plotting_functions import plot_regions

    s_range = (0,3)
    mu_range = (0,1)
    num_sqs = 1000

    pts = [(exp(d['n'] + d['dn']**2/2), d['dn']) for d in seq]
    plot_regions(mu_range,s_range,num_sqs,pts_to_plot=pts,**fire_ant_params)