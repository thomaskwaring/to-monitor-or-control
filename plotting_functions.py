# once more for the people in the back

from math import exp,log
import numpy as np
from itertools import product
import pickle
from scipy.optimize import root_scalar
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib as mpl
import colorcet as cc

from tqdm import tqdm

import action_value
import integrals
import tree_colours

# primary plots ------------------------------

# given funcs calculated with horizon t (ie t actions), calculate t+1
# includes _reward_ for the next step
def add_step_funcs (init_funcs, t, gamma, r, dr, dn, interventions, **_):
    growth_and_reward = exp(r*t + (dn + t*dr)**2/2)

    if t == 0: # no actions
        return {'':(0,growth_and_reward)}
    
    fs = {}
    for a_name,(c,slope) in init_funcs.items():
        prev_effect = 1
        if len(a_name) > 0:
            for a in a_name.split(','):
                prev_effect*=(1-interventions[a][1])
        for b,(cost,rho) in interventions.items():
            next_name = b if len(a_name)==0 else a_name + ',' + b
            
            fs[next_name] = (c + gamma**(t-1) * cost, 
                slope + gamma**t * prev_effect * (1-rho) * growth_and_reward)
    return fs

# compute things separately for s and mu
# break down by how many steps in advance also
def assign_actions (mu_range,s_range,num_sqs,detail,
        mon_dict,gamma,r,dr,interventions,num_steps,waiting_steps=None,**_):
    
    if waiting_steps is None: max_waiting = num_steps-2
    else: max_waiting = min(num_steps-2,waiting_steps)

    mus,mu_step = np.linspace(mu_range[0], mu_range[1], num_sqs, retstep=True)
    ss,s_step = np.linspace(s_range[0], s_range[1], num_sqs, retstep=True)

    if mu_range[0] == 0: mus = np.delete(mus,0)
    if s_range[0] == 0: ss = np.delete(ss,0)

    
    # effect of control interventions
    change_n = {','.join(a_seq): i*r + action_value.action_effects(a_seq,interventions) 
        for i in range(max_waiting+1) for a_seq in product(interventions,repeat=i)}

    pts = []
    qs = []
    actions = set()
    for s in tqdm(ss):
        column = []
        col_qs = []

        # find funcs, before / after monitoring for each time step
        funcs_t = [add_step_funcs({},0,gamma,r,dr,s,interventions)]

        mon_funcs_t = []
        for i in range(1,num_steps+1):
            funcs_t.append(add_step_funcs(funcs_t[-1],i,gamma,r,dr,s,interventions))

            if i <= max_waiting+1:
                dn_before_mon = s+(i-1)*dr
                dn_after_mon = {mon_a:action_value.new_dn(dn_before_mon,err) for mon_a,(_,err,_) in mon_dict.items()}
                mon_funcs_t.append({mon_a:action_value.find_funcs(gamma,r,dr, dn_after + dr, interventions, num_steps-i) 
                    for mon_a,dn_after in dn_after_mon.items()})
        
        

        fs_by_mu = {a:(const,slope*exp(-s*s/2)) for a,(const,slope) in funcs_t[-1].items()}
        
        # compute intersections before monitoring
        # list (x_i,a_i,func) upper bounds x_i, best action a_i, corresponding func
        best_actions = [(x,a,fs_by_mu[a]) for x,a in integrals.find_intersection(fs_by_mu)]
        upper_mu, best_control, f = best_actions.pop(0)

        # compute intersections after monitoring
        pieces_t = [{mon_a:integrals.find_intersection(fs) 
            for mon_a,fs in mon_funcs.items()} for mon_funcs in mon_funcs_t]

        # fill out the column with dn=s
        for mu in mus:
            #control values
            if not (upper_mu is None) and mu >= upper_mu: upper_mu, best_control, f = best_actions.pop(0)
            control_val = f[0] + f[1]*mu

            # estimate of log(n)
            n = log(mu) - s*s/2

            # fill out monitoring
            mon_vals = {}

            # how long to wait before monitoring
            for wait in range(max_waiting+1):

                # actions before monitoring
                for a_name,(rew_c,rew_slope) in funcs_t[wait].items():

                    # monitoring action
                    for mon_a,(cost,err,rho) in mon_dict.items():

                        next_name = mon_a if len(a_name)==0 else a_name + ',' + mon_a # name of the action

                        # integrate each piece of the "after" value function
                        lower = None
                        future = 0
                        for piece in pieces_t[wait][mon_a]:
                            act = piece[1] # action for this piece
                            c,slope = mon_funcs_t[wait][mon_a][act] # params for this action

                            # parameters for the distribution at time of monitoring
                            n_before_mon = n+change_n[a_name]
                            dn_before_mon = s+wait*dr

                            marginal_cost = slope*exp(r)*(1-rho) # exp(r) to account for growth before the next step
                            
                            future += integrals.int_lognorm(n_before_mon, dn_before_mon, c, marginal_cost, lower, piece[0])
                            lower = piece[0] # move to lower bound of next piece
                        
                        before_reward = rew_c + rew_slope*exp(n) # depends on actions before monitoring

                        mon_vals[next_name] = before_reward + gamma**wait * (cost + gamma*future)

            best_mon = min(mon_vals,key=mon_vals.get) # best among monitoring actions

            a = best_control if control_val < mon_vals[best_mon] else best_mon # best action

            # update lists
            actions.add(a)
            column.append(a)
            qs.append(col_qs)
        
        pts.append(column)
    
    return pts,actions,(mu_step,s_step),(mus,ss)


def title (interventions,mon_dict,gamma,r,dr,num_steps,**_):
    ints_str = '(cost,efficacy) pairs ' + ', '.join([f"{name}:({interventions[name][0]:.2f},{interventions[name][1]:.2f})" for name in interventions if interventions[name] != (0,0)])
    mons_str = '(cost,error,efficacy) pairs ' + ', '.join([f"{name}:({mon_dict[name][0]:.2f},{mon_dict[name][1]:.2f},{mon_dict[name][2]:.2f})" for name in mon_dict])

    return f'Optimal strategies with gamma={gamma}, r={r}, dr={dr},\n {ints_str},\n and {mons_str}\nover {num_steps} steps'

def shorten (a,detail): return ','.join(a.split(',')[:detail])


def plot_regions (mu_range,s_range,num_sqs,detail=None,colormap="cet_rainbow4",f=0.75,
                  fname=None,pts_to_plot=None,plot_title=False,savename=None,**params):
    pts,actions,step_sizes,(mus,ss) = assign_actions(mu_range,s_range,num_sqs,detail,
        **params)

    if not fname is None:
        with open(f'data/{fname}.pkl', 'wb') as file:
            pickle.dump((pts,actions,mu_range,s_range,num_sqs,params),file)

    reduced_actions = {shorten(a,detail) for a in actions}

    if 'glasbey' in colormap:
        colours = plt.get_cmap(colormap).colors
        colour_key = dict(zip(sorted(reduced_actions),colours))
    else:
        cmap = plt.get_cmap(colormap)

        key_d = {a:rho for a,(_,rho) in params['interventions'].items()}
        key_d.update({a:(1+cost) for a,(cost,_,_) in params['mon_dict'].items()})
        atree = tree_colours.Tree('',children=tree_colours.forest_from_list(reduced_actions,key=key_d.get))
        
        colour_key = {act:cmap(x) for (act,x) in tree_colours.colours_dict(atree,f=f).items()}

    image = np.array([[colour_key[shorten(a,detail)] for a in column] for column in pts])

    fig,ax = plt.subplots()

    ax.imshow(np.transpose(image,axes=(1,0,2)), origin='lower', aspect='auto')

    ax.set_xlabel('uncertainty')
    ax.set_ylabel('abundance', labelpad = 10)
    if plot_title:
        ax.set_title(title(**params))


    rescaler_x = ticker.FuncFormatter(lambda x, pos: f'{ss[0]+x*step_sizes[1] : .1f}')
    rescaler_y = ticker.FuncFormatter(lambda x, pos: f'{mus[0]+x*step_sizes[0] : .1f}')
    ax.yaxis.set_major_formatter(rescaler_y)
    ax.xaxis.set_major_formatter(rescaler_x)

    # create a patch (proxy artist) for every color
    patches = [ mpatches.Patch(color=colour_key[k], label=k) for k in colour_key if k in reduced_actions]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches)

    # add reference points to the phase plot, eg states in a time series
    if not pts_to_plot is None:
        pt_mus = [int((mu-mus[0])/step_sizes[0]) for mu,_ in pts_to_plot]
        pt_dns = [int((dn-ss[0])/step_sizes[1]) for _,dn in pts_to_plot]
        ax.plot(pt_dns,pt_mus, 'ko')

    if not savename is None:
        plt.savefig(f'figs/{savename}.png', dpi=300)
    plt.show()

    red_ckey = {a:colour_key[a] for a in reduced_actions}
    return red_ckey

def plot_from_file (fname, detail, colormap, f, title=None):
    fig,ax = plt.subplots()

    with open(f'data/{fname}.pkl','rb') as file:
        pts,actions,mu_range,s_range,num_sqs,params = pickle.load(file)
    
    if detail is None:
        detail=params['num_steps']

    reduced_actions = {shorten(a,detail) for a in actions}

    if 'glasbey' in colormap:
        colours = plt.get_cmap(colormap).colors
        colour_key = dict(zip(sorted(reduced_actions),colours))
    else:
        cmap = plt.get_cmap(colormap)

        key_d = {a:rho for a,(_,rho) in params['interventions'].items()}
        key_d.update({a:(1+cost) for a,(cost,_,_) in params['mon_dict'].items()})
        atree = tree_colours.Tree('',children=tree_colours.forest_from_list(reduced_actions,key=key_d.get))
        
        colour_key = {act:cmap(x) for (act,x) in tree_colours.colours_dict(atree,f=f).items()}

    image = np.array([[colour_key[shorten(a,detail)] for a in column] for column in pts])
    ax.imshow(np.transpose(image,axes=(1,0,2)), origin='lower', aspect='auto')

    mus,mu_step = np.linspace(mu_range[0], mu_range[1], num_sqs, retstep=True)
    ss,s_step = np.linspace(s_range[0], s_range[1], num_sqs, retstep=True)

    rescaler_x = ticker.FuncFormatter(lambda x, pos: f'{s_range[0]+x*mu_step : .1f}')
    rescaler_y = ticker.FuncFormatter(lambda x, pos: f'{mu_range[0]+x*s_step : .1f}')
    ax.yaxis.set_major_formatter(rescaler_y)
    ax.xaxis.set_major_formatter(rescaler_x)

    if not title is None:
        ax.set_title(title)

    ax.set_xlabel('uncertainty')
    ax.set_ylabel('abundance', labelpad = 10)

    patches = [ mpatches.Patch(color=colour_key[k], label=k) for k in colour_key if k in reduced_actions]
    plt.legend(handles=patches)

    plt.show()


# parameter sensitivity plots ------------------------------------


def all_mon_actions (num_steps,interventions,mon_dict,**_):
    mon_actions = [','.join(bef)+','+mon_a for i in range(1,num_steps-1) for bef in product(interventions,repeat=i) for mon_a in mon_dict]
    mon_actions = list(mon_dict) + mon_actions

    return mon_actions

def find_frontiers (mu_range,num_pts,fst_guess,s_range,**params):

    mon_actions = all_mon_actions(**params)

    ss = np.linspace(s_range[0],s_range[1])
    mus = np.linspace(mu_range[0],mu_range[1],num_pts)

    def _func_for_root_new (s,mu,action,**params):
        m = log(mu) - s*s/2
        interested = action_value.Q(action,dn=s,n=m,**params)
        vals = action_value.Q_interventions(n=m,dn=s,**params).values()
        return interested - min(vals)

    lines = {}
    for a in mon_actions:
        line = []
        for mu in mus:
            prev = fst_guess
            try:
                ans = root_scalar(partial(_func_for_root_new,mu=mu,action=a,**params),
                    x0=prev,x1=prev+0.1,bracket=(s_range[0]+0.01,s_range[1]))
                line.append((ans.root,mu))
                prev = ans.root
                # print(ans.function_calls)
            except ValueError:
                # print(f'hello {mu}')
                pass
        if len(line) > 0:
            # print(a,line[-1])
            line = [(s_range[1],line[0][1])] + line + [(s_range[1],line[-1][1])]
            # print(a,line[0],line[-1])
            lines[a] = line
    return lines

def varying_frontiers (param_setter, param_values, name, action, mu_range, 
                       s_range, num_pts, cmap_name, include_title=True,
                        savename=None, **params):
    
    fig,ax = plt.subplots()

    cmap = mpl.cm.get_cmap(cmap_name)

    def _get_ind (val):
        return (val - min(param_values)) / (max(param_values) - min(param_values))

    for new_val in param_values:
        if isinstance(param_setter,str):
            params[param_setter] = new_val
        else:
            param_setter(params, new_val)
        lines = find_frontiers(mu_range, num_pts, 
                                              fst_guess=(s_range[1]-s_range[0])/2, 
                                              s_range=s_range, **params)
        if action in lines: ax.plot(*zip(*lines[action]), label=f'{name}={new_val}', color=cmap(_get_ind(new_val)))
        
    
    ax.legend()
    ax.set_xlabel('uncertainty')
    ax.set_ylabel('abundance')
    if include_title: ax.set_title('Monitoring boundary')
    if not savename is None:
        plt.savefig(f'figs/{savename}.png', dpi=300)
    plt.show()

def varying_crossovers (param_setter, param_range, name, mu_range, num_pts, cmap_name, include_title=True, 
                        savename=None,**params):

    param_vals,p_step = np.linspace(*param_range, num=num_pts, retstep=True)
    mu_vals,mu_step = np.linspace(*mu_range, num=num_pts, retstep=True)

    fig,ax = plt.subplots()

    pts = []
    actions = set()
    
    for new_val in tqdm(param_vals):
        if isinstance(param_setter,str):
            params[param_setter] = new_val
        else:
            param_setter(params, new_val)

        fs = action_value.find_funcs(dn=0,**params)
        
        # list (x_i,a_i,func) upper bounds x_i, best action a_i, corresponding func
        best_actions = [(x,a,fs[a]) for x,a in integrals.find_intersection(fs)]
        upper_mu, best_control, f = best_actions.pop(0)

        c = []
        for mu in mu_vals:
            if not (upper_mu is None) and mu >= upper_mu: upper_mu, best_control, f = best_actions.pop(0)
            c.append(best_control)
            actions.add(best_control)
        pts.append(c)

    reduced_actions = {shorten(a,params['detail']) for a in actions}

    if 'glasbey' in cmap_name:
        colours = plt.get_cmap(cmap_name).colors
        colour_key = dict(zip(sorted(reduced_actions),colours))
    else:
        cmap = plt.get_cmap(cmap_name)

        key_d = {a:rho for a,(_,rho) in params['interventions'].items()}
        key_d.update({a:(1+cost) for a,(cost,_,_) in params['mon_dict'].items()})
        atree = tree_colours.Tree('',children=tree_colours.forest_from_list(reduced_actions,key=key_d.get))
        
        colour_key = {act:cmap(x) for (act,x) in tree_colours.colours_dict(atree,f=.7).items()}

    image = np.array([[colour_key[shorten(a,params['detail'])] for a in column] for column in pts])
    ax.imshow(np.transpose(image,axes=(1,0,2)), origin='lower', aspect='auto')

    patches = [ mpatches.Patch(color=colour_key[k], label=k) for k in colour_key if k in reduced_actions]
    ax.legend(handles=patches)

    if include_title: ax.set_title(f'Optimal control actions, at dn=0')
    ax.set_xlabel(name)
    ax.set_ylabel('abundance')

    rescaler_x = ticker.FuncFormatter(lambda x, pos: f'{param_vals[0]+x*p_step : .1f}')
    rescaler_y = ticker.FuncFormatter(lambda x, pos: f'{mu_vals[0]+x*mu_step : .1f}')
    ax.yaxis.set_major_formatter(rescaler_y)
    ax.xaxis.set_major_formatter(rescaler_x)

    if not savename is None:
        plt.savefig(f'figs/{savename}.png', dpi=300)
    plt.show()