from math import exp, log
import itertools

import integrals

# carry out action-value computations

def reward(n, dn): 
    return exp(n + dn*dn/2)

# compute the action-value functions for control interventions only
# computation is section 2.3.2 of the manuscript
def find_funcs (gamma, r, dr, dn, interventions, num_steps, **_):
    near = exp((dn*dn)/2)
    if num_steps == 1:
        far = exp(r+(dn+dr)**2/2)
        funcs = {a:(interventions[a][0],near+gamma*far*(1-interventions[a][1])) for a in interventions}
    else:
        next_step = find_funcs(gamma,r,dr,dn+dr,interventions,num_steps-1)
        funcs = {}
        for a in interventions:
            for b in next_step:
                funcs[a+','+b] = (interventions[a][0]+gamma*next_step[b][0], 
                    near+gamma*next_step[b][1]*exp(r)*(1-interventions[a][1]))
    return funcs

# action-value for control interventions
def Q_interventions (n, dn, gamma, r, dr, interventions, num_steps,**_):
    funcs = find_funcs(gamma,r,dr,dn,interventions,num_steps)
    Q_dict = {a:(funcs[a][0]+funcs[a][1]*exp(n)) for a in funcs}

    return Q_dict

# compute the cumulative effect of a sequence of controls
def action_effects (a_seq,interventions):
    return sum([log(1-interventions[a][1]) for a in a_seq])

# Bayesian update rule for uncertainty
def new_dn (dn,err): return dn*err/(dn+err)

# calculate individual cost
def Q (action,n,dn,mon_dict,gamma,r,dr,interventions,num_steps, **_):
    a_list = action.split(',')
    if a_list[-1] in mon_dict:
        return Q_up_to_monitor(a_list,n,dn,mon_dict,gamma,r,dr,interventions,num_steps)
    f = find_funcs(gamma,r,dr,dn,interventions,num_steps)[action]
    return f[0] + f[1]*exp(n)

# auxillary functions in the monitoring case
def Q_up_to_monitor(a_list,n,dn,mon_dict,gamma,r,dr,interventions,num_steps):
    if len(a_list) == 1:
        return single_Q_monitor(a_list[0],n,dn,mon_dict,gamma,r,dr,interventions,num_steps)
    else:
        a = a_list[0]
        return reward(n,dn) + interventions[a][0] + gamma * Q_up_to_monitor(a_list[1:],
                n+r+log(1-interventions[a][1]),dn+dr,mon_dict,gamma,r,dr,interventions,num_steps-1)

def single_Q_monitor (mon_a, n, dn, mon_dict, gamma, r, dr, interventions, num_steps,**_):
    cost,err,rho = mon_dict[mon_a]

    new_dn = dn*err / (dn + err)
    funcs = find_funcs(gamma,r,dr,new_dn + dr,interventions,num_steps-1)
    
    pieces = integrals.find_intersection(funcs)

    lower = None
    future = 0
    for piece in pieces:
        act = piece[1]
        # exp(r) to account for growth before funcs
        future += integrals.int_lognorm(n, dn, funcs[act][0], funcs[act][1]*exp(r)*(1-rho), lower, piece[0])
        lower = piece[0]
    
    return reward(n,dn) + cost + gamma*future


def Q_mon (n,dn,mon_dict,gamma,r,dr,interventions,num_steps,waiting_steps=None,**_):
    if waiting_steps is None: max_waiting = num_steps-2
    else: max_waiting = min(num_steps-2,waiting_steps)

    waiting_times = list(range(max_waiting+1))

    vals = {}

    for wait in waiting_times:
        before_seqs = list(itertools.product(interventions,repeat=wait))
        
        ns_before_mon = {a_seq: wait*r + action_effects(a_seq,interventions) for a_seq in before_seqs}
        dn_before_mon = dn + dr*wait

        if wait > 0:
            before_reward_fs = find_funcs(gamma,r,dr,dn,interventions,num_steps=wait)
            reward_before_mon = {a_name:f[0] + f[1]*exp(n) for a_name,f in before_reward_fs.items()}
        else:
            reward_before_mon = {'':reward(n,dn)}

        after_mon_fs = {mon_a:find_funcs(gamma,r,dr,new_dn(dn_before_mon,mon_dict[mon_a][1])+dr,interventions,num_steps-wait-1) 
            for mon_a in mon_dict}
        pieces = {mon_a:integrals.find_intersection(fs) for mon_a,fs in after_mon_fs.items()}

        for a_seq in before_seqs:
            a_name = ','.join(a_seq)

            for mon_a in mon_dict:
                cost,err,rho = mon_dict[mon_a]

                lower = None
                future = 0
                for piece in pieces[mon_a]:
                    act = piece[1]
                    c,slope = after_mon_fs[mon_a][act]

                    # exp(r) to account for growth before funcs
                    future += integrals.int_lognorm(n+ns_before_mon[a_seq], dn_before_mon, c,slope*exp(r)*(1-rho), lower, piece[0])
                    lower = piece[0]

                vals[','.join(a_seq+(mon_a,))] = reward_before_mon[a_name] + gamma**wait * (cost + gamma*future)
    
    return vals

# compute all action value functions,
# this is done more efficiently for the plots themselves
def Q_all (n,dn,mon_dict,gamma,r,dr,interventions,num_steps,**_):
    Q_dict = Q_interventions(n,dn,gamma,r,dr,interventions,num_steps)
    Q_mon_actions = Q_mon(n,dn,mon_dict,gamma,r,dr,interventions,num_steps,num_steps)
    Q_dict.update(Q_mon_actions)
    return Q_dict