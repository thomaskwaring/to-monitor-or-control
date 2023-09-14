import plotting_functions
import simulations
import math

# "control" parameters
ce = 4 # cost
rho = 0.6 # efficacy

# with multiple control options, use a dictionary (these are two options)
# the second offers "control twice in one timestep"
interventions = {'ignore':(0.0,0.0), 'control':(ce,rho)}
more_interventions = {'ignore':(0.0,0.0), 'control':(ce,rho), 'control^2':(2*ce,1-(1-rho)**2)}

# "monitoring" parameters
cm = 1 # cost
err_mon = 0.5 # std error in the monitoring process
# specifically, we model the monitoring as a sample (of log(N))
# from a normal distribution with mean ~ true value, given std error
# for m measurements, divide error by m

# specify these in a dictionary, triples (cost, error, control efficacy)
# these are some options
one_mon = {'monitor':(cm,.1,0)}
cont_mon = {'monitor':(cm,err_mon/2,0.0),'control/monitor':(cm+ce,err_mon,rho)}
two_mon = {'monitor1':(cm,0.5,0.0), 'monitor2':(2*cm,0.1,0.0)}

waiting_steps = None # max number of steps to wait before monitoring, None = no limit

# disease parameters
r = 1.2 # growth rate
dr = 0.05 # uncertainty in growth rate

gamma = .9 # discount
num_steps = 3 # horizon, plot involves this many "choices"

# plot parameters
mu_range = (0,1.5) # range of mu values
s_range = (0,3) # range of s values

num_sqs = 1000 # resolution, plots grid of num_sqs x num_sqs pixels

# how many steps to put in the legend (gets hard to read with >= 3)
# given "None" defaults to "num_steps"
detail = None

# collect them all to pass around easily
params = dict(interventions=more_interventions,
                mon_dict=one_mon,
                gamma=gamma,
                r=r,
                dr=dr,
                num_steps=num_steps,
                waiting_steps=waiting_steps)

# simpler version of the above, for the first plot in the manuscript
simple_params = dict(interventions=interventions,
                     mon_dict=one_mon,
                     gamma=gamma,
                     r=r,
                     dr=dr,
                     num_steps=2,
                     waiting_steps=waiting_steps)


# some possible colourmaps ("rainbow" is probably best)
rainbow = 'cet_rainbow4'
colourblind = 'cet_CET_CBL2'
gould = 'cet_gouldian'


# do the plots

# figure 1
plotting_functions.plot_regions(mu_range,s_range,num_sqs,detail,
    colormap=rainbow,fname=None,**simple_params)

# figure 2
plotting_functions.plot_regions(mu_range,s_range,num_sqs,detail,
    colormap=rainbow,fname=None,**params)

# if fname is not None, the above saves data into a file
# you can plot from this using the function below
# (good for testing colours, etc)

# load image from fname
# the files data/simple and data/main are the previous two plots
# plotting_functions.plot_from_file('simple',detail=1,colormap=rainbow,f=0.9,title='Optimum management actions')
# plotting_functions.plot_from_file('main',detail=2,colormap=rainbow,f=0.9,title='Optimum management actions')




# the following defines auxillary parameters and functions, for the sensitivity plots (figs 3 and 4)

def set_cm (p,x):
    p['mon_dict']['monitor'] = (x,err_mon,0)

def set_err (p,x):
    p['mon_dict']['monitor'] = (cm,x,0)

def set_ce (p,x):
    cur_rho = p['interventions']['control'][1]
    p['interventions']['control'] = (x,cur_rho)
    if 'control^2' in p['interventions']:
        p['interventions']['control^2'] = (2*x,1-(1-cur_rho)**2)
ce_ps = (set_ce, (3,5), 'ce')

def set_rho (p,x):
    cur_ce = p['interventions']['control'][0]
    p['interventions']['control'] = (cur_ce,x)
    if 'control^2' in p['interventions']:
        p['interventions']['control^2'] = (2*cur_ce,1-(1-x)**2)
rho_ps = (set_rho, (0.1,0.9), 'rho')

def set_gamma_r (p,x):
    const = p['gamma'] * math.exp(p['r'])
    p['gamma'] = x
    p['r'] = math.log (const / x)
gamma_ps = (set_gamma_r, (0.5,0.9), 'gamma')
gamma_ps_simple = ('gamma', (0.5,0.9), 'gamma')



# figure 3
plotting_functions.varying_crossovers(*ce_ps, mu_range=(0,2),
                                            num_pts=1000, cmap_name=gould, detail=2, include_title=False, **params)
plotting_functions.varying_crossovers(*rho_ps, mu_range=(0,2),
                                            num_pts=1000, cmap_name=gould, detail=2, include_title=False, **params)

# figure 4
plotting_functions.varying_frontiers(set_cm, 
                                           [.5,1,1.5,2],'cm','monitor',(0,2),s_range,
                                           num_pts=100,cmap_name='cet_bmy',include_title=False,**params)
plotting_functions.varying_frontiers(set_err, [0,.2,.4,.6,.8,1],'err_mon','monitor',(0,2),s_range,
                                           num_pts=100,cmap_name='cet_bmy',include_title=False,**params)


# plots for the simulations and case study

# parameters & initial values for the fire ant case study
fa_init_dn = 2
fa_init_mu = .25
fa_init_n = math.log(fa_init_mu) - fa_init_dn**2/2
fire_ant_params = {'gamma': 0.9, 'num_steps': 2, 'detail': None, 'r': 2.82, 'dr': 0.015, 
                'interventions': {'ignore': (0.0, 0.0), 'control': (31.82244784482759, 0.9989054351771577)}, 
                'mon_dict': {'monitor': (6.204010721551725, 0, 0)}, 'seq_len':5}


from scipy.stats import norm
import pandas as pd

# plot timeseries as a sequence of actions and states
q = .8
fa_init_true_n = norm.ppf(q, loc=fa_init_n, scale=fa_init_dn)
seq = simulations.run_sim(fa_init_true_n, fa_init_n, fa_init_dn, **fire_ant_params)[0]['full']
simulations.plot_seq(pd.DataFrame(seq))


# add timeseries to phase plot

s_range = (0,3)
mu_range = (0,1)
num_sqs = 1000

pts = [(math.exp(d['n'] + d['dn']**2/2), d['dn']) for d in seq]
plotting_functions.plot_regions(mu_range,s_range,num_sqs,pts_to_plot=pts,**fire_ant_params)

# plot payoffs for the naive & full models
simulations.plot_payoffs(init_mu=0.3, init_dn=1.8, num_sims=1000, num_qs=100, **params)