# compute some necessary integrals for the value iteration

from math import log, exp, erf, sqrt

ROOT2 = sqrt(2)

# lognormal expectation of linear function (y = a + bx) 
# restricted to interval [l,u]
def int_lognorm (m, s, a, b, l, u): # log scale on bounds
    if l is None:
        lower_const = -1
        lower_lin = -1
    else:
        lower_const = erf((log(l)-m)/(s*ROOT2))
        lower_lin = erf((log(l)-m-s*s)/(s*ROOT2))

    if u is None:
        upper_const = 1
        upper_lin = 1
    else:
        upper_const = erf((log(u)-m)/(s*ROOT2))
        upper_lin = erf((log(u)-m-s*s)/(s*ROOT2))
    
    const = 0.5 * (upper_const - lower_const)
    lin = 0.5 * exp(m + s*s/2) * (upper_lin - lower_lin)

    return a*const + b*lin

# take in dict with values pairs (A,B) defining linear A+BX
# return list of pairs (x_1,a_1), (x_2,a_2), ... so that:
# x_1 < x_2 < ... and on interval x_{i-1} < X < x_i
# a_i in funcs gives min_a {A_a + B_aX}
# last x_i should be None indicating +inf
def find_intersection(funcs):
    if len(funcs) == 0:
        return []
    if len(funcs) == 1:
        return [(None,a) for a in funcs]

    best_a = min(funcs, key = funcs.get) # tuples compare lexicographically
    best_vals = funcs[best_a]

    redun = []
    int_points = {}
    for act in funcs:
        if funcs[act][0] >= best_vals[0] and funcs[act][1] >= best_vals[1]:
            redun.append(act)
        else:
            int_points[act] = (funcs[act][0] - best_vals[0])/(best_vals[1] - funcs[act][1])
    
    next_x = min(int_points.values(), default = None)

    if next_x is None:
        return [(None, best_a)]
    else:
        new_funcs = {act:(funcs[act][0]+funcs[act][1]*next_x-best_vals[0],funcs[act][1]) for act in int_points}
        next_ans = find_intersection(new_funcs)
        return [(next_x,best_a)] + [(x+next_x,a) for x,a in next_ans[:-1]] + [next_ans[-1]]

# d = {'a':(0.0,3.0), 'b':(1.0,1.0), 'c':(3.0,0.0), 'd':(0.0,2.0)}
# print(find_intersection(d))
