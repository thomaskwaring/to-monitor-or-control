# divide a range into segments corresponding to nodes on a tree

from itertools import groupby
import colorcet as cc
import matplotlib.pyplot as plt

# divide into equal segments
def divide_range (r, n):
    d = (r[1] - r[0])/n
    return [(r[0]+i*d, r[0]+(i+1)*d) for i in range(n)]

# shrink an interval towards its midpoint
def shrink_range(r, f):
    L = r[1] - r[0]
    diff = (1-f) * L / 2
    return (r[0] + diff, r[1] - diff)

def midpoint (r):
    return (r[0] + r[1]) / 2

class Tree:

    def __init__ (self, label, children=[]):
        self.label = label
        self.children = children

    def __len__ (self):
        return len(self.children)

    # root not included
    def dec_as_list (self, sep=',', inc_int=False):
        return [child.label + sep + x for child in self.children 
            for x in child.dec_as_list(inc_int=inc_int)] + [child.label for child in self.children if len(child) == 0]

def forest_from_list (l, sep=',',key=None):
    if not key is None: 
        key_func = lambda x: tuple(map(key,x))
        paths = sorted([x.split(sep) for x in l if len(x)>0], key=key_func)
    else:
        paths = sorted([x.split(sep) for x in l if len(x)>0])
    forest = []
    for fst,paths in groupby(paths, lambda x: x[0]):
        after = [sep.join(p[1:]) for p in paths]
        forest.append(Tree(fst, forest_from_list(after,sep=sep,key=key)))
    return forest

# complete structure given by T(N,L) = N*E(T) + L
def make_trees (node_labels, leaf_labels, depth):
    lleaves = [Tree(l) for l in leaf_labels]
    # nleaves = [Tree(l) for l in node_labels] if depth <= 1 else []
    dec = make_trees(node_labels,leaf_labels,depth-1) if depth > 1 else []
    return lleaves + [Tree(l,children=dec) for l in node_labels]

def make_action_trees (cont_actions, mon_actions, depth):
    if depth > 1:
        leaves = [Tree(l) for l in mon_actions]
        dec = make_action_trees(cont_actions,mon_actions,depth-1)
        return leaves + [Tree(l,children=dec) for l in cont_actions]
    else:
        return [Tree(l) for l in cont_actions]

# def stretch_colours_dict (d):
#     min_s,max_s = min(d.values()),max(d.values())
    

def colours_dict (tree, r=(0,1), f=0.75, sep=','):
    # d = {tree.label:midpoint(r)}
    d = {}
    # d[tree.label] = midpoint(r)
    
    if len(tree) > 0:
        for i,child in zip(divide_range(r, len(tree)), tree.children):
            d[child.label] = midpoint(i)
            if len(child) > 0:
                child_cols = colours_dict(child,r=shrink_range(i,f), f=f)
                for l,c in child_cols.items():
                    d[child.label + sep + l] = c
            
    return d

def display_colours (params, colourkey, cmap_name, f):
    key_d = {a:rho for a,(_,rho) in params['interventions'].items()}
    key_d.update({a:(1+cost) for a,(cost,_,_) in params['mon_dict'].items()})
    tree = Tree('',children=forest_from_list(colourkey.keys(),key=key_d.get))

    scalar_d = colours_dict(tree, f=f)
    scalar_d = {a:scalar_d[a] for a in scalar_d if a in colourkey}
    cmap = plt.get_cmap(cmap_name)
    colours_d = {a:cmap(x) for a,x in scalar_d.items()}

    fig,ax = plt.subplots()

    plt.axis('off')

    ht = 100
    img=[[cmap(i/256) for i in range(256)] for _ in range(ht)]
    ax.imshow(img)

    ax.vlines([x*256 for x in scalar_d.values()],ymin=0,ymax=ht,colors='w')

    for a,x in scalar_d.items():
        ax.text(x=x*256,y=-2,s=a,rotation=45)

    plt.show()
