# -*- coding: utf-8 -*-
# @Author: Chen Cui
# @Date:   2021-01-09 11:02:45
# @Last Modified by:   Chen Cui
# @Last Modified time: 2021-04-03 01:06:21
# %%

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pandas as pd
# from mshr import *
from dolfin import *
from .misc import relative_error
#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

def mesh2triang(mesh):
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def plot_prediction(save_dir, truth, prediction):

    csv_data = pd.read_csv(
        '/home/kaijiang/cuichen/thermal_block/expriments/class1/PCA/prediction/prediction.csv')
    # rectangle = Rectangle(Point(-1., -1.), Point(1., 1.))
    # circle = Circle(Point(0., 0.), 0.5, segments=32)
    # domain = rectangle
    # domain.set_subdomain(1, circle)
    # domain.set_subdomain(2, rectangle - circle)
    # mesh = generate_mesh(domain, 60)

    mesh = Mesh("./thermal_block.xml")
    t = csv_data["t"].values
    mu_1 = csv_data["mu_1"].values
    mu_2 = csv_data["mu_2"].values

    # DL-ROM
    # idx = np.array([6378, 6389, 6398])
    # for i in idx:
    #     fig, ax = plt.subplots(figsize=(6,5))
    #     ax.axis('equal')
    #     cb = ax.tripcolor(mesh2triang(mesh), prediction[i])
    #     fig.colorbar(cb)
    #     fig.savefig(save_dir+'/DL-ROM_mu=[{:.2f},{:.2f},{:.2f},{:.2f}],t={:.2f}.png'.format(mu_1[i], mu_2[i],mu_3[i],mu_4[i],t[i]),
    #                 dpi=300, bbox_inches='tight', facecolor="#FFFFFF")
    #     plt.close()        
    #     fig, ax = plt.subplots(figsize=(6,5))
    #     ax.axis('equal')
    #     cb = ax.tripcolor(mesh2triang(mesh), truth[i])
    #     fig.colorbar(cb)
    #     fig.savefig(save_dir+'/FOM_mu=[{:.2f},{:.2f},{:.2f},{:.2f}],t={:.2f}.png'.format(mu_1[i], mu_2[i],mu_3[i],mu_4[i],t[i]),
    #                 dpi=300, bbox_inches='tight', facecolor="#FFFFFF")
    #     plt.close()     


    # DL-MC-ROM          
    for i in range(truth.shape[0]):
        if i % 100 == 0:              
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(1, 2, 1)
            ax.axis('equal')
            cb = ax.tripcolor(mesh2triang(mesh), truth[i])
            ax.set_title('truth idx={}'.format(i))
            fig.colorbar(cb)

            ax = fig.add_subplot(1, 2, 2)
            ax.axis('equal')
            cb = ax.tripcolor(mesh2triang(mesh), prediction[i])
            ax.set_title('pred idx={}'.format(i))
            fig.colorbar(cb)

            mre = relative_error(truth[i], prediction[i])
            fig.suptitle('mu=[{:.2f},{:.2f}],t={:.2f},mre={:.4f}.png'.format(mu_1[i], mu_2[i], t[i], mre), fontsize=10)
            fig.savefig(save_dir + '/pred_idx{}.png'.format(i), dpi=300, bbox_inches='tight')
            plt.close()
            


def plot_stat(save_dir, List):

    plt.semilogy(List, label='train_error')

    plt.legend()
    plt.title("Train error")
    plt.savefig(save_dir + "/train_error", dpi=600)
