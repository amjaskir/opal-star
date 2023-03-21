##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Make summary of complexity with specified ks
# using pregenerated images from hist.s
#
# Name: opal/sims/summary_hist.py
##################################################################


import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec

import sys
import os
import ast
import gc

def main(ks,pltn,env,variant=None,cmpme=False):

    levels = ["","_6","_9"]

    sz = 25
    plt.rcParams.update({'font.size': sz})

    #####################################################
    # some options

    # restrict alpha actor to range with progressive performance
    restrict = False

    #anneal?
    anneal = True
    T = 100

    #####################################################

    # specify root directory search
    if anneal:
        base = "results/anneal_%d/complexity_500_5000/" %T
    elif (variant is None) or (cmpme == True):
        # general summary of DA mod model
        base = "results/complexity_500_5000/"
    else:
        # general summary of variant model
        base = "results/complexity_%s_500_5000/" %(variant)

    # specify save directory
    if restrict:
        fname = "%ssummary_%dtrials_r8_%s.pdf" %(env,pltn,str(ks))
    else:
        fname = "%ssummary_%dtrials_%s.pdf" %(env,pltn,str(ks))
    if (cmpme == True):
        # save model comparisons in separate folder
        savename = base + "summaries/" + "cmp/" + "%s_"%(variant)
    else:
        savename = base + "summaries/"
    os.makedirs(savename, exist_ok=True)
    savename = savename + fname 

    with PdfPages(savename) as pdf:
        # print first page with complexity summary
        # across levels
        base_env = base + "%s/" %env
        fig, ax = plt.subplots(1,len(ks))
        fig.set_size_inches(22, 17)
        fig.suptitle("%s" %env, fontsize=sz*2)

        # only save specified ks
        for idx,k in enumerate(ks):

            if restrict:
                getme = base_env + "ntrials%d_r8/" %(pltn)
            else:
                getme = base_env + "ntrials%d/" %(pltn)
            if cmpme == True:
                getme = getme + "cmp_%s/"%(variant)
            getme = getme + "k%dlearning.png" %(k)

            img = mpimg.imread(getme)
            plt.axis('off')
            ax[idx].imshow(img)
            ax[idx].set_title("k = %d" %k)
            plt.ylabel("Avg AUC (Mod - Bal)/Bal")
            plt.xlabel("Complexity")
            plt.xticks([], [])
            plt.yticks([], [])
            plt.axis('off')

        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()

        ## uncomment below to plot the complexity curves for all ks
        # fig, ax = plt.subplots()
        # fig.set_size_inches(22, 17)
        # if restrict:
        #     getme = base_env + "ntrials%d_r8/learning.png" %(pltn)
        # else:
        #     getme = base_env + "ntrials%d/learning.png" %(pltn)
        # img = mpimg.imread(getme)
        # ax.imshow(img)
        # plt.title("%s" %env, fontsize=sz*2)
        # plt.ylabel("Avg AUC (Mod - Bal)/Bal")
        # plt.xlabel("Complexity")
        # plt.xticks([], [])
        # plt.yticks([], [])
        # pdf.savefig()  # saves the current figure into a pdf page
        # plt.close()

        del img
        gc.collect()

        # go through select levels 
        for l in levels:
            print("level %s" %l)
            this_l = env + l

            # print second page hist summary of ks
            fig, ax = plt.subplots(2,len(ks))
            plt.axis('off')
            fig.suptitle("%s" %this_l)
            plt.axis('off')
            fig.set_size_inches(22, 17)
            plt.axis('off')
            for idx,k in enumerate(ks):

                if restrict:
                    getme = base_env + "%s/k_%s/ntrials%d_r8/" \
                        %(this_l,float(k),pltn)
                else:
                    getme = base_env + "%s/k_%s/ntrials%d/" \
                        %(this_l,float(k),pltn)
                if cmpme == True:
                    getme = getme + "cmp_%s/"%(variant)

                img1 = mpimg.imread(getme + "diff_Learning.png")
                plt.axis('off')
                ax[0,idx].imshow(img1)
                plt.axis('off')
                ax[0,idx].set_title("k=%s" %k)
                plt.axis('off')
                img2 = mpimg.imread(getme + "learnhist.png")
                plt.axis('off')
                ax[1,idx].imshow(img2)
                plt.axis('off')
            plt.axis('off')
            pdf.savefig()  # saves the current figure into a pdf page
            plt.close()
            del img1, img2
            gc.collect()

            # print learning curves and scatters for each k
            for idx,k in enumerate(ks):

                fig = plt.figure()
                fig.set_size_inches(22, 17)

                s1 = plt.subplot2grid((3, 5), (0, 0), colspan=1, rowspan=1)
                plt.axis('off')
                s2 = plt.subplot2grid((3, 5), (1, 0), colspan=1, rowspan=1)
                plt.axis('off')
                s3 = plt.subplot2grid((3, 5), (2, 0), colspan=1, rowspan=1)
                plt.axis('off')
                s4 = plt.subplot2grid((3, 5), (0, 1), colspan=4, rowspan=3)
                plt.axis('off')

                fig.suptitle("%s k=%d" %(this_l,k))

                if restrict:
                    getme = base_env + "%s/k_%s/ntrials%d_r8/" \
                        %(this_l,float(k),pltn)
                else:
                    getme = base_env + "%s/k_%s/ntrials%d/" \
                        %(this_l,float(k),pltn)
                if cmpme == True:
                    getme = getme + "cmp_%s/"%(variant)

                # scatter graphs
                img1 = mpimg.imread(getme + "diff_by_aucLearning.png")
                s1.imshow(img1)
                plt.axis('off')
                img2 = mpimg.imread(getme + "diff_by_auc_alphaaLearning.png")
                s2.imshow(img2)
                plt.axis('off')
                img3 = mpimg.imread(getme + "diff_by_auc_betaLearning.png")
                s3.imshow(img3)
                plt.axis('off')
                del img1, img2, img3
                gc.collect()

                # curves
                img_larger = mpimg.imread(getme + "allcurves.png")
                s4.imshow(img_larger)
                plt.axis('off')
                plt.tight_layout()
                pdf.savefig(dpi = 400) 
                plt.close()

                del img_larger
                gc.collect()
            print("end ks")
        print("end levels")

if __name__ == '__main__':
    ks = ast.literal_eval(sys.argv[1])
    if len(sys.argv) <= 4:
        main(ks,int(sys.argv[2]),str(sys.argv[3]))
    else:
        main(ks,int(sys.argv[2]),str(sys.argv[3]),str(sys.argv[4]),bool(int(sys.argv[5])))