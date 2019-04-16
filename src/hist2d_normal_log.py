#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:00:04 2019

@author: rainfall
"""

        # MAKE DATA ###################################################################
        x = y_test
        y = test_predictions
        # INIT FIGURE #################################################################
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        # AX1 #########################################################################
        H = ax1.hist2d(x, y, bins=50, cmin=1, cmap=plt.cm.jet)
        fig.colorbar(H[3], ax=ax1)
        ax1.set_title("Normal scale")
        ax1.set_xlabel("Observed rain rate (mm/h) - Training")
        ax1.set_ylabel("Predicted rain rate (mm/h) - Training")
        # AX2 #########################################################################
        xmin = np.log10(x.min())
        xmax = np.log10(x.max())
        ymin = np.log10(y.min())
        ymax = np.log10(y.max())

        xbins = np.logspace(xmin, xmax, 50) # <- make a range from 10**xmin to 10**xmax
        ybins = np.logspace(ymin, ymax, 50) # <- make a range from 10**ymin to 10**ymax
        
        print(xbins)
        print(ybins)
        
        counts, _, _ = np.histogram2d(x, y, bins=(xbins, ybins))
        
        print(counts)
        #
        #pcm = ax2.pcolormesh(xbins, ybins, counts)
        #plt.colorbar(pcm)
        #fig.colorbar(pcm, ax=ax2)  # this works too
        
        ## The following line doesn't actually work...
        ## See http://stackoverflow.com/questions/29175093/creating-a-log-linear-plot-in-matplotlib-using-hist2d
        H = ax2.hist2d(x, y, bins=[xbins, ybins], cmin=1, cmap=plt.cm.jet)
        fig.colorbar(H[3], ax=ax2)
        
        ax2.set_xscale("log")               # <- Activate log scale on X axis
        ax2.set_yscale("log")               # <- Activate log scale on Y axis
        
        ax2.set_xlim(xmin=xbins[0])
        ax2.set_xlim(xmax=xbins[-1])
        ax2.set_ylim(ymin=ybins[0])
        ax2.set_ylim(ymax=ybins[-1])
        
        ax2.set_xlabel("Observed rain rate (mm/h) - Training")
        ax2.set_ylabel("Predicted rain rate (mm/h) - Training")

        ax2.set_title("Log scale")
		
		        
        # SHOW AND SAVE FILE ##########################################################
        fig_name = self.fig_title + "_hist2D.png"
        plt.savefig(self.path_fig + fig_name)
        plt.clf()