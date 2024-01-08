# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 10:44:00 2023

@author: Maria
"""

'''
This file contains the functions that perform the post-processing of MCMC 
chains outlined in the notes by Meixi (July 2023). The "Main Program
Functions" section contains the main functions for loading the chains
and computing the model posteriors. The "Analysis Functions" section 
contains other functions that do various analyses of the chains and 
model posterior dependencies. The "Helper Functions" section contains
any other functions either used in the main functions or that may be 
useful for spontaneous consistency checks. All the functions are 
demonstrated in the "DEMO_postProcessing" Jupyter notebook.
'''

# Imports
import numpy as np
import pandas as pd
import scipy.stats as st
import math
import statsmodels.stats.weightstats as ssw

from copy import deepcopy
import time
import os
import pathlib

import seaborn as sns
import matplotlib.pyplot as plt
import getdist as gd
from getdist import plots # for some reason I need to do this one separately

# Main Program Functions ------------------------------------------------------

# Function to load chains into pandas DataFrame
def dfChains(roots,models,chainDir,burnin,stacked=True,nmax='all'):
    '''
    Loads Cobaya chains into DataFrame structures with the appropriate 
    column headers and returns a dictionary containing the chains corresponding
    to each root provided.

    Parameters
    ----------
    roots : list of str
        Root names of chains to load.
    models : list of str
        Names of models corresponding to 'roots'.
    chainDir : str
        Chain directory from which to retrieve chains.
    burnin : list of float or int
        List of fractional burn-in OR number of burn-in samples to remove from 
        chains.
    stacked : bool, optional
        If True, return all chains stacked together for each root. The default
        value is True.
    nmax : either 1, 'all' or a fraction 0<=nmax<=1
        The fraction of the total chains' length to use (after burn-in has been removed)

    Returns
    -------
    all_chains : dict of DataFrames
        Each key is a root file name, and its corresponding entry is the 
        chains of that root.

    '''
    
    # start = time.time()
    all_chains = {}
    chainPath = pathlib.Path(chainDir) # turn chain directory into path object
    
    for k,r in enumerate(roots):
        print('Root:',r)
        chains = []
        for file in chainPath.iterdir():
            if (r in str(file) and '.txt' in str(file)
                and 'minimum' not in str(file)
                and r == str(file)[len(chainDir):len(chainDir)+len(r)]):
                # print('\tFile:',file)
                
                # Read header from file
                f = open(file,'r')
                for line in f:
                    split = line.split(' ')
                    # print(split)
                    for i,el in enumerate(split):
                        if '\n' in el:
                            split[i] = el.replace('\n','')
                        elif '#' in el:
                            split[i] = el.replace('#','')
                    spaces = split.count('')
                    for s in range(spaces):
                        split.remove('')
                    # print(split)
                    break
                f.close()
                header = split
                
                # Load data
                # print(file)
                data = np.loadtxt(file)
                # print(data)
                
                # Create DataFrame 
                # print(header)
                samples = pd.DataFrame(data,columns=header)
                
                # Remove burn-in
                if burnin[k] < 1:
                    burnin_ind = int(burnin[k]*len(samples))
                else:
                    burnin_ind = int(burnin[k])
                if nmax == 'all' or nmax == 1:
                    burned_samples = samples.loc[burnin_ind:]
                else:
                    burned_samples = samples.loc[burnin_ind:]
                    max_index = int(len(burned_samples)*nmax)
                    burned_samples = burned_samples[:max_index]
                
                chains.append(deepcopy(burned_samples))
                    
        # print(stacked)
        if stacked:
            # Stack chains for each model
            stacked_chains = pd.concat(chains,ignore_index=True)
            all_chains[models[k]] = stacked_chains
        else:
            # Return each chain independently
            all_chains[models[k]] = chains
            
    # print(time.time()-start,'s')
                
    return all_chains

# Function to calculate likelihoods from chain data: -log(posterior)s and 
## -log(prior)s
def get_lik(chains,stacked=True):
    '''
    Derives the joint likelihood values from the joint -log(posterior) and 
    joint -log(prior) values for MCMC samples.

    Parameters
    ----------
    chains : dict of DataFrame
        Contains MCMC samples with -log(posterior) and -log(prior) values
        for the joint distributions. The keys are the model names.
    stacked : bool, optional
        If True, the loaded chains are stacked for each model. If False, each
        chain was loaded separately. The default is True.

    Returns
    -------
    likelihoods : dict of Series
        Joint likelihood values for MCMC samples for each set of chains.

    '''
    
    # Calculate -log(likelihood)s
    # 'minuslogprior__0' is the separable joint prior over all parameters
    minuslogliks = {}
    for m in chains:
        # m = model name
        if stacked:
            minusloglik = chains[m]['minuslogpost'] - chains[m]['minuslogprior']
        else:
            minusloglik = [s['minuslogpost'] - s['minuslogprior'] for s in 
                           chains[m]]
        minuslogliks[m] = minusloglik
    
    # Get common normalization factor
    mins = []
    for m in minuslogliks:
        if stacked:
            mins.append(np.amin(minuslogliks[m]))
        else:
            for s in minuslogliks[m]:
                mins.append(np.amin(s))
    minlik = max(mins) # absolute minimum over all chains
        
    # Compute likelihoods
    if stacked:
        likelihoods = {m:np.exp(-(minuslogliks[m] - minlik)) for m in minuslogliks}
    else:
        likelihoods = {}
        for m in minuslogliks:
            liks = [np.exp(-(s - minlik)) for s in minuslogliks[m]]
            likelihoods[m] = liks
    
    return likelihoods
        
# Function to compute model likelihoods
def model_lik(likelihoods,stacked=True):
    '''
    Computes the estimate of the likelihood of the model which the MCMC data
    depend on given the likelihood associated with that model.

    Parameters
    ----------
    likelihoods : dict of Series
        Contains joint likelihood values corresponding to MCMC samples for 
        each model.
    stacked : bool, optional
        If True, the loaded chains are stacked for each model. If False, each
        chain was loaded separately. The default is True.

    Returns
    -------
    mliks : dict of float
        Estimate of model likelihood for each model.

    '''
    
    mliks = {}
    for m in likelihoods:
        # m = model name
        
        if stacked:
            N = len(likelihoods[m]) # number of samples
            inds = likelihoods[m].index
        
            insum = [1/likelihoods[m][i] for i in inds]
            infcount = insum.count(np.inf) # happens if likelihood = 0
            # print(infcount/N)
            for i in range(infcount):
                insum.remove(np.inf)
            mlik = (N-infcount)/np.sum(1/likelihoods[m][:])
            mliks[m] = mlik
        else:
            liks = []
            for s in likelihoods[m]:
                # do the computation for each chain
                N = len(s) # number of samples
                inds = s.index
            
                insum = [1/s[i] for i in inds]
                infcount = insum.count(np.inf) # happens if likelihood = 0
                # print(infcount/N)
                for i in range(infcount):
                    insum.remove(np.inf)
                mlik = (N-infcount)/np.sum(insum)
                liks.append(mlik)
        
            mliks[m] = np.mean(liks) # average likelihood over all chains
    
    return mliks

# Function to compute model posteriors
def model_post(models,model_liks,priors):
    '''
    Computes the model posteriors for each model that corresponds to the 
    likelihoods and priors passed.

    Parameters
    ----------
    models : list of str
        List of model identifiers.
    model_liks : dict of float
        Contains the likelihood for each model, as computed from the MCMC 
        samples.
    priors : dict of float
        Contains the prior for each model, as specified by user.

    Returns
    -------
    posts : dict of float
        Contains posterior probability for each model.

    '''
    
    posts = {} # model posteriors
    
    # Compute normalization
    arr_liks = np.array(list(model_liks.values())) # for nice math
    arr_priors = np.array(list(priors.values()))
    sumposts = sum(arr_liks*arr_priors)
    
    # Calculate model posteriors
    for m in models:
        mpost = model_liks[m]*priors[m]/sumposts
        posts[m] = mpost
        
    return posts

# Get reweighted chains
def reweight(chains,model_posts):
    '''
    Calculates the new weights for all samples in the set of chains based on
    the model posteriors and the previously-assigned weights from Cobaya.

    Parameters
    ----------
    chains : dict of DataFrame
        Contains chains from each model.
    model_posts : dict of float
        Model posteriors.

    Returns
    -------
    newchains : dict of DataFrame
        Returns 'chains' with an additional entry: the full stack of all the 
        chains, with a 'model_weights' column that provides the weight for 
        each sample.

    '''
    
    newchains = {}
    for m in chains:
        # m = model name
        
        # print(m)
        # print(chains[m])
        N = len(chains[m]) # number of samples 
        # weights = [model_posts[m]*chains[m]['weight'][n] for n in range(N)] 
        weights = [((N*0.1)/N)*model_posts[m]*chains[m]['weight'][n] for n in range(N)] 
            # weight every sample by the model posterior times the Cobaya weight
        editchain = deepcopy(chains[m])
        # print(editchain)
        editchain.insert(0,'model_weights',weights)
        newchains[m] = editchain
    # print(newchains)
        
    full_stack = pd.concat(list(newchains.values()),ignore_index=True)
    newchains['Reweighted'] = full_stack
    
    return newchains

# Run full computation to retrieve original and reweighted chains
def run(roots,models,chainDir,burnin,priors,temperature):
    '''
    Computes the model posteriors from the chains given in 'roots', using 
    'burnin' fractional burn-in and the specified model priors, 'priors'.

    Parameters
    ----------
    roots : list of str
        List of chain root names to analyse.
    models : list of str
        Model identifiers corresponding to each set of chains specified in 
        'roots'.
    chainDir : str
        Directory to fetch chains from.
    burnin : float
        Fractional burn-in to remove from chain samples when loading.
    priors : dict of float
        Model priors for each model in 'models'.
    temperature: dict of float
        Chain temperature - used to retrieve the correct -logP

    Returns
    -------
    stuff : dict
        Entries:
            chains : dict of DataFrame
                Loaded MCMC chains.
            likelihoods : dict of Series
                Each Series holds the likelihood values for the MCMC samples 
                from that chain.
            model_liks : dict of float
                Model likelihoods as computed from chains.
            model_posts : dict of float
                Model posteriors.
            newchains : dict of DataFrame
                Loaded MCMC chains with the additional DataFrame containing the
                stack of all chains reweighted according to the model 
                posteriors.

    '''
    
    print('Models:',models)
    
    # Load MCMC samples
    print('\nLoading chains from roots:',roots)
    print('\tUsing burn-in',burnin)
    chains = dfChains(roots,models,chainDir,burnin) # assume always using stacked chains
    for m in models:
        chains[m]['minuslogpost'] *= temperature[m]

    # Compute model likelihoods
    print('Computing model likelihoods')
    likelihoods = get_lik(chains)
    model_liks = model_lik(likelihoods)
    
    # Compute model posteriors
    print('Computing model posteriors')
    model_posts = model_post(models,model_liks,priors)
    
    # Construct reweighted chains
    print('Reweighting chains')
    newchains = reweight(chains,model_posts)
    
    stuff = {'chains':chains,'likelihoods':likelihoods,'model_liks':model_liks,
             'model_posts':model_posts,'newchains':newchains}
    
    return stuff

# Analysis Functions -----------------------------------------------

# Function to calculate and plot prior-dependence of model posteriors
def prior_dependence(model_likelihoods,leg_labs=None,subtitle='',
                     saveFig=False,imgDir=None,figName=None,showFig=True,
                     context='notebook',dataDir=None):
    '''
    This function calculates the model posteriors from the model likelihoods
    for TWO models for a series of different model priors. The purpose is to 
    assess the prior-dependence of the model posteriors. This dependence is
    shown in a plot as well as returned as numerical data.

    Parameters
    ----------
    model_likelihoods : dict of float
        Likelihood for each model.
    leg_labs : dict of str, optional
        Labels for each chain in 'chains' to be reported on the plot legend.
        The default is the keys of 'chains'.
    subtitle : str, optional
        Additional information to put in the title of the plot, if desired.
        The default is an empty string.
    saveFig : bool, optional
        Whether to save the figure. The default is False.
    imgDir : str, optional
        Directory to save image to. The default is the current working 
        directory.
    figName : str, optional
        Name to save figure under (excluding extension). The default is 
        "margePostComp_[param]".
    showFig : bool, optional
        Whether to show figure in console. The default is True.
    context : None, dict, or one of {paper, notebook, talk, poster}, optional
        The 'context' variable to use in ``seaborn.set_theme`` to set the style
        of the plot. The default is 'notebook'.
    dataDir : str, optional
        Directory to save prior and posterior data to. The default is the 
        current working directory.

    Returns
    -------
    data : DataFrame
        Contains the first model priors and the corresponding model posteriors
        for each model.

    '''
    
    models = list(model_likelihoods.keys()) # list of models
    
    if not imgDir:
        imgDir = os.getcwd()
    if not figName:
        figName = 'model_posterior_priorDependence'
    if leg_labs == None:
        leg_labs = {k:k for k in models} # just label with keys
    if not dataDir:
        dataDir = os.getcwd()
    
    # Compute model posteriors
    priors = np.arange(0.05,1.0,0.05) # prior likelihoods of first model to 
        # loop over
    M1_posts = [] # posteriors of first model
    M2_posts = [] # posteriors of second model
    for i,pr in enumerate(priors):
        
        # Set model priors
        # Sum of priors = 1
        model_priors = {models[0]:pr,models[1]:1-pr}
        
        # Compute model posteriors
        model_posts = model_post(models,model_likelihoods,model_priors)
        M1_posts.append(model_posts[models[0]])
        M2_posts.append(model_posts[models[1]])
        
    # Plot results
    sns.set_theme(context=context,style='white',palette='Set2')
    
    plt.plot(priors,M1_posts,'-o',label=leg_labs[models[0]])
    plt.plot(priors,M2_posts,'-o',label=leg_labs[models[1]])
    plt.legend(loc='best',frameon=False)
    plt.xlabel(leg_labs[models[0]]+' Model Prior Probability')
    plt.ylabel('Model Posterior Probability')
    plt.title('Prior Dependence of Model Posteriors'+subtitle)
    if saveFig:
        plt.savefig(imgDir+figName+'.pdf',dpi=300) # dpi = dots per inch (res)
    if showFig:
        plt.show()
        
    # Create data structuce
    data = pd.DataFrame(columns=[leg_labs[models[0]]+' Prior',
                                 leg_labs[models[0]]+' Posterior',
                                 leg_labs[models[1]]+' Posterior'])
    data[leg_labs[models[0]]+' Prior'] = priors
    data[leg_labs[models[0]]+' Posterior'] = M1_posts
    data[leg_labs[models[1]]+' Posterior'] = M2_posts
    
    # Save data to .csv file
    data.to_csv(dataDir+figName+'.csv',index=False)
    
    return data

# Burn-in dependence
def burnin_dependence(roots,models,chainDir,burnin=0,prior='flat',
                      leg_labs=None,subtitle='',saveFig=False,imgDir=None,
                      figName=None,showFig=True,context='notebook'):
    '''
    Computes burn-in dependence of model posteriors for a set of model chains.

    Parameters
    ----------
    roots : list of str
        Chains to analyze.
    models : list of str
        Model identifiers corresponding to each root.
    chainDir : str
        Directory to load chains from.
    burnin : float or int, optional
        Fractional burn-in OR number of burn-in samples to reove. The default 
        is 0.
    prior : one of {'flat','occam'} or dict of float, optional
        The model prior to impose on chains. 'flat' imposes a 50/50 prior while
        'occam' imposes a 90/10 prior for the first specified model. The user
        may also specify his own prior. The default is 'flat'.
    leg_labs : dict of str, optional
        Labels for each chain in 'chains' to be reported on the plot legend.
        The default is the keys of 'chains'.
    subtitle : str, optional
        Additional information to put in the title of the plot, if desired.
        The default is an empty string.
    saveFig : bool, optional
        Whether to save the figure. The default is False.
    imgDir : str, optional
        Directory to save image to. The default is the current working 
        directory.
    figName : str, optional
        Name to save figure under (excluding extension). The default is 
        "margePostComp_[param]".
    showFig : bool, optional
        Whether to show figure in console. The default is True.
    context : None, dict, or one of {paper, notebook, talk, poster}, optional
        The 'context' variable to use in ``seaborn.set_theme`` to set the style
        of the plot. The default is 'notebook'.

    Returns
    -------
    None.

    '''
    
    if not imgDir:
        imgDir = os.getcwd()
    if not figName:
        figName = 'model_posterior_burninDependence'
    if leg_labs == None:
        leg_labs = {k:k for k in models} # just label with keys
    
    # Set prior
    if prior == 'flat':
        prior = {m:0.5 for m in models} # flat prior
    elif prior == 'occam':
        prior = {models[0]:0.9,models[1]:0.1} # occam prior
    else:
        pass # prior is specified by user
        
    # Specify burn-ins to iterate over
    if not burnin:
        # if None, do default values
        burnin = np.arange(0,0.5,0.05)
        
    # Get model posteriors
    posts = {m:[] for m in models}
    for b in burnin:
        data = run(roots,models,chainDir,b,prior)
        for m in models:
            posts[m].append(data['model_posts'][m])
    
    # Plot burn-in dependence of model posteriors
    sns.set_theme(context=context,style='white',palette='Set2',font_scale=1.2)

    for m in models:
        plt.plot(burnin,posts[m],'o-',label=leg_labs[m])
    plt.xlabel('Fractional Burn-in')
    plt.xticks(burnin)
    plt.ylabel('Model Posterior')
    plt.title('Burn-in Dependence of Model Posteriors'+subtitle)
    plt.legend(loc='best',frameon=False)
    if saveFig:
        plt.savefig(imgDir+figName+'.pdf',dpi=300) # dpi = dots per inch (res)
    if showFig:
        plt.show()
        
# Parameter data
def param_table(chains,params,CI=[0.68],latex=None,subtitle='',
                saveFig=False,imgDir=None,figName=None,showFig=True,
                context='notebook',dataDir=None):
    '''
    Collects the mean, standard deviation, and credible intervals for the 
    specified parameter from weighted MCMC chains, plots the distribution and 
    CIs, and saves a .csv table with the values formatted for copy/pasting
    into latex.

    Parameters
    ----------
    chains : DataFrame
        Set of MCMC chains (can be the model-weighted chains).
    params : list of str
        Parameters to analyze.
    CI : list of float, optional
        Contains fractional credible intervals to analyze. The default is [0.68].
    latex : dict of str, optional
        Latex labels for passed parameters. The default is the passed parameters.
    subtitle : str, optional
        Subtitle for plots. The default is an empty string.
    saveFig : bool, optional
        Whether to save the figures produced. The default is False.
    imgDir : str, optional
        Image directory to save figures to. The default is the current working
        directory.
    figName : str, optional
        Name to save figure and data as. The default is '_paramConstraints'.
    showFig : bool, optional
        Whether to make plots of the parameters and stats. The default is True.
    context : None, dict, or one of {paper, notebook, talk, poster}, optional
        The 'context' variable to use in ``seaborn.set_theme`` to set the style
        of the plot. The default is 'notebook'.
    dataDir : str, optional
        Directory to save data to. The default is the current working directory.

    Returns
    -------
    None.

    '''
    
    if not imgDir:
        imgDir = os.getcwd()
    if not figName:
        figName = '_paramConstraints'
    if not dataDir:
        dataDir = os.getcwd()
    if not latex:
        latex = {p:p for p in params}
    
    # Get parameter stats
    means = {}
    stds = {}
    for p in params:
        try:
            # chains are reweighted chains
            mean = np.average(chains[p],weights=chains['model_weights'])
            std = np.sqrt(np.average((chains[p] - mean)**2,
                                     weights=chains['model_weights']))
                # weighted standard deviation
        except:
            # chains are regular
            mean = np.average(chains[p],weights=chains['weight'])
            std = np.sqrt(np.average((chains[p] - mean)**2,
                                     weights=chains['weight']))
        means[p], stds[p] = mean, std
        
    # return means, stds
    
    # Get parameter credible intervals
    # NOTE: As of now, the intervals are symmetric about the 50% point, NOT
    ## symmetric about the mean.
    cis = {}
    for p in params:
        ints = {}
        for interval in CI:
            lower = (1.0-interval)/2 # lower half quantile
            upper = 1.0 - lower # upper half quantile
            try:
                # reweighted chains
                wstats = ssw.DescrStatsW(chains[p],weights=chains['model_weights'])
            except:
                # regular chains
                wstats = ssw.DescrStatsW(chains[p],weights=chains['weight'])
            quantiles = wstats.quantile([lower,upper],return_pandas=False)
                # array of p-values at lower & upper %s
            ints[interval] = quantiles # store quantiles of each interval
        cis[p] = ints # store interval quantiles for each parameter
    
    # Put into table format (see paper template)
    data = pd.DataFrame(index=params,columns=CI)
    for p in params:
        for c in CI:
            low = means[p] - cis[p][c][0]
            high = cis[p][c][1] - means[p]
            rounding = min(find_rounding(low),find_rounding(high)) + 1
            data[c][p] = f'${round(means[p],rounding)}^'+'{+'+str(
                round(high,rounding))+'}_{-'+str(round(low,rounding))+'}$'
    data.to_csv(dataDir+figName+'.csv',index=True)
    
    # Plot
    if showFig:
        sns.set_theme(context=context,style='white',palette='Set2')
        line_colors = ['grey','gold','navy','firebrick']
        
        for p in params:
            try:
                # chains are reweighted chains
                sns.kdeplot(chains,x=p,weights='model_weights',alpha=0.5,
                            fill=True,linewidth=0)
            except:
                # chains are regular
                sns.kdeplot(chains,x=p,weights='weight',alpha=0.5,
                            fill=True,linewidth=0)
            
            plt.axvline(means[p],linestyle='--',color='k',label='mean')
            for i,c in enumerate(CI):
                plt.axvline(cis[p][c][0],linestyle='-',color=line_colors[i],
                            linewidth=0.5)
                plt.axvline(cis[p][c][1],linestyle='-',color=line_colors[i],
                            linewidth=0.5,label=str(round(c*100))+'% CI')
                ylims = plt.ylim()
                plt.ylim(ylims)
                xlims = plt.xlim()
                plt.xlim(xlims)
                plt.fill_between([xlims[0],cis[p][c][0]],0,ylims[1],
                                 color=line_colors[i],alpha=0.1,edgecolor=None)
                plt.fill_between([cis[p][c][1],xlims[1]],0,ylims[1],
                                 color=line_colors[i],alpha=0.1,edgecolor=None)
            plt.legend(loc='best',frameon=False)
            plt.ylabel('')
            plt.xlabel(f'${latex[p]}$')
            plt.title(f'${latex[p]}$ Marginal Posterior'+subtitle)
            if saveFig:
                plt.savefig(imgDir+figName+'.pdf',dpi=300)
            plt.show()
        plt.close()
        
# Generate plots for paper
def paper_plots(chains,params,models,leg_labs=None,subtitle='',latex=None,
                prior_colours=None,saveFig=False,imgDir=None,figName=None,
                showFig=True,context='notebook',fontscale=1):
    '''
    This function generates the parameter marginal posterior plots for the 
    paper as a grid plot, comparing the regular posteriors from the original
    chains as well as the reweighted posteriors from any number (I think) of
    model priors.

    Parameters
    ----------
    chains : dict of dict of DataFrame
        The top-level entries are the dictionaries of chains run through 
        the ``run`` function, with keys corresponding to the model priors used. 
        These level-2 dictionaries contain the original chains and the final
        reweighted chain as DataFrames for the model prior used.
    params : list of str
        Parameters to plot.
    models : list of str
        Model identifiers for original chains.
    leg_labs : dict of str, optional
        Labels for each model AND each model prior used in 'chains'.The default 
        is the model identifiers, 'models', and the top-level keys of 'chains'.
    subtitle : str, optional
        Title of the plot. Best practice is to add the data sets included in 
        the chains used. The default is an empty string.
    latex : dict of str, optional
        Latex labels for parameters. The default is the parameter names.
    prior_colours : list of colors, optional
        Colours to use for the curves of each reweighted posterior. The default 
        is None (automatic colour selection).
    saveFig : bool, optional
        Whether to save the figure. The default is False.
    imgDir : str, optional
        Directory to save image to. The default is the current working 
        directory.
    figName : str, optional
        Name to save figure under (excluding extension). The default is 
        "margePostComp_[param]".
    showFig : bool, optional
        Whether to show figure in console. The default is True.
    context : None, dict, or one of {paper, notebook, talk, poster}, optional
        The 'context' variable to use in ``seaborn.set_theme`` to set the style
        of the plot. The default is 'notebook'.
    fontscale : float, optional
        Value to scale the font size by. The default is 1.

    Returns
    -------
    None.

    '''
    
    prior_types = list(chains.keys()) # each set of chains corresponds to a 
        # different model prior
    # print(prior_types)
    data = models + prior_types # [M1,M2,...,prior1,prior2,...]; 
        # identifiers for chains to plot
    
    if not imgDir:
        imgDir = os.getcwd()
    if not figName:
        figName = 'marginalPosteriors'
    if not latex:
        latex = {p:p for p in params}
    if leg_labs == None:
        leg_labs = {k:k for k in data} # just label with IDs
        
    # Organize data to plot
    # Only difference between sets of chains will be the weights in the 
    ## "Reweighted" chains
    newchains = {}
    for m in models:
        newchains[m] = chains[prior_types[0]][m]
    for pr in prior_types:
        newchains[pr] = chains[pr]['Reweighted']
        
    # Plot
    sns.set_theme(context=context,style='white',palette='Set2',
                  font_scale=fontscale)
    plt.clf()
    if not prior_colours:
        prior_colours = ['k','dimgrey','darkgreen']
    lnsty = ['-','dashdot'] # prior curve linestyles
    
    nrows = 2
    ncols = int(len(params)/nrows)
    if len(params) == 1:
        nrows = 1
        ncols = 1
    fig, ax = plt.subplots(nrows=nrows,ncols=ncols,
                           figsize=(6*ncols,int(4.5*nrows)),layout='tight',squeeze=False)
        # ax is an nrows x ncols array
    newparams = np.array([[p for p in params[i:i+ncols]] for i in range(0,len(params),ncols)])
        # structure parameters like axes for easy iteration
    # print(ax)
    # print(newparams)
    for j in range(nrows):
        for k in range(ncols):
            for m in models:
                sns.kdeplot(newchains[m],x=newparams[j][k],bw_method='scott',
                            weights='weight',label=leg_labs[m],fill=True,
                            alpha=0.5,linewidth=0,bw_adjust=2,ax=ax[j][k])
            for i,pr in enumerate(prior_types):
                sns.kdeplot(newchains[pr],x=newparams[j][k],weights='model_weights',
                            bw_method='scott',label=leg_labs[pr],
                            color=prior_colours[i],bw_adjust=2,ax=ax[j][k],
                            linewidth=2.5,linestyle=lnsty[i])
            ax[j][k].set_xlabel(f'${latex[newparams[j][k]]}$')
            ax[j][k].set_ylabel('')
    # ax[0][0].set_title('Parameter Marginal Posteriors',loc='right')
    #ax[0][1].set_title(subtitle)
    ax[0][0].legend(loc='best',frameon=False)
    if saveFig:
        plt.savefig(imgDir+figName+'.pdf',dpi=300)
    if showFig:
        plt.show()
    plt.close()
        
def GRstat_Harmonic_check(roots,models,chainDir,burnin,temperature,prior,Rm1_stop=0.05, out_var=False, nmax='all'):
    '''
    Computes the Gelman-Rubin R-1 statistic for the model posterior. It uses the 
    harmonic mean method to compute multiple model posteriors from a single chain
    and produce the mean and variance of the model posteriors for each chain.
    It indicates whether the chains are considered "converged" according to the
    specified R-1 threshhold for convergence and returns the R-1 statistics 
    for each model posterior.

    Parameters
    ----------
    roots : list of str
        List of chain root names to analyse.
    models : list of str
        Model identifiers corresponding to each set of chains specified in 
        'roots'.
    chainDir : str
        Directory to fetch chains from.
    burnin : list of float or int
        List of fractional burn-in OR number of burn-in samples to remove from 
        chains.
    temperature: dict of float
        Chain temperature - used to retrieve the correct -logP
    prior : dict of float
        Model priors for each model in 'models'.
    Rm1_stop : float, optional
        Threshhold R-1 value for the chains to be considered converged. The 
        default is 0.05.
    out_var: bool
        Wether to output or not W and B as well
    nmax: 'all' or 0<=nmax<=1
        the chains' length fraction used to compute all quantities 
        (after burn-in has been removed)

    Returns
    -------
    str
        Whether the chains are converged: "yes" or "no".
    Rm1s : dict of float
        R-1 statistics for each model posterior.

    '''
    
    # 1. Remove burn-in
    # Load chains
    print('Loading chains')
    chains = dfChains(roots,models,chainDir,burnin,stacked=False, nmax=nmax)
    for m in models:
        for i in range(len(chains[m])):
            chains[m][i]['minuslogpost'] *= temperature[m]
    
    # a) choose L as mean of all lengths because it's negligible anyways
    length = []
    for m in models:
        length = length + [len(chains[m][i]) for i in range(len(chains[m]))]
    L = int(np.mean(length)) # average number of samples in each chain
    print('\tL =',L)
    print('\tmin(length) =',np.amin(length))
    print('\tmax(length) =',np.amax(length))
    
    # 2. Calculate...
    
    print('Calculating model posterior')
    
    # # (i) Compute sample likelihoods
    likelihoods = get_lik(chains,stacked=False)
    
    # (ii) Compute model posteriors for each pair of chains
    nch = np.amin([len(chains[m]) for m in models]) # minimum number of chains
    print('Number of chains:', nch)

    model_posts = {m:[] for m in models}
    
    for i in range(nch):
        # for each chain
        temp_likelihoods = {m:deepcopy(likelihoods[m][i]) 
                            for m in models}
        ch_model_liks = model_lik(temp_likelihoods) # stacked = True effectively
        ch_model_posts = model_post(models,ch_model_liks,prior)
        for m in models:
            model_posts[m].append(ch_model_posts[m])

            

    # b) grand mean = average model posterior
    print('Calculating average model posterior')
    avg_model_posts = {m:np.mean(model_posts[m]) for m in models}
    print('between chains average model like: ', avg_model_posts)

    # c) between-chain variance
    print('Calculating between-chain variance')
    J = nch # number of chains
    # L = int(1/removal) # number of blocks used in jackknife
    Bs = {}
    for m in models:
        insum = []
        for j in range(J):
            insum.append((model_posts[m][j] - avg_model_posts[m])**2)
        insum = np.array(insum)
        tmp = L/(J-1) * np.sum(insum)
        Bs[m] = tmp
    B = Bs['LCDM']
    print('\tB =',B/L)
        
    # d) within-chain variance
    print('Calculating within-chain variance')
    # print('\tnch =',nch)
    # print('\tlen(chains) =',[len(chains[m]) for m in models])
    wc_vars = {m:[] for m in models}
    for i in range(nch):
        # for each chain, compute the variance
        var_chains = {m:deepcopy(chains[m][i]) for m in models}
        wcm, wcv = harmonic_stat(var_chains,prior)
        #print(np.mean((1./wcm['LCDM'])/(1./wcm['LCDM']+1./wcm['EDE'])))
        for m in models:
            wc_vars[m].append(wcv[m])
    #print(wc_vars['LCDM'])
    

    # e) W = mean of within-chain variances
    print('Calculating W')
    Ws = {}
    for m in models:
        Ws[m] = np.mean(wc_vars[m])
        #print('\tW =',np.mean(wc_vars[m])/(L-1))
    W = Ws['LCDM']
    print('W=',W)

    # 3. Gelman-Rubin statistic
    # NOTE: R should be the same for each model in a 2-model comparison
    Rs = {}
    # L = int(1/removal) # number of points used to compute WCV
    # print('\taverage L =',L)
    # print('\tstd L =',np.std([np.sum(ch['weight']) for ch in newchains]))
        
    # 4. R-1 convergence test
    R = (W + B/L)
    R = np.sqrt(R/W*L/(L-1))
    for m in models:
        Rs[m] = R
    print('R-1=',Rs['LCDM']-1)
        
    # 4. R-1 convergence test
    Rm1s = {m:Rs[m]-1 for m in models} # R-1
    for rm1 in Rm1s.values():
        if rm1 > Rm1_stop:
            # converged
            if out_var:
                return 'no', Rm1s, W, B/L
            else:
                return 'no', Rm1s
    if out_var:
        return 'yes', Rm1s, W, B/L
    else:
        return 'yes', Rm1s

def harmonic_stat(chains,prior):
    '''
    Uses the harmonic mean method to compute the mean and variance of the model
    posteriors from a single chain.

    Parameters
    ----------
    chains : dict of DataFrame
        Each entry is a single chain from the model specified as its key
    prior : dict of float
        Model priors for each model in 'models'.

    Returns
    -------
    means : dict of float
        Each entry is the mean of the model posteriors from the chain of the 
        same key.
    varis : dict of float
        Each entry is the variance of the model posteriors from the chain of
        the same key.

    '''
    
    models = list(chains.keys()) # chains directly loaded from dfChains
    mu_R  = {m:[] for m in models}
    Var_R = {m:[] for m in models}
    N = {m:[] for m in models}

    likelihoods = get_lik(chains)
    model_liks = model_lik(likelihoods)
    model_posts = model_post(models,model_liks,prior)
    for m in models:
        N = len(chains[m])
        mu_R[m]  = 1./model_liks[m]
        Var_R[m] = 1./N * np.sum((1./likelihoods[m][:]-mu_R[m])**2)
        Var_R[m] = Var_R[m]/(N*mu_R[m]**4)

    var = ( Var_R[models[0]]*(model_liks[models[1]]*prior[models[1]])**2 + 
           Var_R[models[1]]*(model_liks[models[0]]*prior[models[0]])**2 ) 
    var = var / ((model_liks[models[1]]*prior[models[1]])**4)
    for m in models:
        Var_R[m] = var
            
    # Evaluate within-chain mean & variance of model posteriors
    
    return mu_R, Var_R
# Helper Functions ------------------------------------------------------------

# Turn data into latex table
def latex_table(dataName,rounding=False,caption='',label='',dataDir=None):
    '''
    This function takes a data set in .csv format and saves it in a text file 
    in a latex table format.

    Parameters
    ----------
    dataName : str
        Name .csv data is saved under (excluding extension).
    rounding : False or int, optional
        If not False, specifies the number of digits to round the table entries
        to. The default is False.
    caption : str, optional
        Caption under the table. The default is ''.
    label : str, optional
        Identifier of table in the latex file. The default is ''.
    dataDir : str, optional
        Directory data is saved in. The default is the current working 
        directory.

    Returns
    -------
    None.

    '''
    
    if not dataDir:
        dataDir = os.getcwd()
    
    # Load data
    data = pd.read_csv(dataDir+dataName+'.csv')
    
    # Set up top matter
    f = open(dataDir+dataName+'_latexTable.txt','w')
    f.write('\\begin{table}')
    f.write('\n\caption{'+caption+'}') # title of table
    f.write('\n\label{tab:'+label+'}') # used to refer to table in text
    f.write('\n\centering') # centering table
    cols = data.columns
    f.write('\n\\begin{tabular}{') # centered columns, no line separation
    for c in cols:
        f.write('c')
    f.write('}')
    
    # Write column headers
    f.write('\n\hline\hline\n') # double horizontal line
    for i,c in enumerate(cols):
        f.write(c+' & ')
        if i == len(cols)-2:
            break
    f.write(cols[-1]+'\\\\\n\hline\hline') # double horizontal line
    
    # Write data
    for i in data.index:
        f.write('\n\t\t\t')
        for j,c in enumerate(cols):
            if not rounding:
                f.write(str(data[c][i])+' & ')
            else:
                f.write(str(round(data[c][i],rounding))+' & ')
            if j == len(cols)-2:
                break
        if not rounding:
            f.write(str(data[cols[-1]][i]))
        else:
            f.write(str(round(data[cols[-1]][i],rounding)))
        f.write('\\\\')
    
    # Bottom matter
    f.write('\n\hline') # bottom line
    f.write('\n\end{tabular}')
    f.write('\n\end{table}')
    
    f.close()

# Function that finds significance of a number
def find_rounding(num):
    '''
    Finds the significance of a number and returns the location of that 
    digit.

    Parameters
    ----------
    num : float or int
        Number to find significance of.

    Returns
    -------
    float
        Location of significance.

    '''
    
    try:
        decimal = str(num).split('.')[-1] # get decimal part of number
    except:
        return 0 # it's a whole number, don't round it
    for i,digit in enumerate(decimal):
        if digit != '0':
            return i+1
