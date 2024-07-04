# Fast-MPC

Fast-MPC is a computational strategy for Bayesian Model Averaging (BMA) that exploits existing MCMC software and combines model-specific posteriors post-hoc.

It is currently only a collection of useful functions, but the long-term plan is to turn it into a proper python package.

It currently implements two different estimators: the standard harmonic mean and the learnt harmonic mean (from https://arxiv.org/abs/2111.12720). You can use either one or the other to evaluate the model marginal posterior distribution.


--------------------------------
If you use this code please cite the following papers:

* For standard harmonic mean estimator case:

        @article{Paradiso:2023,
            author = {Paradiso, S and DiMarco, M and Chen, M and McGee, G and Percival, W J},
            title = "{A convenient approach to characterizing model uncertainty with application to early dark energy solutions of the Hubble tension}",
            journal = {Monthly Notices of the Royal Astronomical Society},
            volume = {528},
            number = {2},
            pages = {1531-1540},
            year = {2024},
            month = {01},
            issn = {0035-8711},
            doi = {10.1093/mnras/stae101},
            url = {https://doi.org/10.1093/mnras/stae101},
            eprint = {https://academic.oup.com/mnras/article-pdf/528/2/1531/56410678/stae101.pdf},
        }

* and the submitted paper: https://arxiv.org/abs/2403.02120


* For the learnt harmonic mean estimator include also:

        @article{harmonic,
           author = {Jason~D.~McEwen and Christopher~G.~R.~Wallis and Matthew~A.~Price and Matthew~M.~Docherty},
            title = {Machine learning assisted {B}ayesian model comparison: learnt harmonic mean estimator},
          journal = {ArXiv},
           eprint = {arXiv:2111.12720},
             year = 2021
        }

--------------------------------------------