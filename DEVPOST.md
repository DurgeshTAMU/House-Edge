## Inspiration
House Edge scores our probabilities against the simulator's hidden true odds, so luck can't decide the result. We asked what the simulator is doing underneath, instead of adding a bigger black box.

## What we built
For each machine we predict $p=(p_0,p_1,p_2,p_5,p_{20})$ from history before the cutoff. We replaced boosted models (~39/60) with exact Bayesian inference over the simulator's rules: which lagged payouts matter, their direction, and when flips happen. Flip evidence is pooled across the episode, the payout table is inferred, and a small logit correction handles residuals. Cross-validated score: about **58.9/60**.

## Learned
Finding the generating structure beat adding capacity. We kept whole episodes together and used leave-one-episode-out scoring to stay honest.

## Challenges
Hidden state, sparse short histories, missing ticks (unknown, not zero), and packaging for an offline stdlib-only runner. Round two refits the rules automatically, with a `family_check` to flag a changed rule family.

## Built with
`python` `numpy` `bayesian-inference` `probabilistic-forecasting` `time-series` `machine-learning` `cross-validation` `reproducible-research` `cpu-only` `github`
