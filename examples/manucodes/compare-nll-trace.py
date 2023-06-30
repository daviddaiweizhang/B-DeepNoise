import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp


def eff_sample_size(x):
    n = tfp.mcmc.effective_sample_size(x).numpy()
    n = np.nanmean(n)
    return n


def main():
    n_steps = 500
    outfile = 'nll-trace.png'
    bnn_hmc_file = 'nll-bnn-hmc.pickle'
    dalea_hmc_file = 'nll-dalea-hmc.pickle'
    dalea_gibbs_file = 'nll-dalea-gibbs.pickle'

    nll_bnn_hmc = pickle.load(open(bnn_hmc_file, 'rb'))['nll']
    nll_dalea_hmc = pickle.load(open(dalea_hmc_file, 'rb'))
    nll_dalea_gibbs = pickle.load(open(dalea_gibbs_file, 'rb'))
    n_states = nll_dalea_gibbs.shape[0]
    steps = np.round(np.linspace(0, 1, n_states) * (n_steps-1)).astype(int)

    n_states_last = int(n_states * 0.4)
    factor = n_steps // n_states
    n_steps_last = n_states_last * factor
    ess_bnn_hmc = eff_sample_size(nll_bnn_hmc[-n_states_last:])
    ess_bnn_hmc = int(ess_bnn_hmc * factor)
    ess_dalea_hmc = eff_sample_size(nll_dalea_hmc[-n_states_last:])
    ess_dalea_hmc = int(ess_dalea_hmc * factor)
    ess_dalea_gibbs = eff_sample_size(nll_dalea_gibbs[-n_states_last:])
    ess_dalea_gibbs = int(ess_dalea_gibbs * factor)

    for i, y in enumerate(nll_bnn_hmc.T):
        lab = None
        if i == 0:
            lab = 'BNN + HMC'
            lab = f'{lab} (ESS: {ess_bnn_hmc} / {n_steps_last})'
        plt.plot(steps, y, color='tab:blue', label=lab)
    for i, y in enumerate(nll_dalea_hmc.T):
        lab = None
        if i == 0:
            lab = 'B-DeepNoise + HMC'
            lab = f'{lab} (ESS: {ess_dalea_hmc} / {n_steps_last})'
        plt.plot(steps, y, color='tab:green', label=lab)
    for i, y in enumerate(nll_dalea_gibbs.T):
        lab = None
        if i == 0:
            lab = 'B-DeepNoise + Gibbs'
            lab = f'{lab} (ESS: {ess_dalea_gibbs} / {n_steps_last})'
        plt.plot(steps, y, color='tab:orange', label=lab)
    plt.ylim(-0.5, 1.5)
    plt.ylabel('Testing negative log-likelihood')
    plt.xlabel('Number of iterations')
    plt.title('Posterior sampling convergence comparison')
    plt.legend()
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(outfile)


if __name__ == '__main__':
    main()
