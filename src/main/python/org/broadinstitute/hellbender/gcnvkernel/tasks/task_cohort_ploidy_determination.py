import logging
import numpy as np
import pymc3 as pm
from typing import Callable
import matplotlib.pyplot as plt
from scipy.stats import nbinom

from .inference_task_base import Sampler, Caller, CallerUpdateSummary, \
    HybridInferenceTask, HybridInferenceParameters
from .. import config, types
from ..models.model_ploidy import PloidyModelConfig, PloidyModel, \
    PloidyWorkspace, PloidyEmissionBasicSampler, PloidyBasicCaller

_logger = logging.getLogger(__name__)


class PloidyCaller(Caller):
    """This class is a wrapper around `PloidyBasicCaller` to be used in a `HybridInferenceTask`."""
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 ploidy_workspace: PloidyWorkspace):
        self.hybrid_inference_params = hybrid_inference_params
        self.ploidy_basic_caller = PloidyBasicCaller(hybrid_inference_params, ploidy_workspace)
        self.ploidy_workspace = ploidy_workspace

    def snapshot(self):
        """Snapshot is not necessary since there is no internal consistency loop."""
        pass

    def finalize(self):
        """Finalizing is not necessary since there is no internal consistency loop."""
        pass

    def call(self) -> 'PloidyCallerUpdateSummary':
        update_norm_sj = self.ploidy_basic_caller.call()
        return PloidyCallerUpdateSummary(
            update_norm_sj, self.hybrid_inference_params.caller_summary_statistics_reducer)

    def update_auxiliary_vars(self):
        pass


class PloidyCallerUpdateSummary(CallerUpdateSummary):
    def __init__(self,
                 update_norm_sj: np.ndarray,
                 reducer: Callable[[np.ndarray], float]):
        self.scalar_update = reducer(update_norm_sj)

    def __repr__(self):
        return "ploidy update size: {0:2.6}".format(self.scalar_update)

    def reduce_to_scalar(self) -> float:
        return self.scalar_update


class PloidyEmissionSampler(Sampler):
    """This class is a wrapper around `PloidyEmissionBasicSampler` to be used in a `HybridInferenceTask`."""
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 ploidy_model: PloidyModel,
                 ploidy_workspace: PloidyWorkspace):
        super().__init__(hybrid_inference_params)
        self.ploidy_workspace = ploidy_workspace
        self.ploidy_emission_basic_sampler = PloidyEmissionBasicSampler(
            ploidy_model, self.hybrid_inference_params.log_emission_samples_per_round)

    def update_approximation(self, approx: pm.approximations.MeanField):
        self.ploidy_emission_basic_sampler.update_approximation(approx)

    def draw(self) -> np.ndarray:
        return self.ploidy_emission_basic_sampler.draw()

    def reset(self):
        self.ploidy_workspace.log_ploidy_emission_sjl.set_value(
            np.zeros((self.ploidy_workspace.num_samples,
                      self.ploidy_workspace.num_contigs,
                      self.ploidy_workspace.num_ploidies),
                     dtype=types.floatX), borrow=config.borrow_numpy)

    def increment(self, update):
        self.ploidy_workspace.log_ploidy_emission_sjl.set_value(
            self.ploidy_workspace.log_ploidy_emission_sjl.get_value(borrow=True) + update)

    def get_latest_log_emission_posterior_mean_estimator(self) -> np.ndarray:
        return self.ploidy_workspace.log_ploidy_emission_sjl.get_value(borrow=True)


class CohortPloidyInferenceTask(HybridInferenceTask):
    """Cohort germline contig ploidy determination task."""
    def __init__(self,
                 hybrid_inference_params: HybridInferenceParameters,
                 ploidy_config: PloidyModelConfig,
                 ploidy_workspace: PloidyWorkspace):
        _logger.info("Instantiating the germline contig ploidy determination model...")
        self.ploidy_model = PloidyModel(ploidy_config, ploidy_workspace)

        _logger.info("Instantiating the ploidy emission sampler...")
        ploidy_emission_sampler = PloidyEmissionSampler(hybrid_inference_params, self.ploidy_model, ploidy_workspace)

        _logger.info("Instantiating the ploidy caller...")
        ploidy_caller = PloidyCaller(hybrid_inference_params, ploidy_workspace)

        elbo_normalization_factor = ploidy_workspace.num_samples * ploidy_workspace.num_contigs
        super().__init__(hybrid_inference_params, self.ploidy_model, ploidy_emission_sampler, ploidy_caller,
                         elbo_normalization_factor=elbo_normalization_factor,
                         advi_task_name="denoising",
                         calling_task_name="ploidy calling")

        self.ploidy_config = ploidy_config
        self.ploidy_workspace = ploidy_workspace

    def disengage(self):
        num_samples = 1000
        trace = self.continuous_model_approx.sample(num_samples)
        pi_i_sk = [np.mean(trace['pi_%d_sk' % i], axis=0)
                   for i in range(self.ploidy_workspace.num_contig_tuples)]
        d_s = np.mean(trace['d_s'], axis=0)
        b_j_norm = np.mean(trace['b_j_norm'], axis=0)
        error_rate_js = np.mean(trace['error_rate_js'], axis=0)
        mu_j_sk = [np.mean(trace['mu_%d_sk' % j], axis=0)
                   for j in range(self.ploidy_workspace.num_contigs)]

        fit_mu_sj = self.ploidy_workspace.fit_mu_sj
        fit_mu_sd_sj = self.ploidy_workspace.fit_mu_sd_sj
        fit_alpha_sj = self.ploidy_workspace.fit_alpha_sj

        print("pi_i_sk")
        print(pi_i_sk)
        print("d_s")
        print(d_s)
        print("b_j_norm")
        print(b_j_norm)
        q_ploidy_sjl = np.exp(self.ploidy_workspace.log_q_ploidy_sjl.get_value(borrow=True))
        for s, q_ploidy_jl in enumerate(q_ploidy_sjl):
            print('sample_{0}:'.format(s), np.argmax(q_ploidy_jl, axis=1))
        for s in range(self.ploidy_workspace.num_samples):
            l_j = np.argmax(q_ploidy_sjl[s], axis=1)
            fig, axarr = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw = {'height_ratios':[3, 1]})
            for i, contig_tuple in enumerate(self.ploidy_workspace.contig_tuples):
                for contig in contig_tuple:
                    j = self.ploidy_workspace.contig_to_index_map[contig]
                    hist_mask_m = np.logical_not(self.ploidy_workspace.hist_mask_sjm[s, j])
                    counts_m = self.ploidy_workspace.counts_m
                    hist_norm_m = self.ploidy_workspace.hist_sjm_full[s, j] / np.sum(self.ploidy_workspace.hist_sjm_full[s, j] * self.ploidy_workspace.hist_mask_sjm[s, j])
                    axarr[0].semilogy(counts_m, hist_norm_m, c='k', alpha=0.25)
                    axarr[0].semilogy(counts_m, np.ma.array(hist_norm_m, mask=hist_mask_m), c='b', alpha=0.5)
                    mu = fit_mu_sj[s, j]
                    alpha = fit_alpha_sj[s, j]
                    pdf_m = nbinom.pmf(k=counts_m, n=alpha, p=alpha / (mu + alpha))
                    axarr[0].semilogy(counts_m, np.ma.array(pdf_m, mask=hist_mask_m), c='g', lw=2)
                    axarr[0].set_xlim([0, self.ploidy_workspace.num_counts])
            axarr[0].set_ylim([1 / np.max(np.sum(self.ploidy_workspace.hist_sjm_full[s] * self.ploidy_workspace.hist_mask_sjm[s], axis=-1)), 1E-1])
            axarr[0].set_xlabel('count', size=14)
            axarr[0].set_ylabel('density', size=14)

            k_j = [np.argmax(pi_i_sk[i][s])
                   for i, contig_tuple in enumerate(self.ploidy_workspace.contig_tuples)
                   for j in range(len(contig_tuple))]
            mu_j = np.array([mu_j_sk[j][s, k_j[j]] for j in range(self.ploidy_workspace.num_contigs)])

            j = np.arange(self.ploidy_workspace.num_contigs)

            axarr[1].errorbar(j, fit_mu_sj[s] / (d_s[s] * b_j_norm),
                              yerr=fit_mu_sd_sj[s] / (d_s[s] * b_j_norm),
                              c='g', fmt='o', elinewidth=2, alpha=0.5)
            axarr[1].scatter(j, l_j, c='r')
            axarr[1].set_xticks(j)
            axarr[1].set_xticklabels(self.ploidy_workspace.contigs)
            axarr[1].set_xlabel('contig', size=14)
            axarr[1].set_ylabel('ploidy', size=14)
            axarr[1].set_ylim([0, np.shape(q_ploidy_sjl)[2]])

            fig.tight_layout(pad=0.5)
            fig.savefig('/home/slee/working/gatk/test_files/plots/sample_{0}.png'.format(s))
