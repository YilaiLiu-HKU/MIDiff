"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import torch.nn.functional as F
import numpy as np
import torch as th

from nn import mean_flat
from losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        discriminator=None,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.step=0
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        #
        model_output,max_prediction= model(x, self._scale_timesteps(t), **model_kwargs)
        
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'max_prediction':max_prediction
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"],"max_prediction":out["max_prediction"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
       
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"],final["max_prediction"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        #import pdb;pdb.set_trace()
      
        if not hasattr(self, "call_count"):
            self.call_count = 0
        self.call_count += 1
        if model_kwargs is None:
            model_kwargs = {}
        mask = model_kwargs.pop("padding_mask", None)
        using_MAE = model_kwargs.pop("using_MAE", False)
        special_weight = model_kwargs.pop('special_weight', 1)
        special_value = model_kwargs.pop('special_value', None)
        cos_weight=model_kwargs.pop('cos_weight', 0)
        max_value=model_kwargs.pop('max_value', None)
        traffic_target=model_kwargs.pop('traffic',None)
        post_traffic= model_kwargs.pop('post_traffic',False)
        use_heatmap=model_kwargs.pop('use_heatmap',False)
        norm_img=model_kwargs.pop('norm_img',None)
        use_FFT=model_kwargs.pop('use_FFT',False)
        discriminator=model_kwargs.pop('discriminator',None)
        tile_mask=model_kwargs.pop('tile_mask',None)
        use_pixel_refiner=model_kwargs.pop('use_pixel_refiner',False)
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:

            model_output,traffic_prediction = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C, H, W = x_t.shape[:4]
                
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: [r,None],
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            ########这里加入padding mask忽略梯度计算
      
            if not using_MAE:
                diff = (target - model_output) ** 2
            else:
                diff = th.abs(target - model_output)



            
            ####加入余弦损失
            def fft_mse_loss_with_mask(x, y, mask):
                """
                计算仅在mask区域内的频域MSE损失。

                Args:
                    x (Tensor): 输入图像 [B, C, H, W]
                    y (Tensor): 目标图像 [B, C, H, W]
                    mask (Tensor): 掩码 [B, 1, H, W]，其中值为1的地方是感兴趣的区域

                Returns:
                    Tensor: 频域MSE损失
                """
                # 仅保留mask区域的输入和目标
                x_masked = x * mask.unsqueeze(1)  # 将x中mask外的区域置为0
                y_masked = y * mask.unsqueeze(1)  # 将y中mask外的区域置为0

                # 计算FFT
                x_fft = th.fft.fft2(x_masked)  # [B, C, H, W]
                y_fft = th.fft.fft2(y_masked)  # [B, C, H, W]

                # 计算幅度差的平方（MSE）
                x_fft_abs = th.abs(x_fft)
                y_fft_abs = th.abs(y_fft)
                x_fft_abs=x_fft_abs
                y_fft_abs=y_fft_abs
                fft_diff=(x_fft_abs - y_fft_abs) ** 2
                if mask is not None:
                    fft_diff=fft_diff*mask  # apply mask
                    fft_mse = fft_diff.sum() / (mask.sum() * fft_diff.shape[1] + 1e-8)
                # 计算频域MSE
            

                return fft_mse
            def cosine_loss(x, y, reduction='mean'):
                """
                计算余弦损失（1 - cos similarity）

                Args:
                    x (Tensor): shape (batch_size, dim)
                    y (Tensor): shape (batch_size, dim)
                    reduction (str): 'mean', 'sum', or 'none'

                Returns:
                    Tensor: scalar loss or per-sample loss
                """
                x_norm = F.normalize(x, p=2, dim=1)
                y_norm = F.normalize(y, p=2, dim=1)
                cosine_sim = (x_norm * y_norm).sum(dim=1)
                loss = 1 - cosine_sim  # 越小越好
                if reduction == 'mean':
                    return loss.mean()
                elif reduction == 'sum':
                    return loss.sum()
                else:
                    return loss
            #terms['cos']=cosine_loss(model_output,target)
            np.set_printoptions(threshold=np.inf) 
            
            if mask is not None:
                mask = mask.to(diff.device)
                #import pdb;pdb.set_trace()
                # 统一成 (B,1,...) 以便按通道广播
                if mask.dim() == diff.dim() - 1:
                    mask = mask.unsqueeze(1)
                # 现在 mask: (B,1,H,W) 与 diff: (B,C,H,W) 可广播

                if special_weight != 1:
                    gt = target
                    # 这里请确认语义：等于还是不等于？
                    special_mask = (gt != special_value) & (mask > 0.5)  # (B,C,H,W) via broadcast
                    weights = th.ones_like(diff)
                    weights[special_mask] = special_weight
                    masked = diff * mask * weights
                else:
                    masked = diff * mask

                # 每个样本的 masked 元素个数 = (mask 扩展到通道以后).sum
                den = mask.expand_as(diff).sum(dim=(1, 2, 3)).clamp_min(1e-8)  # (B,)
                mse = masked.flatten(1).sum(dim=1) / den                        # (B,)
            else:
                # 约定：mean_flat 返回 (B,)
                mse = mean_flat(diff)
            terms["loss"]=0
            terms["mse"] = mse
            def binary_weighted_loss(
                prediction,
                target,
                tile_mask,
                alpha: float = 5.0,
                beta: float = 1.0,
                reduction: str = "none",   # "none" | "mean" | "sum"
            ):
                """
                分样本的二元加权L1损失（MAE）。对信号(tail)与背景(head)像素分别求均值后加权。
                返回值在 batch 维度上不做归约：形状为 (B,)。

                Args:
                    prediction: (B, C, ...) 或 (B, ...)
                    target:     与 prediction 同形状
                    tile_mask:  (B, 1, ...) 或 (B, ...) 或与 prediction 可广播
                                1 表示信号像素(tail)，0 表示背景(head)
                    alpha:      tail 权重
                    beta:       head 权重
                    reduction:  "none" 返回 (B,)
                                "mean"/"sum" 会在 batch 维上再做一次归约
                Returns:
                    tuple:
                    total_loss: (B,) 或标量（取决于 reduction）
                    tail_loss:  (B,) 或标量
                    head_loss:  (B,) 或标量
                """
                # 确保在同一设备与dtype
                device = prediction.device
                dtype = prediction.dtype
                target = target.to(device=device, dtype=dtype)
                tile_mask = tile_mask.to(device=device, dtype=dtype)

                # 将 mask 扩展到与 prediction 可广播的形状（常见的是补上通道维）
                # 例：prediction: (B, C, H, W), mask: (B, H, W) -> (B, 1, H, W)
                while tile_mask.dim() < prediction.dim():
                    tile_mask = tile_mask.unsqueeze(1)

                diff = (prediction - target)**2
                head_mask = 1.0 - tile_mask

                # 在除 batch 以外的所有维度上做求和/计数
                reduce_dims = list(range(1, diff.ndim))

                tail_sum = (diff * tile_mask).sum(dim=reduce_dims)
                head_sum = (diff * head_mask).sum(dim=reduce_dims)

                # 每个样本内的像素数（避免0除）
                eps = 1e-8
                num_tail = tile_mask.sum(dim=reduce_dims).clamp_min(eps)
                num_head = head_mask.sum(dim=reduce_dims).clamp_min(eps)

                # 每个样本内的均值损失 -> 形状 (B,)
                tail_loss = tail_sum / num_tail
                head_loss = head_sum / num_head

                total_loss = alpha * tail_loss + beta * head_loss

                if reduction == "mean":
                    return total_loss.mean(), tail_loss.mean(), head_loss.mean()
                elif reduction == "sum":
                    return total_loss.sum(), tail_loss.sum(), head_loss.sum()
                else:
                    # "none"：保留批次维度 (B,)
                    return total_loss, tail_loss, head_loss
        
            if tile_mask is not None:
                
                total_loss, tail_loss, head_loss = binary_weighted_loss(model_output, target, tile_mask, alpha=5.0, beta=1.0)
                terms['loss']+=total_loss
                terms['tile_tail_loss']=tail_loss
                terms['tile_head_loss']=head_loss
            else:
                terms["loss"] += terms["mse"]
                #terms['loss']=mse+0.1*mse_tile
                #terms['loss']=mse
                #import pdb;pdb.set_trace()
                #print(f"tile_mse:{mse_tile.item()}")

            #if "vb" in terms:
                #terms["loss"] =terms["mse"]+cos_weight*terms['cos'] + terms["vb"]
                
                #terms["loss"]+=terms["vb"]
            
            if discriminator is not None:
                mse_per_sample = diff.view(B, -1).mean(dim=1)  # (B,) 每个样本的MSE

                # 将 mse 限制在 [0, 1] 范围内，得到每个样本的 p 值
                p = th.clamp(mse_per_sample.detach(), min=0.0, max=1.0)  # 每个样本独立计算p

                # 为每一行生成一个概率，基于每个样本的 p 值
                rand_prob = th.rand(B, H, device=norm_img.device)  # 生成一个 [B, H] 的随机数（0到1之间）
                mask_replace = rand_prob.unsqueeze(1).expand(-1, C, -1)  # 扩展为 (B, C, H)

                # 扩展 p 使其形状为 (B, C, H) 以便进行比较
                p_expanded = p.view(B, 1, 1).expand(B, C, H)  # 将 p 扩展为 (B, C, H)

                # 用 target 的该行替换 norm_img 中满足条件的行
                norm_img = th.where(mask_replace.unsqueeze(-1) < p_expanded.unsqueeze(-1), target, norm_img)

                # GAN损失计算
                real_labels = th.ones(B, C, device=model_output.device)
                fake_labels = th.zeros(B, C, device=model_output.device)

                real_loss = F.binary_cross_entropy(discriminator(model_output), real_labels)
                fake_loss = F.binary_cross_entropy(discriminator(norm_img), fake_labels)
                gan_loss = (real_loss + fake_loss) / 2.0

                terms["gan_loss"] = gan_loss
                terms["loss"] += 0.1*gan_loss # 根据时间步调整 GAN 损失的权重

            """if post_traffic:

                traffic_diff=(traffic_target - traffic_prediction) ** 2
                if mask is not None:
                    traffic_mask=th.all(traffic_target!=0,dim=1).float()
                    traffic_diff = traffic_diff * traffic_mask.unsqueeze(1)  # apply mask
                    traffic_mse = traffic_diff.sum() / (traffic_mask.sum() * traffic_diff.shape[1] + 1e-8)  # 每个通道都masked
                    terms["traffic_mse"]=traffic_mse
                    terms["loss"] += 0.1*traffic_mse"""
            """if norm_img is not None:
                
                norm_diff=(norm_img - model_output) ** 2
                if mask is not None:
                    norm_diff=norm_diff * mask.unsqueeze(1)  # apply mask
                    norm_mse = norm_diff.sum() / (mask.sum() * norm_diff.shape[1] + 1e-8)  # 每个通道都masked
                    terms["norm_mse"]=0.02*norm_mse
                    terms["loss"] += norm_mse"""
            if norm_img is not None:
                #import pdb;pdb.set_trace()
                """norm_res=(target - norm_img).abs().mean(1, keepdim=True)
                norm_output=(model_output - norm_img).abs().mean(1, keepdim=True)
                norm_diff=(norm_res - norm_output) ** 2"""
                """norm_diff=(norm_img - model_output) ** 2
                if mask is not None:
                    norm_masked=norm_diff * mask  # apply mask
                    den = mask.expand_as(norm_diff).sum(dim=(1, 2, 3)).clamp_min(1e-8)  # (B,)
                    norm_mse = norm_masked.flatten(1).sum(dim=1) / den           # 每个通道都masked
                    
                    terms["norm_mse"]=norm_mse
                    terms["loss"] += 0.005*norm_mse"""
                if self.model_mean_type == ModelMeanType.PREVIOUS_X:
                    pred_xstart = self._predict_xstart_from_xprev(x_t=x_t, t=t, xprev=model_output)
                elif self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart = model_output
                else: # ModelMeanType.EPSILON
                    pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)

                # 步骤 2: 计算您的批内方差正则化损失 (norm_mse)
                pred_xstart_clamped = pred_xstart.clamp(-1, 1)
                avg_img = pred_xstart_clamped.mean(dim=0, keepdim=True)
                norm_diff = (avg_img - pred_xstart_clamped) ** 2

                if mask is not None:
                    norm_masked = norm_diff * mask
                    den = mask.expand_as(norm_masked).sum(dim=(1, 2, 3)).clamp_min(1e-8)
                    norm_mse = norm_masked.flatten(1).sum(dim=1) / den # 得到形状为 [B,] 的 per-sample loss
                else:
                    norm_mse = mean_flat(norm_diff) # 得到形状为 [B,] 的 per-sample loss

                # ------------------- 优化部分开始 -------------------
                # 步骤 3: 定义一个随 t 变化的权重函数 w(t)
                # t 是一个形状为 [B,] 的张量，包含了批次中每个样本的时间步
                # 我们希望 t 越大，权重越小。一个简单的线性衰减函数：
                # self.num_timesteps 是总的时间步数，例如 1000
                #weights = 1.0 - (t.float() / self.num_timesteps) 权重从接近1 (t=0) 线性衰减到接近0 (t=999)

                # 为了防止在t=0时过度惩罚，可以再加一个系数或者使用其他函数，例如高斯函数：
                # mid_point = self.num_timesteps / 2
                # sigma = self.num_timesteps / 4
                # weights = torch.exp(-((t.float() - mid_point) ** 2) / (2 * sigma ** 2))

                # 步骤 4: 将权重应用到损失上
                # weights 的形状是 [B,]，norm_mse 的形状也是 [B,]
                # 我们先逐元素相乘，然后再求整个批次的平均值
                total_timesteps = self.num_timesteps
                weights = th.cos((math.pi / 2) * t.float() / total_timesteps)
                weights = weights/weights.sum()
                print(weights)
                # 步骤 4: 将权重应用到损失上，并计算整个批次的均值
                # weights 和 norm_mse 都是 [B,] 形状，逐元素相乘后求均值
                weighted_norm_mse = (weights * norm_mse).sum()
                # -------------------- 优化部分结束 --------------------


                # 步骤 5: 将加权后的正则化损失添加到总损失中
                reg_strength = 0.01 # 这是一个可以调整的超参数，代表正则化的基础强度
                terms["diversity_loss"] = weighted_norm_mse
                terms["loss"] += reg_strength * weighted_norm_mse
            if use_FFT:
                fft_mse_loss_value = fft_mse_loss_with_mask(model_output, target, mask)
                # 将FFT MSE损失加入到总损失
                
                terms["fft_mse_loss"] = fft_mse_loss_value
                terms["loss"] +=0.001 * fft_mse_loss_value
            """#### ====== 辅助头损失 ====== 误差在10%以内则忽略
            # max_prediction: [B, 1], x_start: [B, C, H, W]
              # [B, 1]
            max_value = max_value.to(max_prediction.device)
            max_diff = (max_value - max_prediction) ** 2
            max_mse  = max_diff.mean()   
            if self.call_count % 10 == 0:
                max_list = max_value.tolist()
                pred_list = max_prediction.tolist()

                if not isinstance(max_list, list):
                    max_list = [max_list]
                if not isinstance(pred_list, list):
                    pred_list = [pred_list] # Basic scalar handling

                max_str = [f"{v:.4f}" for v in max_list]
                pred_str = []
                for v in pred_list:
                    if isinstance(v, list) or isinstance(v, tuple):
                        # Handle nested list/tuple like [value]
                        pred_str.append(f"{v[0]:.4f}")
                    else:
                        # Handle flat list of floats
                        pred_str.append(f"{v:.4f}")

                print(f"max_value: {max_str} prediction: {pred_str}")

            terms["aux_loss"] = 0.1 * max_mse
            terms["loss"] = terms["mse"] + terms["aux_loss"]"""
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
