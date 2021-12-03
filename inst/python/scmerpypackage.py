# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 00:27:12 2021

@author: liuy47
"""

#._utils
import time

#._interfaces
from abc import ABC, abstractmethod
from torch import nn

#._base_selector
import torch
import numpy as np

#._owlqn
from functools import reduce
from torch.optim.optimizer import Optimizer

#._umap_torch_models
import warnings

#._umap_l1
from typing import List, Union, Optional
from typing import Type
import multiprocessing
from sklearn.decomposition import PCA



#._utils

class TicToc:
    def __init__(self):
        self.tic()

    def tic(self):
        self.t = [time.time()] * 2

    def toc(self):
        now = time.time()
        s = "Elapsed time: %.2f seconds. Total: %.2f seconds." % (now - self.t[1], now - self.t[0])
        self.t[1] = now
        return s

class VerbosePrint:
    def __init__(self, verbosity):
        self._verbosity = verbosity
        self.prints = [self.print0, self.print1, self.print2, self.print3]

    def __call__(self, priority, *args, **kwargs):
        """
        Print based on Verbosity
        :param priority: print the message only if the priority is smaller than verbosity
        :param args: args for normal print
        :param kwargs: kwargs for normal print
        :return: None
        """
        if priority < self._verbosity:
            print(*args, **kwargs)

    def print0(self, *args, **kwargs):
        self(0, *args, **kwargs)

    def print1(self, *args, **kwargs):
        self(1, *args, **kwargs)

    def print2(self, *args, **kwargs):
        self(2, *args, **kwargs)

    def print3(self, *args, **kwargs):
        self(3, *args, **kwargs)
        


#._interfaces
class _ABCSelector(ABC):
    @abstractmethod
    def __init__(self, verbosity):
        self.verbose_print = VerbosePrint(verbosity)

    @abstractmethod
    def fit(self, X, **kwargs):
        pass

    @abstractmethod
    def transform(self, X, **kwargs):
        pass

    @abstractmethod
    def fit_transform(self, X, **kwargs):
        pass

class _ABCTorchModel(ABC):
    @abstractmethod
    def __init__(self, P, X, w, beta, dtype, cdist_compute_mode, t_distr, must_keep, ridge):
        pass

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def get_w(self):
        pass
    
    @abstractmethod
    def use_gpu(self):
        pass

    

#._base_selector
class _BaseSelector(_ABCSelector):
    def __init__(self, w: Union[float, str, list, np.ndarray] = 'ones',
                 lasso: float = 1e-4, n_pcs: Optional[int] = None, perplexity: float = 30.,
                 use_beta_in_Q: bool = True,
                 max_outer_iter: int = 5, max_inner_iter: int = 20, owlqn_history_size: int = 100,
                 eps: float = 1e-12, verbosity: int = 2, torch_precision: Union[int, str, torch.dtype] = 32,
                 torch_cdist_compute_mode: str = "use_mm_for_euclid_dist",
                 t_distr: bool = True, n_threads: int = 1, use_gpu: bool = False, pca_seed: int = 0, ridge: float = 0.,
                 _keep_fitting_info: bool = False):
        super(_BaseSelector, self).__init__(verbosity)
        self._max_outer_iter = max_outer_iter
        self._max_inner_iter = max_inner_iter
        self._owlqn_history_size = owlqn_history_size
        self._n_pcs = n_pcs
        self.w = w
        self._lasso = lasso
        self._eps = eps
        self._use_beta_in_Q = use_beta_in_Q
        self._perplexity = perplexity
        self._torch_precision = torch_precision
        self._torch_cdist_compute_mode = torch_cdist_compute_mode
        self._t_distr = t_distr
        self._n_threads = n_threads
        self._use_gpu = use_gpu
        self._pca_seed = pca_seed
        self._ridge = ridge
        self._keep_fitting_info = _keep_fitting_info

    def get_mask(self, target_n_features=None):
        """
        Get the feature selection mask.
        For AnnData in scanpy, it can be used as adata[:, model.get_mask()]
        :param target_n_features: If None, all features with w > 0 are selected. If not None, only select
            `target_n_features` largest features
        :return: mask
        """
        if target_n_features is None:
            return self.w > 0.
        else:
            n_nonzero = (self.w > 0.).sum()
            if target_n_features > n_nonzero:
                raise ValueError(f"Only {n_nonzero} features have nonzero weights. "
                                 f"target_n_features may not exceed the number.")
            return self.w >= self.w[np.argpartition(self.w, -target_n_features)[-target_n_features]]

    def transform(self, X, target_n_features=None, **kwargs):
        """
        Shrink a matrix / AnnData object with full markers to the selected markers only.
        If such operation is not supported by your data object,
        you can do it manually using :func:`~UmapL1.get_mask`.
        :param X: Matrix / AnnData to be shrunk
        :param target_n_features: If None, all features with w > 0 are selected. If not None, only select
            `target_n_features` largest features
        :return: Shrunk matrix / Anndata
        """
        return X[:, self.get_mask(target_n_features)]

    def fit_transform(self, X, **kwargs):
        """
        Fit on a matrix / AnnData and then transfer it.
        :param X: The matrix / AnnData to be transformed
        :param kwargs: Other parameters for :func:`UmapL1.fit`.
        :return: Shrunk matrix / Anndata
        """
        return self.fit(X, **kwargs).transform(X)



#._owlqn
def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
    # Compute bounds of interpolation area
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1 ** 2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.

def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    # ported from https://github.com/torch/optim/blob/master/lswolfe.lua
    d_norm = d.abs().max()
    g = g.clone(memory_format=torch.contiguous_format)
    # evaluate objective and gradient using initial step
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        # check conditions
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

    # reached max number of iterations?
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = False
    # find high and low points in bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        # line-search bracket is so small
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        # compute new trial value
        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # test that we are making sufficient progress:
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # interpolation close to boundary
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # evaluate at 0.1 away from boundary
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # Evaluate new point
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if abs(gtd_new) <= -c2 * gtd:
                # Wolfe conditions satisfied
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)
            bracket_gtd[low_pos] = gtd_new

    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    g_new = bracket_g[low_pos]
    return f_new, g_new, t, ls_func_evals

class OWLQN0(Optimizer):
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.
    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.
    Arguments:
        lr (float): learning rate (default: 1)
        lasso (float): lasso (L1 regularization) strength (default: 1.) 
            (L2 regularization is differentiable and thus can be in the loss)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 lasso=1.,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None,
                 print_callback=print):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            lasso=lasso,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn)
        super(OWLQN0, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("OWLQN doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        lasso = self.param_groups[0]['lasso']
        # SL: with pseudo gradient
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)

            # SL: find psuedo-gradient
            # SL: at point 0
            view = view.clone()
            border_case = (p == 0.).reshape(-1)
            go_left = (view > lasso)
            go_right = (view < -lasso)
            stay = ~(go_left | go_right)
            view[go_left & border_case] -= lasso  # go left, but be slower (gradient - lasso > 0)
            view[go_right & border_case] += lasso  # go right, but also be slower (gradient + lasso < 0)
            view[stay & border_case] = 0.

            # SL: at non-zero points
            view[(p > 0.).reshape(-1)] += lasso
            view[(p < 0.).reshape(-1)] -= lasso

            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        # SL: WITH PROJECTION
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            sign0 = torch.sign(p)  # SL: get sign before update
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            p[sign0 != torch.sign(p)] = 0.  # SL: project to 0 if sign changed
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        # SL
        l1 = 0.
        for p in self._params:
            l1 = l1 + torch.norm(p, 1)
        l1 = l1 * self.param_groups[0]['lasso']
        loss = float(closure()) + l1
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        l1 = 0.  # SL: Add the L1 regularization
        for p in self._params:  # SL: ..
            l1 = l1 + torch.norm(p, 1)  # SL: ..
        l1 = l1 * self.param_groups[0]['lasso']  # SL: ..
        orig_loss = closure() + l1  # SL: ..
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    l1 = 0.  # SL: ..
                    for p in self._params:  # SL:
                        l1 = l1 + torch.norm(p, 1)  # SL:
                    l1 = l1 * self.param_groups[0]['lasso']  # SL:
                    with torch.enable_grad():
                        loss = float(closure())
                    loss += l1  # SL:
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss

class OWLQN0_masked(Optimizer):
    """Implements L-BFGS algorithm, heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`.
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.
    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.
    Arguments:
        lr (float): learning rate (default: 1)
        lasso (float): lasso (L1 regularization) strength (default: 1.)
            (L2 regularization is differentiable and thus can be in the loss)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    def __init__(self,
                 params,
                 lr=1,
                 lasso=None,
                 mask=None,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 history_size=100,
                 line_search_fn=None,
                 print_callback=print):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(
            lr=lr,
            lasso=lasso,
            mask=mask,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn)
        super(OWLQN0_masked, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("OWLQN doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        lasso = self.param_groups[0]['lasso']
        mask = self.param_groups[0]['mask']
        # SL: with pseudo gradient
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)

            # only apply to masked ones
            # SL: find psuedo-gradient
            # SL: at point 0
            view = view.clone()
            border_case = (p == 0.).reshape(-1)
            go_left = (view > lasso)
            go_right = (view < -lasso)
            stay = ~(go_left | go_right)
            view[go_left & border_case & mask] -= lasso[go_left & border_case & mask]  # go left, but be slower (gradient - lasso > 0)
            view[go_right & border_case & mask] += lasso[go_right & border_case & mask]  # go right, but also be slower (gradient + lasso < 0)
            view[stay & border_case & mask] = 0.

            # SL: at non-zero points
            gt0 = (p > 0.).reshape(-1)
            lt0 = (p < 0.).reshape(-1)
            view[gt0] += lasso[gt0]
            view[lt0] -= lasso[lt0]

            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        # SL: WITH PROJECTION
        mask = self.param_groups[0]['mask']
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            sign0 = torch.sign(p)  # SL: get sign before update
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            p[(sign0 != torch.sign(p)) & mask] = 0.  # SL: project to 0 if sign changed
            offset += numel
        assert offset == self._numel()

    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            p.copy_(pdata)

    def _directional_evaluate(self, closure, x, t, d):
        self._add_grad(t, d)
        # SL
        l1 = 0.
        for p in self._params:
            l1 = l1 + torch.norm(p * self.param_groups[0]['lasso'], 1)
        # l1 = l1 * self.param_groups[0]['lasso']
        loss = float(closure()) + l1
        flat_grad = self._gather_flat_grad()
        self._set_param(x)
        return loss, flat_grad

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        group = self.param_groups[0]
        lr = group['lr']
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        line_search_fn = group['line_search_fn']
        history_size = group['history_size']

        # NOTE: LBFGS has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        state = self.state[self._params[0]]
        state.setdefault('func_evals', 0)
        state.setdefault('n_iter', 0)

        # evaluate initial f(x) and df/dx
        l1 = 0.  # SL: Add the L1 regularization
        for p in self._params:  # SL: ..
            l1 = l1 + torch.norm(p * self.param_groups[0]['lasso'], 1)  # SL: ..
        orig_loss = closure() + l1  # SL: ..
        loss = float(orig_loss)
        current_evals = 1
        state['func_evals'] += 1

        flat_grad = self._gather_flat_grad()
        opt_cond = flat_grad.abs().max() <= tolerance_grad

        # optimal condition
        if opt_cond:
            return orig_loss

        # tensors cached in state (for tracing)
        d = state.get('d')
        t = state.get('t')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_flat_grad = state.get('prev_flat_grad')
        prev_loss = state.get('prev_loss')

        n_iter = 0
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] += 1

            ############################################################
            # compute gradient descent direction
            ############################################################
            if state['n_iter'] == 1:
                d = flat_grad.neg()
                old_dirs = []
                old_stps = []
                ro = []
                H_diag = 1
            else:
                # do lbfgs update (update memory)
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)  # y*s
                if ys > 1e-10:
                    # updating memory
                    if len(old_dirs) == history_size:
                        # shift history by one (limited-memory)
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)

                    # store new direction/step
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1. / ys)

                    # update scale of initial Hessian approximation
                    H_diag = ys / y.dot(y)  # (y*y)

                # compute the approximate (L-BFGS) inverse Hessian
                # multiplied by the gradient
                num_old = len(old_dirs)

                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                q = flat_grad.neg()
                for i in range(num_old - 1, -1, -1):
                    al[i] = old_stps[i].dot(q) * ro[i]
                    q.add_(old_dirs[i], alpha=-al[i])

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, H_diag)
                for i in range(num_old):
                    be_i = old_dirs[i].dot(r) * ro[i]
                    r.add_(old_stps[i], alpha=al[i] - be_i)

            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss

            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
            if state['n_iter'] == 1:
                t = min(1., 1. / flat_grad.abs().sum()) * lr
            else:
                t = lr

            # directional derivative
            gtd = flat_grad.dot(d)  # g * d

            # directional derivative is below tolerance
            if gtd > -tolerance_change:
                break

            # optional line search: user function
            ls_func_evals = 0
            if line_search_fn is not None:
                # perform line search, using user function
                if line_search_fn != "strong_wolfe":
                    raise RuntimeError("only 'strong_wolfe' is supported")
                else:
                    x_init = self._clone_param()

                    def obj_func(x, t, d):
                        return self._directional_evaluate(closure, x, t, d)

                    loss, flat_grad, t, ls_func_evals = _strong_wolfe(
                        obj_func, x_init, t, d, loss, flat_grad, gtd)
                self._add_grad(t, d)
                opt_cond = flat_grad.abs().max() <= tolerance_grad
            else:
                # no line search, simply move with fixed-step
                self._add_grad(t, d)
                if n_iter != max_iter:
                    # re-evaluate function only if not in last iteration
                    # the reason we do this: in a stochastic setting,
                    # no use to re-evaluate that function here
                    l1 = 0.  # SL: ..
                    for p in self._params:  # SL:
                        l1 = l1 + torch.norm(p * self.param_groups[0]['lasso'], 1)  # SL:
                    #l1 = l1 * self.param_groups[0]['lasso']  # SL:
                    with torch.enable_grad():
                        loss = float(closure())
                    loss += l1  # SL:
                    flat_grad = self._gather_flat_grad()
                    opt_cond = flat_grad.abs().max() <= tolerance_grad
                    ls_func_evals = 1

            # update func eval
            current_evals += ls_func_evals
            state['func_evals'] += ls_func_evals

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            if current_evals >= max_eval:
                break

            # optimal condition
            if opt_cond:
                break

            # lack of progress
            if d.mul(t).abs().max() <= tolerance_change:
                break

            if abs(loss - prev_loss) < tolerance_change:
                break

        state['d'] = d
        state['t'] = t
        state['old_dirs'] = old_dirs
        state['old_stps'] = old_stps
        state['ro'] = ro
        state['H_diag'] = H_diag
        state['prev_flat_grad'] = prev_flat_grad
        state['prev_loss'] = prev_loss

        return orig_loss

def OWLQN(params,
          lr=1,
          lasso: Union[float, int, np.ndarray, list] = 1.,
          max_iter=20,
          max_eval=None,
          tolerance_grad=1e-7,
          tolerance_change=1e-9,
          history_size=100,
          line_search_fn=None,
          print_callback=print,
          use_gpu=False):
    """
    Dispatching to proper OWLQN class depending on the structure of lasso
    :param params:
    :param lr:
    :param lasso:
    :param max_iter:
    :param max_eval:
    :param tolerance_grad:
    :param tolerance_change:
    :param history_size:
    :param line_search_fn:
    :param print_callback:
    :return:
    """
    if isinstance(lasso, float) or isinstance(lasso, int):
        return OWLQN0(params, lr, float(lasso),
                      max_iter, max_eval, tolerance_grad, tolerance_change, history_size, line_search_fn,
                      print_callback)
    if isinstance(lasso, list) or isinstance(lasso, np.ndarray):
        mask = (np.ndarray != 0.)
        lasso = torch.tensor(lasso)
        if use_gpu:
            lasso = lasso.cuda()
        return OWLQN0_masked(params, lr, lasso, mask,
                             max_iter, max_eval, tolerance_grad, tolerance_change, history_size, line_search_fn,
                             print_callback)



#._base_torch_model
class _BaseTorchModel(nn.Module, _ABCTorchModel):
    def __init__(self, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True, must_keep=None, ridge=0.):
        super(_BaseTorchModel, self).__init__()
        if dtype == "32" or dtype == 32:
            self.dtype = torch.float32
        elif dtype == "64" or dtype == 64:
            self.dtype = torch.float64
        else:
            self.dtype = dtype

        if must_keep is None:
            self.must_keep = None
        else:
            self.must_keep = must_keep.squeeze()
        self.cdist_compute_mode = cdist_compute_mode
        self.t_distr = t_distr
        self.ridge = ridge

    def preprocess_X(self, X):
        if self.must_keep is None:
            add_pdist2 = 0.
            X = torch.tensor(X, dtype=self.dtype, requires_grad=False)
            n_instances, n_features = X.shape
        else:
            must_keep = self.must_keep
            add_X = torch.tensor(X * must_keep.reshape([1, -1]), dtype=self.dtype, requires_grad=False)
            add_pdist2 = torch.square(torch.cdist(add_X, add_X, compute_mode=self.cdist_compute_mode))
            X = torch.tensor(X[:, must_keep == 0], dtype=self.dtype, requires_grad=False)
            n_instances, n_features = X.shape
            n_features = (must_keep == 0).sum()
        return X, add_pdist2, n_instances, n_features

    def init_w(self, w):
        if isinstance(w, float) or isinstance(w, int):
            w = np.zeros([1, self.n_features]) + w
        elif isinstance(w, str) and w == 'uniform':
            w = np.random.uniform(size=[1, self.n_features])
        elif isinstance(w, str) and w == 'ones':
            w = np.ones([1, self.n_features])
        else:
            w = np.array(w).reshape([1, self.n_features])
        self.W = torch.nn.Parameter(
            torch.tensor(w, dtype=self.dtype, requires_grad=True))

    def get_w0(self):
        if self.W.is_cuda:
            w0 = self.W.detach().cpu().numpy().squeeze()
        else:
            w0 = self.W.detach().numpy().squeeze()
        return w0

    def get_w(self):
        w0 = self.get_w0()
        if self.must_keep is None:
            w = w0
        else:
            w = self.must_keep.copy()
            w[self.must_keep == 0] += w0
        return w



#._umap_torch_models
class _BaseUmapModel(_BaseTorchModel):
    def __init__(self, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist", t_distr=True, must_keep=None, ridge=0.):
        """
        Base class for umap models
        :param cdist_compute_mode: compute mode for torch.cdist. By default, "use_mm_for_euclid_dist" to (daramatically)
            improve performance. However, if numerical stability became an issue, "donot_use_mm_for_euclid_dist" may be
            used.
        :param dtype: The dtype used inside torch model. By default, tf.float32 (a.k.a. tf.float) is used.
            However, if precision become an issue, tf.float64 may be worth trying.
        """
        super(_BaseUmapModel, self).__init__(dtype, cdist_compute_mode, t_distr, must_keep, ridge)

        self.epsilon = torch.tensor(1e-30, dtype=self.dtype)

    @staticmethod
    def preprocess_P(P):
        P = P + P.T - P * P.T
        P = P / np.sum(P)
        P = np.maximum(P, 0.)

        return P

    def calc_kl(self, P, X, W, beta, add_pdist2):
        Y = X * W
        if isinstance(add_pdist2, float):
            pdist2 = torch.cdist(Y, Y, compute_mode=self.cdist_compute_mode)
        else:
            pdist2 = torch.sqrt(torch.square(torch.cdist(Y, Y, compute_mode=self.cdist_compute_mode)) + add_pdist2)
        pdist2 = pdist2 - torch.min(pdist2.clone().fill_diagonal_(float("inf")), dim=1, keepdim=True)[0]

        if beta is not None:
            pdist2 = pdist2 * beta

        temp = 1. / (1. + pdist2)
        temp = temp + temp.T - temp * temp.T
        temp.fill_diagonal_(0.)

        Q = temp / temp.sum()
        Q = torch.max(Q, self.epsilon)

        mask = P > 0.

        Q = Q[mask]
        P = P[mask]

        kl = P * torch.log(P / Q)
        kl = kl.sum()
        return kl.sum()

class _RegUmapModel(_BaseUmapModel):
    def __init__(self, P, X, w, beta, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist",
                 t_distr=True, must_keep=None, ridge=0.):
        super(_RegUmapModel, self).__init__(dtype, cdist_compute_mode, t_distr, must_keep, ridge)

        self.P = torch.tensor(self.preprocess_P(P), dtype=self.dtype, requires_grad=False)
        self.X, self.add_pdist2, self.n_instances, self.n_features = self.preprocess_X(X)

        if beta is not None:
            self.beta = torch.tensor(beta, dtype=self.dtype, requires_grad=False)
        else:
            self.beta = None

        self.init_w(w)

    def forward(self):
        kl = self.calc_kl(self.P, self.X, self.W, self.beta, self.add_pdist2)
        if self.ridge > 0.:
            return kl + torch.sum(self.W ** 2) * self.ridge
        else:
            return kl

    def use_gpu(self):
        self.P = self.P.cuda()
        self.X = self.X.cuda()
        self.epsilon = self.epsilon.cuda()
        if self.beta is not None:
            self.beta = self.beta.cuda()
        if not isinstance(self.add_pdist2, float):
            self.add_pdist2 = self.add_pdist2.cuda()
        self.cuda()

class _StratifiedRegUmapModel(_BaseUmapModel):
    def __init__(self, Ps, Xs, w, betas, dtype=torch.float, cdist_compute_mode="use_mm_for_euclid_dist",
                 t_distr=True, must_keep=None, ridge=0.):
        super(_StratifiedRegUmapModel, self).__init__(dtype, cdist_compute_mode, t_distr, must_keep, ridge)

        self.n_batches = len(Xs)

        if not (len(Ps) == self.n_batches):
            raise ValueError("Lengths of Ps and Xs must be equal.")

        if betas is not None:
            self.betas = betas
            if not (len(Ps) == self.n_batches):
                raise ValueError("Lengths of Xs and betas must be equal.")
        else:
            self.betas = betas

        self.Ps = [torch.tensor(self.preprocess_P(P), dtype=self.dtype, requires_grad=False) for P in Ps]

        self.Xs = []
        self.add_pdist2s = []
        self.n_instances = []
        self.n_features = None
        for X in Xs:
            X, add_pdist2, n_instances, n_features = self.preprocess_X(X)
            self.Xs.append(X)
            self.add_pdist2s.append(add_pdist2)
            self.n_instances.append(n_instances)
            if self.n_features is None:
                self.n_features = n_features
            elif self.n_features != n_features:
                raise ValueError("All matrices must have the same number of features.")

        self.betas = []
        if betas is not None:
            for beta in betas:
                self.betas.append(torch.tensor(beta, dtype=self.dtype, requires_grad=False))
        else:
            self.betas = None

        self.init_w(w)

    def forward(self):
        loss = 0
        for batch in range(self.n_batches):
            loss += self.calc_kl(self.Ps[batch], self.Xs[batch], self.W,
                                 self.betas[batch] if self.betas is not None else None,
                                 self.add_pdist2s[batch])
        if self.ridge > 0.:
            return loss / self.n_batches + torch.sum(self.W ** 2) * self.ridge
        else:
            return loss / self.n_batches

    def use_gpu(self):
        self.Ps = [P.cuda() for P in self.Ps]
        self.X = [X.cuda() for X in self.Xs]
        self.epsilon = self.epsilon.cuda()
        if self.betas is not None:
            self.betas = [beta.cuda() for beta in self.betas]
        self.add_pdist2s = [add_pdist2 if isinstance(self.add_pdist2, float) else add_pdist2.cuda()
                            for add_pdist2 in self.add_pdist2s]
        self.cuda()



#._umap_l1
class UmapL1(_BaseSelector):
    def __init__(self, *, w: Union[float, str, list, np.ndarray] = 'ones',
                 lasso: float = 1e-4, n_pcs: Optional[int] = None, perplexity: float = 30.,
                 use_beta_in_Q: bool = True,
                 max_outer_iter: int = 5, max_inner_iter: int = 20, owlqn_history_size: int = 100,
                 eps: float = 1e-12, verbosity: int = 2, torch_precision: Union[int, str, torch.dtype] = 32,
                 torch_cdist_compute_mode: str = "use_mm_for_euclid_dist",
                 t_distr: bool = True, n_threads: int = 1, use_gpu: bool = False, pca_seed: int = 0, ridge: float = 0.,
                 _keep_fitting_info: bool = False):
        """
        UmapL1 model
        :param w: initial value of w, weight of each marker. Acceptable values are 'ones' (all 1),
            'uniform' (random [0, 1] values), float numbers (all set to that number),
            or a list or numpy array with specific numbers.
        :param lasso: lasso strength (i.e., strength of L1 regularization in elastic net)
        :param n_pcs: Number of PCs used to generate P matrix. Skip PCA if set to `None`.
        :param perplexity: perplexity of t-SNE modeling
        :param use_beta_in_Q: whether to use the cell specific sigma^2 calculated from P in Q. (1 / beta)
        :param max_outer_iter: number of iterations of OWL-QN
        :param max_inner_iter: number of iterations inside OWL-QN
        :param owlqn_history_size: history size for OWL-QN.
        :param eps: epsilon for considering a value to be 0.
        :param verbosity: verbosity level (0 ~ 2).
        :param torch_precision: The dtype used inside torch model. By default, tf.float32 (a.k.a. tf.float) is used.
            However, if precision become an issue, tf.float64 may be worth trying. You can input 32, "32", 64, or "64".
        :param torch_cdist_compute_mode: cdist_compute_mode: compute mode for torch.cdist. By default,
            "use_mm_for_euclid_dist" to (daramatically) improve performance. However, if numerical stability became an
            issue, "donot_use_mm_for_euclid_dist" may be used instead. This option does not affect distances computed
            outside of pytorch, e.g., matrix P. Only matrix Q is affect.
        :param t_distr: By default, use t-distribution (1. / (1. + pdist2)) for Q.
            Use Normal distribution instead (exp(-pdist2)) if set to False. The latter one is not stable.
        :param n_threads: number of threads (currently only for calculating P and beta)
        :param use_gpu: whether to use GPU to train the model.
        :param pca_seed: random seed used by PCA (if applicable)
        :param ridge: ridge strength (i.e., strength of L2 regularization in elastic net)
        :param _keep_fitting_info: if `True`, write similarity matrix P to `self.P` and PyTorch model to `self.model`
        """
        super(UmapL1, self).__init__(w, lasso, n_pcs, perplexity, use_beta_in_Q, max_outer_iter, max_inner_iter,
                                     owlqn_history_size, eps, verbosity, torch_precision, torch_cdist_compute_mode,
                                     t_distr, n_threads, use_gpu, pca_seed, ridge, _keep_fitting_info)


    def fit(self, X, *, X_teacher=None, batches=None, P=None, beta=None, must_keep=None):
        """
        Select markers from one dataset to keep the cell-cell similarities in the same dataset
        :param X: data matrix (cells (rows) x genes/proteins (columns))
        :param X_teacher: get target similarities from this dataset
        :param batches: (optional) batch labels
        :param P: The P matrix, if calculated in advance
        :param beta: The beta associated with P, if calculated in advance
        :param must_keep: A boolean vector indicating if a feature must be kept.
            Those features will have a fixed weight 1.
        :return:
        """
        tictoc = TicToc()
        trans = True
        if X_teacher is None: # if there is no other assay to mimic, just mimic itself
            X_teacher = X
            trans = False

        if batches is None:
            if must_keep is None and (isinstance(self._lasso, float) or isinstance(self._lasso, str)):
                model_class = _RegUmapModel #_SimpleRegTsneModel
            else:
                model_class = _RegUmapModel

            if self._n_pcs is None:
                P, beta = self._resolve_P_beta(X_teacher, P, beta, self._perplexity, tictoc, self.verbose_print.prints,
                                               self._n_threads)
            else:
                pcs = PCA(self._n_pcs, random_state=self._pca_seed).fit_transform(X_teacher)
                # print(pcs)
                P, beta = self._resolve_P_beta(pcs, P, beta, self._perplexity, tictoc, self.verbose_print.prints,
                                               self._n_threads)
        else:
            model_class = _StratifiedRegUmapModel
            if P is None:
                if trans:
                    Xs = []
                    for batch in np.unique(batches):
                        batch_mask = (batches == batch)
                        Xs.append(X[batch_mask, :])
                    X = Xs
                    _, P, beta = self._resolve_batches(X_teacher, None, batches, self._n_pcs, self._perplexity, tictoc,
                                                       self.verbose_print, self._pca_seed, self._n_threads)
                else:
                    X, P, beta = self._resolve_batches(X_teacher, None, batches, self._n_pcs, self._perplexity, tictoc,
                                                       self.verbose_print, self._pca_seed, self._n_threads)
            else:
                raise NotImplementedError()

        if self._keep_fitting_info:
            self.P = P

        return self._fit_core(X, P, beta, must_keep, model_class, tictoc)

    def get_mask(self, target_n_features=None):
        """
        Get the feature selection mask.
        For AnnData in scanpy, it can be used as adata[:, model.get_mask()]
        :param target_n_features: If None, all features with w > 0 are selected. If not None, only select
            `target_n_features` largest features
        :return: mask
        """
        if target_n_features is None:
            return self.w > 0.
        else:
            n_nonzero = (self.w > 0.).sum()
            if target_n_features > n_nonzero:
                raise ValueError(f"Only {n_nonzero} features have nonzero weights. "
                                 f"target_n_features may not exceed the number.")
            return self.w >= self.w[np.argpartition(self.w, -target_n_features)[-target_n_features]]

    def transform(self, X, target_n_features=None, **kwargs):
        """
        Shrink a matrix / AnnData object with full markers to the selected markers only.
        If such operation is not supported by your data object,
        you can do it manually using :func:`~UmapL1.get_mask`.
        :param X: Matrix / AnnData to be shrunk
        :param target_n_features: If None, all features with w > 0 are selected. If not None, only select
            `target_n_features` largest features
        :return: Shrunk matrix / Anndata
        """
        return X[:, self.get_mask(target_n_features)]

    def fit_transform(self, X, **kwargs):
        """
        Fit on a matrix / AnnData and then transfer it.
        :param X: The matrix / AnnData to be transformed
        :param kwargs: Other parameters for :func:`UmapL1.fit`.
        :return: Shrunk matrix / Anndata
        """
        return self.fit(X, **kwargs).transform(X)

    @classmethod
    def tune(cls, target_n_features, X=None, *, X_teacher=None, batches=None,
             P=None, beta=None, must_keep=None, perplexity=30., n_pcs=None, w='ones',
             min_lasso=1e-8, max_lasso=1e-2, tolerance=0, smallest_log10_fold_change=0.1, max_iter=100,
             return_P_beta=False, n_threads=6,
             **kwargs):
        """
        Automatically find proper lasso strength that returns the preferred number of markers
        :param target_n_features: number of features
        :param return_P_beta: controls what to return
        :param kwargs: all other parameters are the same for a UmapL1 model or :func:`UmapL1.fit`.
        :return: if return_P_beta is True and there are batches, (model, X, P, beta);
                 if return_P_beta is True and there is no batches, (model, P, beta);
                 otherwise, only model by default.
        """
        if "lasso" in kwargs:
            raise ValueError("Parameter lasso should be substituted by max_lasso and min_lasso to set a range.")
        if "verbosity" in kwargs:
            verbosity = kwargs['verbosity']
        else:
            verbosity = 3
        verbose_print = VerbosePrint(verbosity)
        tictoc = TicToc()

        n_features = X.shape[1]

        # initialize w
        if isinstance(w, float) or isinstance(w, int):
            w = np.zeros([1, n_features]) + w
        elif isinstance(w, str) and w == 'uniform':
            w = np.random.uniform(size=[1, n_features])
        elif isinstance(w, str) and w == 'ones':
            w = np.ones([1, n_features])
        else:
            w = np.array(w).reshape([1, n_features])

        max_log_lasso = np.log10(max_lasso)
        min_log_lasso = np.log10(min_lasso)

        if X_teacher is None: # if there is no other assay to mimic, just mimic itself
            X_teacher = X

        if batches is None:
            model_class = _RegUmapModel
            if n_pcs is None:
                P, beta = cls._resolve_P_beta(X_teacher, P, beta, perplexity, tictoc, verbose_print.prints, n_threads)
            else:
                pcs = PCA(n_pcs).fit_transform(X_teacher)
                P, beta = cls._resolve_P_beta(pcs, P, beta, perplexity, tictoc, verbose_print.prints, n_threads)
        else:
            model_class = _StratifiedRegUmapModel
            if P is None:
                X, P, beta = cls._resolve_batches(X_teacher, None, batches, n_pcs, perplexity, tictoc, verbose_print,
                                                  n_threads)

        sup = n_features
        inf = 0

        model = None
        for it in range(max_iter):
            log_lasso = max_log_lasso / 2 + min_log_lasso / 2
            verbose_print(0, "Iteration", it, "with lasso =", 10 ** log_lasso,
                          "in [", 10 ** min_log_lasso, ",", 10 ** max_log_lasso, "]...", end=" ")
            model = cls(w=w, lasso=10 ** log_lasso, n_pcs=n_pcs, perplexity=perplexity, **kwargs)
            n = model._fit_core(X, P, beta, must_keep, model_class, tictoc).get_mask().sum()
            verbose_print(0, "Done. Number of features:", n, ".", tictoc.toc())
            if np.abs(n - target_n_features) <= tolerance:  # Good number of features, return
                break

            #if it > 0 and np.abs(log_lasso - prev_log_lasso) < smallest_log10_fold_change:
            #    warnings.warn("smallest_log10_fold_change reached before achieving target number of features.")
            #    break

            #prev_log_lasso = log_lasso

            if n > target_n_features:  # Too many features, need more l1 regularization
                if n <= sup:
                    sup = n
                else:
                    warnings.warn("Monotonicity is violated. Value larger than current supremum. "
                                  "Binary search may fail. "
                                  "Consider use more max_outer_iter (default: 5) and max_inner_iter (default: 20).")
                min_log_lasso = log_lasso
            elif n < target_n_features:  # Too few features, need less l1 regularization
                if n >= inf:
                    inf = n
                else:
                    warnings.warn("Monotonicity is violated. Value lower than current infimum. "
                                  "Binary search may fail. "
                                  "Consider use more max_outer_iter (default: 5) and max_inner_iter (default: 20).")
                max_log_lasso = log_lasso
        else:  # max_iter reached
            warnings.warn("max_iter before reached achieving target number of features.")

        if return_P_beta:
            if batches is None:
                return model, P, beta
            else:
                return model, X, P, beta
        else:
            return model

    @staticmethod
    def _resolve_P_beta(X, P, beta, perplexity, tictoc, print_callbacks, n_threads):
        if P is None and beta is None:
            print_callbacks[0]("Calculating distance matrix and scaling factors...")
            P, beta = UmapL1._x2p(X, perplexity=perplexity, print_callback=print_callbacks[1], n_threads=n_threads)
            print_callbacks[0]("Done.", tictoc.toc())
        elif P is None and beta is not None:
            print_callbacks[0]("Calculating distance matrix...")
            P = UmapL1._x2p_given_beta(X, beta)
            print_callbacks[0]("Done.", tictoc.toc())

        return P, beta

    @staticmethod
    def _resolve_batches(X, beta, batches, n_pcs, perplexity, tictoc, verbose_print, pca_seed, n_threads):
        batches = np.array(batches)
        batch_names = np.unique(batches)
        Xs = []
        Ps = []
        betas = []
        for batch in batch_names:
            batch_mask = (batches == batch)
            verbose_print(0, "Batch", batch, "with", sum(batch_mask), "instances.")

            Xs.append(X[batch_mask, :])
            if n_pcs is None:
                if beta is not None:
                    new_beta = beta[batches == batch]
                else:
                    new_beta = None
                P, new_beta = UmapL1._resolve_P_beta(Xs[-1], None, new_beta, perplexity, tictoc, verbose_print.prints, n_threads)
            else:
                pcs = PCA(n_pcs, random_state=pca_seed).fit_transform(Xs[-1])
                P, new_beta = UmapL1._resolve_P_beta(pcs, None, None, perplexity, tictoc, verbose_print.prints, n_threads)
            Ps.append(P)
            betas.append(new_beta)
        return Xs, Ps, betas

    def _fit_core(self, X, P, beta, must_keep, model_class: Type[_ABCTorchModel], tictoc):

        if self._use_beta_in_Q:
            self.verbose_print(0, "Creating model without batches...")
            model = model_class(P, X, self.w, beta, self._torch_precision, self._torch_cdist_compute_mode,
                                self._t_distr, must_keep, ridge=self._ridge)
        else:
            self.verbose_print(0, "Creating batch-stratified model...")
            model = model_class(P, X, self.w, None, self._torch_precision, self._torch_cdist_compute_mode,
                                self._t_distr, must_keep, ridge=self._ridge)

        if self._use_gpu:
            model.use_gpu()

        if self._lasso > 0.:
            self.verbose_print(0, "Optimizing using OWLQN (because lasso is nonzero)...")
            optimizer = OWLQN(model.parameters(), lasso=self._lasso, line_search_fn="strong_wolfe",
                              max_iter=self._max_inner_iter, history_size=self._owlqn_history_size, lr=1.)
        else:
            self.verbose_print(0, "Optimizing using LBFGS (because lasso is zero)...")
            optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe",
                                          max_iter=self._max_inner_iter, history_size=self._owlqn_history_size, lr=1.)

        if self._keep_fitting_info:
            self.model = model

        for t in range(self._max_outer_iter):
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                loss = model.forward()

                if loss.requires_grad:
                    loss.backward()

                return loss

            loss = optimizer.step(closure)
            self.verbose_print(1, t, 'loss (before this step):', loss.item(),
                               "Nonzero (after):", (np.abs(model.get_w0()) > self._eps).sum(),
                               tictoc.toc())

            self.w = model.get_w() # In case the user wants to interrupt the training

        loss = model.forward()
        self.verbose_print(1, 'Final', 'loss:', loss.item(), "Nonzero:", (np.abs(model.get_w0()) > self._eps).sum(),
                           tictoc.toc())

        return self

    @staticmethod
    def _Hbeta(D=np.array([]), beta=1.0):
        """
            Compute the perplexity and the P-row for a specific value of the
            precision of a Gaussian distribution.
        """
        # Compute P-row and corresponding perplexity
        P = np.exp(-(D - np.min(D)) * beta)
        H = sum(P)
        return H, P

    @staticmethod
    def _x2p(X=np.array([]), tol=1e-5, perplexity=30.0, print_callback=print, *, n_threads):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """
        if n_threads > 1:
            return UmapL1._x2p_parallel(X, tol, perplexity, print_callback, n_threads)

        # Initialize some variables
        print_callback("Computing pairwise distances...")
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        D = np.sqrt(np.maximum(D, 0))
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perplexity)

        # Loop over all datapoints
        for i in range(n):

            # Print progress
            if i % 500 == 0:
                print_callback("Computing P-values for point %d of %d..." % (i, n))

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = UmapL1._Hbeta(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0

            while (not np.abs(Hdiff) < tol) and tries < 100:
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] + 1.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = UmapL1._Hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        # Return final P-matrix
        print_callback("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P, beta

    @staticmethod
    def _x2p_given_beta(X, beta):
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        for i in range(n):
            (H, P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]) = UmapL1._Hbeta(
                D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))], beta[i])
        return P

    @staticmethod
    def _x2p_process(Di, logU, tol):
        beta = 1.
        betamin = -np.inf
        betamax = np.inf
        (H, thisP) = UmapL1._Hbeta(Di, beta)

        Hdiff = H - logU
        tries = 0

        while (not np.abs(Hdiff) < tol) and tries < 100:
            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta
                if betamax == np.inf or betamax == -np.inf:
                    beta = beta * 2.
                else:
                    beta = (beta + betamax) / 2.
            else:
                betamax = beta
                if betamin == np.inf or betamin == -np.inf:
                    beta = beta / 2.
                else:
                    beta = (beta + betamin) / 2.

            # Recompute the values
            (H, thisP) = UmapL1._Hbeta(Di, beta)
            Hdiff = H - logU
            tries += 1
        return thisP, beta
        # Set the final row of P

    @staticmethod
    def _x2p_parallel(X=np.array([]), tol=1e-5, perplexity=30.0, print_callback=print, n_threads=6):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """

        # Initialize some variables
        print_callback("Computing pairwise distances...")
        (n, d) = X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        D = np.sqrt(np.maximum(D, 0))
        logU = np.log2(perplexity)

        # Loop over all datapoints
        # for i in range(n):
        # Compute the Gaussian kernel and entropy for the current precision

        print_callback("Using", n_threads, "threads...")
        parameters = [(D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))], logU, tol) for i in range(n)]
        with multiprocessing.Pool(n_threads) as pool:
            results = pool.starmap(UmapL1._x2p_process, parameters)

        beta = np.ones((n, 1))
        P = np.zeros((n, n))
        for i in range(n):
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = results[i][0]
            beta[i] = results[i][1]

        # Return final P-matrix
        print_callback("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P, beta
    
    
    
    
    
    
    







#scmerpy
def scmerpy(trimbetasmat, 
            probes, 
            samples, 
            annodat, 
            labeldat, 
            K, 
            k, 
            lasso = 5.5e-6, 
            ridge = 0, 
            n_pcs = 100, 
            perplexity = 30, 
            threads = 6, 
            seed = 1234, 
            savefigures = True):
    
    import numpy as np
    import pandas as pd
    
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
    #Must add this command when running the function on computing nodes, otherwise 
    #an error TclError: couldn't connect to display "localhost" will appear.
    #The reason is matplotlib chooses an x-using backend by default.
    #The solution is to add the code `matplotlib.use('Agg')` before any other 
    #pylab/matplotlib/pyplot import
    
    import scanpy as sc
    #import scmer
    
    import time
    
    message = 'Start: ' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(message)
    
    n_pcs = int(n_pcs)
    perplexity = int(perplexity)
    threads = int(threads)
    seed = int(seed)
    
    if annodat is not None and labeldat is not None: 
        anno = annodat.set_index('sentrix', drop=False)
        anno = anno.loc[samples,]
    
        y = pd.Series(labeldat)
        y.index = annodat.sentrix
        
        y = y.loc[samples,]
        y = pd.DataFrame(y)
        y.columns = ['label']
    
        
    adata = sc.AnnData(trimbetasmat)
    
    if annodat is not None and labeldat is not None: 
        adata.obs = y
        adata.obs_names = samples
        
    adata.var_names = probes
    
    #Scanpy analysis
    sc.settings.set_figure_params(dpi = 300, facecolor = 'white')
    #sc.pl.highest_expr_genes(adata, n_top = 20)
    
    message = 'PCA before: ' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(message)
    
    sc.tl.pca(adata, svd_solver = 'arpack')
    
    if savefigures:
        
        if annodat is not None and labeldat is not None: 
            sc.pl.pca(adata, color = 'label', title = 'Before', 
                      save = 'before.' + K + '.' + \
                          k + '.' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                      show = False)
        else:
            sc.pl.pca(adata, title = 'Before', 
                      save = 'before.' + K + '.' + \
                          k + '.' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                      show = False)
    
    #sc.pl.pca_variance_ratio(adata, log = False)
    
    message = 'UMAP before: ' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(message)
        
    sc.pp.neighbors(adata, n_neighbors = 30, n_pcs = 30)
    sc.tl.umap(adata)
    
    if savefigures:
        
        if annodat is not None and labeldat is not None: 
            sc.pl.umap(adata, color = 'label', title = 'Before', 
                       save = 'before.' + K + '.' + \
                           k + '.' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                       show = False)
        else:
            sc.pl.umap(adata, title = 'Before', 
                       save = 'before.' + K + '.' + \
                           k + '.' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                       show = False)
        
    
    
    #SCMER feature selection
    
    message = 'SCMER: ' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(message)
    
    model = UmapL1(lasso = lasso, ridge = ridge, 
                   n_pcs = n_pcs, perplexity = perplexity, 
                   use_beta_in_Q = False, n_threads = threads, 
                   pca_seed = seed)
    model.fit(adata.X)
    
    #print(*adata.var_names[model.get_mask()])
        
    #PCA and UMAP validation
    new_adata = model.transform(adata)
    
    message = 'PCA after: ' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(message)
    
    sc.tl.pca(new_adata, svd_solver = 'arpack')
    
    if savefigures:
        
        if annodat is not None and labeldat is not None:
            sc.pl.pca(new_adata, color = 'label', title = 'After', 
                      save = 'after.' + K + '.' + \
                          k + '.' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                      show = False)
        else:
            sc.pl.pca(new_adata, title = 'After', 
                      save = 'after.' + K + '.' + \
                          k + '.' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                      show = False)
    
    
    message = 'UMAP after: ' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(message)
    
    sc.pp.neighbors(new_adata, n_pcs = 30, use_rep = "X_pca", n_neighbors = 30)
    sc.tl.umap(new_adata)
    
    if savefigures:
        
        if annodat is not None and labeldat is not None:
            sc.pl.umap(new_adata, color = 'label', title = 'After', 
                       save = 'after.' + K + '.' + \
                           k + '.' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                       show = False)
        else:
            sc.pl.umap(new_adata, title = 'After', 
                       save = 'after.' + K + '.' + \
                           k + '.' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                       show = False)
    
    
    features = adata.var_names[model.get_mask()]
    features = pd.DataFrame(features)
    features.columns = ['features']
    
    #savename = 'features.' + K + '.' + k + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    #robjects.r.assign(savename, features)
    #robjects.r("save(" + savename + ", file = '" + savename + ".RData')")
    
    message = 'Complete: ' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    print(message)
            
    return features


