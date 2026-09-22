import torch

class StreamingOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, gamma=0.99, lamda=0.8, beta=0.99995, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("lr must be non-negative")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be the retained fraction in [0, 1)")
        if not 0.0 < eps:
            raise ValueError("eps must be positive")
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, eps=eps)
        self.beta = beta
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, delta=1.0, reset=False):
        delta = float(delta)
        abs_delta = abs(delta)
        for group in self.param_groups:
            trace_decay = group["gamma"] * group["lamda"]
            step_size = group["lr"] * delta
            eps = group["eps"]
            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["eligibility_trace"] = torch.zeros_like(p)
                    state["max_v"] = torch.zeros_like(p)
                e = state["eligibility_trace"]
                max_v = state["max_v"]
                e.mul_(trace_decay).add_(grad)
                torch.maximum(max_v.mul_(self.beta), e.abs().mul_(abs_delta), out=max_v)
                p.addcdiv_(e, max_v + eps, value=step_size)
                if reset:
                    e.zero_()

class BoundedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("lr must be non-negative")
        if not all(0.0 <= beta < 1.0 for beta in betas) or len(betas) != 2:
            raise ValueError("betas must contain two values in [0, 1)")
        if not 0.0 < eps:
            raise ValueError("eps must be positive")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("BoundedOptimizer does not support sparse gradients")
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["max_v"] = torch.zeros_like(param)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                max_v = state["max_v"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                corrected_avg = exp_avg / (1 - beta1 ** state["step"])
                torch.maximum(max_v.mul_(beta2), corrected_avg.abs(), out=max_v)
                param.addcdiv_(corrected_avg, max_v + group["eps"], value=-group["lr"])
        return loss