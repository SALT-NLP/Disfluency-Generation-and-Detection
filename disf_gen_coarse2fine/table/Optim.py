import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class Optim(object):

    def __init__(self, method, lr, alpha, max_grad_norm, model_size,
                 lr_decay=1, start_decay_at=None,
                 beta1=0.9, beta2=0.98, warm_up_step=400, warm_up_factor=1.0,
                 opt=None):
        self.last_metric = None
        self.lr = lr
        self.model_size=model_size
        self.factor=warm_up_factor
        self.warmup=warm_up_step
        self.alpha = alpha
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self._step = 0
        self.betas = (beta1, beta2)
        self.opt = opt

    def set_parameters(self, params):
        self.params = [p for p in params if p.requires_grad]
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.params, lr=self.lr, alpha=self.alpha)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr,
                                        betas=self.betas, eps=1e-9)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def _setRate(self, lr):
        self.lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr
        #self.optimizer.param_groups[0]['lr'] = self.lr

    def step(self):
        "Compute gradients norm."
        self._step += 1

        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        if self.opt.warm_up:
            lr = self.rate()
            self.lr=lr
            for p in self.optimizer.param_groups:
                p['lr'] = lr
        
        self.optimizer.step()


    def updateLearningRate(self, metric, epoch):
        """
        Decay learning rate if val perf does not improve
        or we hit the start_decay_at limit.
        """
        if self.opt.warm_up:
            print("Learning rate: %g" % self.lr)
        else:
            if (self.start_decay_at is not None) and (epoch >= self.start_decay_at):
                self.start_decay = True
            if (self.last_metric is not None) and (metric is not None) and (metric > self.last_metric):
                self.start_decay = True

            if self.start_decay:
                self.lr = self.lr * self.lr_decay
                print("Decaying learning rate to %g" % self.lr)

            self.last_metric = metric
            # self.optimizer.param_groups[0]['lr'] = self.lr

            for p in self.optimizer.param_groups:
                p['lr'] = self.lr

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-1) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   optim.Adam(model.parameters(),
                                    lr=0, betas=(0.9, 0.98), eps=1e-9))