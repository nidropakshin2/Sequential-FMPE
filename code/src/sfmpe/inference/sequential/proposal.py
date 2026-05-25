import torch
from torchdiffeq import odeint

from sfmpe.core.distributions import Distribution
from sfmpe.flow.sampler import ODESampler
from sfmpe.tasks.base_task import Task

class ProposalParams:
    """
        method: None | str, one of None, 'NPE-A', 'NPE-B', 'NPE-C'
        x_0: None | torch.Tensor, if sequential mode you have to specify x_0
        weight: None | float, weight used for 'NPE-B' method
        n_steps: None | int, steps of ODESampler
    """
    task: None | Task
    method: None | str 
    method_params: None | dict
    theta_0: torch.Tensor 
    x_0: None | torch.Tensor 
    # weight: None | float = 0.5
    n_steps: int = 8



class Proposal(Distribution):
    def __init__(self, flow_model, params: ProposalParams):
        self.flow_model = flow_model
        self.params = params
        self.sampler = ODESampler(flow_model)
    

    def sample(self, size, **kwargs) -> torch.Tensor:
        # TODO: убрать это дерьмо и сделать просто Truncated
        if self.params.method == "NPE-A": 
            x_0 = self.params.x_0
            x_0_expanded = x_0.unsqueeze(0).expand(*size, *x_0.shape)
            return self.sampler.sample(x_0=x_0_expanded, n_steps=self.params.n_steps)

        elif self.params.method == 'NPE-B':
            total = 1
            for dim in size:
                total *= dim
            assert self.params.method_params is not None 
            weight = self.params.method_params.get("weight", None)
            assert weight is not None

            num_proposals   = int(weight * total)
            num_priors      = total - num_proposals

            x_0 = self.params.x_0
            x_0_expanded = x_0.unsqueeze(0).expand(num_proposals, *x_0.shape)
            proposals = self.sampler.sample(x_0=x_0_expanded, n_steps=self.params.n_steps)
            
            assert self.params.task is not None
            priors = self.params.task.prior.sample((num_priors, *x_0.shape[:-1], ))
            samples = torch.cat([proposals, priors], dim=0)
            samples = samples[torch.randperm(samples.shape[0])]
            samples = samples.view(*size, *x_0.shape[:-1], -1)
            return samples
        
        elif self.params.method == 'NPE-C':
            raise NotImplementedError(f"Method {self.params.method} is not implemented") # type: ignore


        elif self.params.method == 'fast-Truncated':
            assert self.params.method_params is not None
            scale = self.params.method_params.get("scale", None)
            assert scale is not None

            x_0 = self.params.x_0
            x_0_expanded = x_0.unsqueeze(0).expand(*size, *x_0.shape)
            return self.sampler.sample(x_0=x_0_expanded, 
                                       n_steps=self.params.n_steps, 
                                       scale=scale)
    

        elif self.params.method == "Truncated":
            x_0 = self.params.x_0
            x_0_expanded = x_0.unsqueeze(0).expand(*size, *x_0.shape)
            return self.sampler.sample(x_0=x_0_expanded, 
                                       n_steps=self.params.n_steps)


        elif self.params.method == "shitty-Truncated":
            # quantile = kwargs.get("quantile", None)
            assert self.params.method_params is not None
            quantile = self.params.method_params.get("quantile", None)
            assert quantile is not None
            
            self.buffer = []          # список чистых тензоров (возможно, разной длины)
            self.buffer_len = 0       # общее количество чистых образцов в буфере
            self.sample_shape = self.params.task.theta_dim    # форма одного образца, например (data_dim,) или (H, W) для изображений

            def _add_clean(data: torch.Tensor) -> None:
                """Отфильтровать и добавить чистые образцы в буфер."""
                if data.numel() == 0:
                    return
                logp = self.log_prob(data)
                mask = torch.where(logp <= torch.quantile(logp, quantile))

                clean = data[mask]
                # self.logger.debug(f"_add_clean data {data.shape}")
                # self.logger.debug(f"_add_clean mask {mask.shape}")
                # self.logger.debug(f"_add_clean clean {clean.shape}")
                if clean.numel() == 0:
                    return

                # Определяем форму одного образца при первом поступлении чистых данных
                # if self.sample_shape is None:
                #     self.sample_shape = clean.shape[-1]   # первая размерность – число образцов

                # Перемещаем на нужное устройство (если указано) и сохраняем в буфере
                if self.params.x_0.device is not None:
                    clean = clean.to(self.params.x_0.device)
                self.buffer.append(clean)
                self.buffer_len += clean.size(0)

            def _pop_from_buffer(n: int) -> torch.Tensor:
                """Извлечь первые n образцов из буфера (форма [n, *sample_shape])."""
                # Объединяем весь буфер в один тензор
                all_clean = torch.cat(self.buffer, dim=0)
                result = all_clean[:n]
                remaining = all_clean[n:]

                # Обновляем буфер
                if remaining.numel() > 0:
                    self.buffer = [remaining]
                    self.buffer_len = remaining.size(0)
                else:
                    self.buffer = []
                    self.buffer_len = 0
                return result
            
            def _sample(shape) -> torch.Tensor:
                """
                Вернуть тензор чистых образцов формы (*shape, *sample_shape).

                Аргументы:
                    shape: желаемая форма (d1, d2, ..., dn) для многомерной решётки образцов.
                        Например, (32, 32) для 32x32 сетки.
                Возвращает:
                    Тензор формы (*shape, *sample_shape).
                """
                x_0 = self.params.x_0
                x_0_expanded = x_0.unsqueeze(0).expand(*size, *x_0.shape)

                total = 1
                for dim in shape:
                    total *= dim

                # Если в буфере уже достаточно, выдаём напрямую
                # WARNING: проблемы с разменостью
                if self.buffer_len >= total:
                    flat = _pop_from_buffer(total)
                    # self.logger.debug("first")
                    return flat.view(*shape, self.sample_shape)   # flat.shape[1:] = sample_shape

                # Иначе набираем необходимое количество, возможно, досэмплируя
                result_parts = []
                if self.buffer_len > 0:
                    result_parts.append(_pop_from_buffer(self.buffer_len))
                    total -= self.buffer_len

                    
                BRAKE_NUM = 5*total
                i = 0
                left = total
                while left > 0:
                    raw = self.sampler.sample(x_0=x_0_expanded, n_steps=self.params.n_steps, device=x_0.device)
                    raw_shape = raw.shape
                    _add_clean(raw)
                    take = min(left, self.buffer_len)
                    i += take
                    if take > 0:
                        result_parts.append(_pop_from_buffer(take))
                        left -= take
                    if (left // 100 * 100) % (total // 10) == 0:
                        self.params.task.logger.info(f"{left} left to sample")
                    if i > BRAKE_NUM:
                        self.params.task.logger.info(f"Number of simulations reached limit")
                        return torch.tensor([None])

                # Склеиваем все части
                # self.params.task.logger.debug(f"{len(result_parts)}, {result_parts[0].shape}")
                flat_result = torch.cat(result_parts, dim=0) if len(result_parts) > 1 else result_parts[0]
                # Придаём нужную многомерную форму
                # self.params.task.logger.debug(f"flat_result {flat_result}, want shape {(*shape, self.sample_shape)}")
                # self.params.task.logger.debug(f"flat_result return {flat_result.view(*shape, self.sample_shape)}")

                # TODO: плохо, что мы извлекаем нужную нам разменость через raw.shape
                return flat_result.view(*raw_shape)
            
            return _sample(size)

        else:
            raise NotImplementedError(f"Method {self.params.method} is not implemented") # type: ignore


    
    def log_prob(self, value, **kwargs):
            """
            value: (*batch, d)
            returns: (*batch,)
            """
            device = value.device
            batch_shape = value.shape[:-1]
            d = value.shape[-1]
            
            x_0 = self.params.x_0
            # WARNING: проблемы с размерностями
            x_0_expanded = x_0.expand(*batch_shape, self.params.task.data_dim).to(device)
                
            # --- initial state (at t = 0) ---
            theta0 = value

            logp = torch.zeros(batch_shape, device=device)

            # Hutchinson noise
            eps = torch.randn_like(theta0)

            # Ensure velocity model is on correct device and in eval mode
            self.flow_model.velocity_model.eval()
            self.flow_model.velocity_model.to(device)

            # --- ODE function ---
            def ode_func(t, state):
                theta, logp = state

                with torch.enable_grad():
                    theta = theta.requires_grad_(True)
                    t_expanded = t.unsqueeze(0).expand(*batch_shape, 1)
                    u = self.flow_model.velocity_model(t=t_expanded, theta=theta, x=x_0_expanded)

                    jvp = torch.autograd.grad(
                        (u * eps).sum(),
                        theta,
                        create_graph=False,
                        retain_graph=False,
                    )[0]

                div = (jvp * eps).sum(dim=-1)

                dtheta = u
                dlogp = -div

                return dtheta, dlogp

            # --- integrate BACKWARD: t=1 -> t=0 ---
            # WARNING: параметры солвера - константы
            with torch.no_grad():
                _, logp_correction = odeint(
                    ode_func,
                    y0=(theta0, logp),
                    t=torch.linspace(0, 1, steps=8, device=device)
                )

            # --- init dist log prob ---
            if self.params.task is not None:
                # WARNING могут быть проблемы с устройствами
                base_logp = self.flow_model.init_dist.log_prob(theta0).to(device)
            else:
                raise NotImplementedError("Base distribution not defined")

            # --- итог ---
            logp = base_logp + logp_correction[-1]

            return logp