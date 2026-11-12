import torch

from sfmpe.data.simulation_store import SimulationStore
from sfmpe.data.round_dataset import RoundDataset
from sfmpe.utils.logger import Logger, get_default_logger
from sfmpe.utils.plotting import Validator


class RoundManager:
    """
    Controls sequential SBI rounds.
    """

    def __init__(
        self,
        task,
        estimator,
        proposal_params,
        validator=None,
        device="cpu",
        logger=None,
        storage_dir="."
    ):

        self.task = task
        self.estimator = estimator
        self.proposal = task.prior
        self.proposal_params = proposal_params
        self.K = self.proposal_params.theta_0.shape[-2]
        self.device = device
        
        
        if logger is None:
            self.logger = self.task.logger
            self.estimator.logger = self.logger
        else:
            self.logger = logger

        self.store = SimulationStore(storage_dir)
        self.losses = []

        self.validator = validator
        self.truncated_bounds = None
        
        # Log initialization
        self.logger.info(f"Task: {task.__class__.__name__}")
        self.logger.info(f"Estimator: {estimator.__class__.__name__}")

    def load_model(self, path):

        self.estimator.load(path)
        
        posterior = self.build_posterior()
        self.logger.debug(f"Built posterior: {posterior}")
        self.update_proposal(posterior)

    def run_round_experimental(self, round_id, samples_per_round, sims_per_sample, clean_sampling):
    
        # Log round start
        self.logger.info(f"Starting round {round_id} with {samples_per_round} simulations")
        
        # sample parameters
        # self.logger.debug(f"Proposal distribution: {self.proposal}")

        if self.proposal == self.task.prior:
            theta = self.proposal.sample((samples_per_round, self.K), device=self.device)
            # theta = self.clean_sample((samples_per_round, *self.proposal_params.x_0.shape[:-1]), clean_sampling=clean_sampling).to(self.device)
            # theta = self.clean_sample_experimental((samples_per_round, self.K), clean_sampling=clean_sampling).to(self.device)
        else:
            # Truncated means sampling from the prior conditional on the
            # region inferred from the previous posterior, not filtering
            # posterior draws by their flow density.
            effective_clean_sampling = (
                True
                if self.proposal_params.method == "Truncated"
                else clean_sampling
            )
            theta = self.clean_sample_experimental(
                (samples_per_round,),
                clean_sampling=effective_clean_sampling,
            ).to(self.device)
            
        # self.logger.debug(f"Sampled {samples_per_round} parameters with shape {theta.shape}")

        # simulate data
        hasnan = torch.isnan(theta)
        while len(hasnan.shape) > 1:
            hasnan = hasnan.any(dim=-1)
        if hasnan.any():
            self.logger.info(f"Stopping round {round_id} because theta has nan")
            return -1

        """
        theta: [N, K, theta_dim] -> [N*sims_per_sample, K, theta_dim]
        """
        theta = theta.unsqueeze(0).expand(sims_per_sample, *theta.shape).flatten(0, 1)
        x = self.task.simulator.simulate(theta).to(self.device)
        # self.logger.debug(f"Simulated data with shape: {x.shape}")

        # summary statistics
        self.task.summary.to(self.device)
        features = self.task.summary(x).to(self.device)
        # self.logger.debug(f"Computed summary statistics with shape: {features.shape}")

        # store simulations
        self.store.add(theta, features, round_id)

        self.logger.info(f"Round {round_id} completed - stored {samples_per_round} simulations")

    def run_round(self, round_id, sims_per_round, clean_sampling):

        # Log round start
        self.logger.info(f"Starting round {round_id} with {sims_per_round} simulations")
        
        # sample parameters
        # self.logger.debug(f"Proposal distribution: {self.proposal}")

        if self.proposal == self.task.prior:
            # theta = self.proposal.sample((sims_per_round, *self.proposal_params.x_0.shape[:-1]), device=self.device)
            theta = self.clean_sample((sims_per_round, *self.proposal_params.x_0.shape[:-1]), clean_sampling=clean_sampling)
        else:
            # theta = self.proposal.sample((sims_per_round, ), device=self.device)
            theta = self.clean_sample((sims_per_round, ), clean_sampling=clean_sampling)

        # self.logger.debug(f"x_0 shape {self.proposal_params.x_0.shape[:-1]}")
        
        # self.logger.debug(f"Sampled {sims_per_round} parameters with shape {theta.shape}")

        # simulate data
        hasnan = torch.isnan(theta)
        while len(hasnan.shape) > 1:
            hasnan = hasnan.any(dim=-1)
        if hasnan.any():
            self.logger.info(f"Stopping round {round_id} because theta has nan")
            return -1
        x = self.task.simulator.simulate(theta)
        # self.logger.debug(f"Simulated data with shape: {x.shape}")

        # summary statistics
        self.task.summary.to(self.device)
        features = self.task.summary(x.to(self.device))
        # self.logger.debug(f"Computed summary statistics with shape: {features.shape}")

        # store simulations
        self.store.add(theta, features, round_id)

        self.logger.info(f"Round {round_id} completed - stored {sims_per_round} simulations")

    def train_estimator(self, rounds=None, **train_kwargs):

        dataset = RoundDataset(self.store, rounds)
        
        num_samples = len(dataset)
        self.logger.info(f"Training estimator on {num_samples} samples from rounds: {rounds}")
        
        # Extract training parameters for logging
        epochs = train_kwargs.get('epochs', 'default')
        # Train the estimator
        # loss_stats = self.estimator.train(dataset, **train_kwargs)
        loss_stats = self.estimator.train_experimental(dataset, **train_kwargs)
        
        self.logger.info(f"Estimator training completed")
        return loss_stats
    
    def build_posterior(self, update_x=True):
        if update_x:
            new_x_0 = self.task.simulate(self.proposal_params.theta_0).to(self.device)
            new_x_0 = self.task.summarize(new_x_0).to(self.device)
            self.logger.debug(f"old x_0: {self.proposal_params.x_0},\nnew x_0: {new_x_0}")
            self.proposal_params.x_0 = new_x_0
        return self.estimator.build_posterior(self.proposal_params)
    
    def update_proposal(self, posterior):

        self.logger.debug(f"Updating proposal distribution")
        # self.logger.debug(f"Old proposal: {self.proposal}")
        self.proposal = posterior
        if self.proposal_params.method == "Truncated":
            self.truncated_bounds = self._build_truncated_bounds(posterior)
        # self.logger.debug(f"New proposal: {posterior}")

    def _prior_support_mask(self, samples: torch.Tensor) -> torch.Tensor:
        """Return one prior-support validity flag for each parameter vector."""
        try:
            support = self.task.prior.support
        except AttributeError:
            return torch.ones(
                samples.shape[:-1],
                dtype=torch.bool,
                device=samples.device,
            )

        mask = support.check(samples)
        if mask.shape == samples.shape:
            mask = mask.all(dim=-1)
        if mask.shape != samples.shape[:-1]:
            raise ValueError(
                "Prior support must return one boolean per parameter vector."
            )
        return mask

    def _build_truncated_bounds(self, posterior) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.proposal_params.method_params or {}
        mass = float(params.get("quantile", 0.9))
        if not 0.0 < mass < 1.0:
            raise ValueError(
                "Truncated proposal quantile must be a central mass in (0, 1)."
            )
        num_samples = int(params.get("region_samples", 2048))
        if num_samples < 2:
            raise ValueError("Truncated region_samples must be at least 2.")
        padding = float(params.get("padding", 0.1))
        if padding < 0:
            raise ValueError("Truncated proposal padding must be non-negative.")

        samples = posterior.sample((num_samples,), device=self.device)
        if samples.ndim != 3 or samples.shape[1:] != (
            self.K,
            self.task.theta_dim,
        ):
            raise ValueError(
                "Posterior samples for Truncated proposals must have shape "
                f"(samples, {self.K}, {self.task.theta_dim}); "
                f"got {tuple(samples.shape)}."
            )
        valid = self._prior_support_mask(samples) & torch.isfinite(samples).all(-1)
        lower_quantile = (1.0 - mass) / 2.0
        upper_quantile = 1.0 - lower_quantile
        lower = torch.empty(
            (self.K, self.task.theta_dim),
            device=samples.device,
            dtype=samples.dtype,
        )
        upper = torch.empty_like(lower)
        for index in range(self.K):
            valid_samples = samples[valid[:, index], index]
            if valid_samples.shape[0] < 2:
                raise RuntimeError(
                    "Cannot construct a Truncated prior region: fewer than "
                    f"two posterior samples are inside prior support for "
                    f"condition {index}."
                )
            lower[index] = torch.quantile(
                valid_samples, lower_quantile, dim=0
            )
            upper[index] = torch.quantile(
                valid_samples, upper_quantile, dim=0
            )

        width = upper - lower
        padding_amount = padding * width
        lower = lower - padding_amount
        upper = upper + padding_amount
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
            raise RuntimeError("Truncated prior region has non-finite bounds.")
        return lower, upper

    def _sample_from_truncated_prior(
        self, shape: tuple[int, ...]
    ) -> torch.Tensor:
        if self.truncated_bounds is None:
            raise RuntimeError(
                "Truncated prior sampling requires bounds from a previous "
                "posterior; fit the first round and call update_proposal first."
            )

        params = self.proposal_params.method_params or {}
        max_draws = int(params.get("max_prior_draws", 10_000_000))
        if max_draws < 1:
            raise ValueError("Truncated max_prior_draws must be positive.")
        lower, upper = self.truncated_bounds
        total = 1
        for dimension in shape:
            total *= dimension
        result = torch.empty(
            (total, self.K, self.task.theta_dim),
            device=self.device,
            dtype=lower.dtype,
        )
        filled = [0] * self.K
        attempted = 0
        while min(filled) < total and attempted < max_draws:
            remaining = total - min(filled)
            batch_size = min(max(2 * remaining, 64), max_draws - attempted)
            candidates = self.task.prior.sample((batch_size, self.K)).to(
                self.device
            )
            if candidates.shape != (
                batch_size,
                self.K,
                self.task.theta_dim,
            ):
                raise ValueError(
                    "Prior samples for Truncated proposals must have shape "
                    f"({batch_size}, {self.K}, {self.task.theta_dim}); "
                    f"got {tuple(candidates.shape)}."
                )
            attempted += batch_size
            accepted_mask = self._prior_support_mask(candidates)
            accepted_mask &= (
                (candidates >= lower.unsqueeze(0))
                & (candidates <= upper.unsqueeze(0))
            ).all(dim=-1)
            for index in range(self.K):
                condition_remaining = total - filled[index]
                if condition_remaining == 0:
                    continue
                accepted = candidates[accepted_mask[:, index], index]
                take = min(condition_remaining, accepted.shape[0])
                if take:
                    result[filled[index] : filled[index] + take, index] = (
                        accepted[:take]
                    )
                    filled[index] += take

        if min(filled) < total:
            completed = min(filled)
            raise RuntimeError(
                "Could not draw the requested Truncated prior batch: "
                f"accepted {completed}/{total} samples per condition at "
                f"minimum after {attempted} prior "
                "draws. Increase max_prior_draws or widen the region."
            )
        return result.reshape(*shape, self.K, self.task.theta_dim)
    

    def run_sequential(
        self,
        num_rounds,
        sims_per_round,
        clean_sampling=False,
        upd_x=False,
        sims_per_sample=1,
        device='cpu',
        **train_kwargs,
    ):

        self.device = device

        self.logger.info(f"Starting sequential training with {num_rounds} rounds")
        self.logger.info(f"Simulations per round: {sims_per_round}")
        self.logger.info(f"Training kwargs: {train_kwargs}")

        for r in range(num_rounds):

            self.logger.info(f"--- Round {r}/{num_rounds} ---")

            # out = self.run_round(r, sims_per_round, clean_sampling)
            out = self.run_round_experimental(r, sims_per_round, sims_per_sample, clean_sampling=clean_sampling)
            # self.validator.plot_comparison()
            # if r == 1:
            #     self.task.summary.eval() 
            
            if out == -1:
                self.logger.info("Stopping sequential due to posterior extreme shrinkage...")
                return -1

            self.losses += self.train_estimator([r], **train_kwargs)
            # if torch.isnan(torch.tensor(self.losses[-1])):
            #     self.logger.error(f"Loss is nan, stopping execution...")
            #     return -1
            if r == 0 and upd_x:
                update_x = True
            else:
                update_x = False
            posterior = self.build_posterior(update_x=update_x)
            # self.logger.debug(f"Built posterior: {posterior}")

            self.update_proposal(posterior)

            # torch.cuda.empty_cache()
            
            # Log progress
            progress = (r + 1) / num_rounds * 100
            self.logger.info(f"Round {r} completed - Progress: {progress:.1f}%")

        self.logger.info(f"Sequential training completed successfully")
        self.logger.info(f"Total simulations: {num_rounds * sims_per_round}")


    def clean_sample(self, shape, **kwargs):
        if self.proposal_params.method == "Truncated":
            if self.proposal == self.task.prior:
                return self.task.prior.sample(shape).to(self.device)
            if not shape or shape[-1] != self.K:
                raise ValueError(
                    "Truncated prior sampling expects the condition count "
                    f"{self.K} as the last sample dimension."
                )
            return self._sample_from_truncated_prior(tuple(shape[:-1]))

        if not kwargs.get("clean_sampling", False):
            self.logger.debug("NO clean sampling")
            return self.proposal.sample(shape, device=self.device)


        # TODO разобраться с хранением датасетов и вычислением log_prob на GPU

        self.buffer = []          # список чистых тензоров (возможно, разной длины)
        self.buffer_len = 0       # общее количество чистых образцов в буфере
        self.sample_shape = self.task.theta_dim    # форма одного образца, например (data_dim,) или (H, W) для изображений

        def _add_clean(data: torch.Tensor) -> None:
            """Отфильтровать и добавить чистые образцы в буфер."""
            if data.numel() == 0:
                return
            try:
                support_mask = self.task.prior.support.check(data)
            except AttributeError:
                support_mask = torch.ones_like(data, dtype=torch.bool)
                support_mask = support_mask.all(dim=-1)

            clean = data[support_mask]
            # self.logger.debug(f"_add_clean data {data.shape}")
            # self.logger.debug(f"_add_clean mask {mask.shape}")
            # self.logger.debug(f"_add_clean clean {clean.shape}")
            if clean.numel() == 0:
                return

            # Определяем форму одного образца при первом поступлении чистых данных
            # if self.sample_shape is None:
            #     self.sample_shape = clean.shape[-1]   # первая размерность – число образцов

            # Перемещаем на нужное устройство (если указано) и сохраняем в буфере
            if self.device is not None:
                clean = clean.to(self.device)
            self.buffer.append(clean)
            self.buffer_len += clean.size(0)

        def _pop_from_buffer(n: int) -> torch.Tensor:
            """Извлечь первые n образцов из буфера (форма [n, *sample_shape])."""
            # if n == 0:
            #     return torch.empty(0, *self.sample_shape, device=self.device)

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
            # self.logger.debug(f"_pop_from_buffer buffer {len(self.buffer)}")    
            # self.logger.debug(f"_pop_from_buffer result {result.shape}")
            return result

        def sample(shape) -> torch.Tensor:
            """
            Вернуть тензор чистых образцов формы (*shape, *sample_shape).

            Аргументы:
                shape: желаемая форма (d1, d2, ..., dn) для многомерной решётки образцов.
                    Например, (32, 32) для 32x32 сетки.
            Возвращает:
                Тензор формы (*shape, *sample_shape).
            """
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
                # self.logger.debug("second")
                
            BRAKE_NUM = 5*total
            i = 0
            left = total
            while left > 0:
                raw = self.proposal.sample(shape, device=self.device)
                raw_shape = raw.shape
                _add_clean(raw)
                take = min(left, self.buffer_len)
                # self.logger.debug(f"third, {raw.shape}")
                i += take
                if take > 0:
                    result_parts.append(_pop_from_buffer(take))
                    left -= take
                if (left // 100 * 100) % (total // 10) == 0:
                    self.logger.info(f"{left} left to sample")
                if i > BRAKE_NUM:
                    self.logger.info(f"Number of simulations reached limit")
                    return torch.tensor([None])

            # Склеиваем все части
            self.logger.debug(f"{len(result_parts)}, {result_parts[0].shape}")
            flat_result = torch.cat(result_parts, dim=0) if len(result_parts) > 1 else result_parts[0]
            # Придаём нужную многомерную форму
            # self.logger.debug(f"flat_result {flat_result}, want shape {(*shape, self.sample_shape)}")
            # self.logger.debug(f"flat_result return {flat_result.view(*shape, self.sample_shape)}")
            self.logger.debug(f"shape bug {shape, raw_shape, flat_result.size()}")
            # TODO: плохо, что мы извлекаем нужную нам разменость через raw.shape
            return flat_result.view(*raw_shape)
        
        return sample(shape)


    def clean_sample_experimental(self, shape, **kwargs):
        if self.proposal_params.method == "Truncated":
            if self.proposal == self.task.prior:
                return self.task.prior.sample(
                    (*shape, self.K)
                ).to(self.device)
            return self._sample_from_truncated_prior(tuple(shape))

        if not kwargs.get("clean_sampling", False):
            self.logger.debug("NO clean sampling")
            return self.proposal.sample(shape, device=self.device)

        """
            buffer:
                [N, K, D]
            Итоговый буфер

            remaining:
                [N, K]
            True означает, что позиция еще не заполнена

        """
        self.sample_shape = self.task.theta_dim
        
        self.buffer = torch.zeros(*shape, self.K, self.sample_shape, device=self.device)
        self.remaining = torch.ones(*shape, self.K, dtype=torch.bool, device=self.device)


        def _merge_samples(new_samples, valid_mask):
            """
            Заполняет еще пустые позиции в buffer.

            Args:
                new_samples:
                    [N, K, D]
                    Новая партия samples.

                valid_mask:
                    [N, K]
                    True означает, что новый sample валиден.

            Returns:
                new_remaining:
                    [N, K]
                    True означает, что позиция все еще не заполнена.
            """
            # fill_mask = self.remaining & valid_mask

            
            # self.buffer[fill_mask] = new_samples[fill_mask]

            # new_remaining = self.remaining & ~valid_mask

            # return new_remaining
            
            remaining_idx = self.remaining.nonzero(as_tuple=True)[0]
            valid_idx = valid_mask.nonzero(as_tuple=True)[0]

            n_fill = min(
                remaining_idx.numel(),
                valid_idx.numel()
            )

            if n_fill == 0:
                return

            remaining_idx = remaining_idx[:n_fill]
            valid_idx = valid_idx[:n_fill]

            self.buffer[remaining_idx] = new_samples[valid_idx]
            self.remaining[remaining_idx] = False

        def _mask(data: torch.Tensor, quantile_value=None):
            if data.numel() == 0:
                return None, None
            
            try:
                support_mask = self.task.prior.support.check(data)
            except AttributeError:
                support_mask = torch.ones_like(data, dtype=torch.bool)
                support_mask = support_mask.all(dim=-1)

            # self.logger.debug(f"mask1.dtype = {support_mask.dtype}, check support is None: {self.task.check_support is None}")
            # self.logger.debug(f"data {data.shape}, mask1: {support_mask.shape}")

            return support_mask, quantile_value

        samples = self.proposal.sample(shape, device=self.device)
        
        valid_mask, quantile_value = _mask(samples)

        _merge_samples(samples, valid_mask)

        # self.logger.debug(f"shapes self.remaining {self.remaining.shape}, valid_mask {valid_mask.shape}")
                                        
        self.logger.debug(
            f"remaining: {self.remaining.sum().item()}, "
        )
        

        while self.remaining.any():

            samples = self.proposal.sample(shape, device=self.device)

            valid_mask, _ = _mask(samples, quantile_value)

            _merge_samples(samples, valid_mask)

            # self.logger.debug(f"shapes self.remaining {self.remaining.shape}, valid_mask {valid_mask.shape}")
                                    
            self.logger.debug(
                f"remaining: {self.remaining.sum().item()}, "
            )


        return self.buffer