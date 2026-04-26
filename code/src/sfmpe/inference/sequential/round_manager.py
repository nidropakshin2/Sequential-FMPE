import torch

from sfmpe.data.simulation_store import SimulationStore
from sfmpe.data.round_dataset import RoundDataset
from sfmpe.utils.logger import Logger, get_default_logger


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
    ):

        self.task = task
        self.estimator = estimator
        self.proposal = task.prior
        self.proposal_params = proposal_params
        self.validator = validator
        self.device = device
        
        
        if logger is None:
            self.logger = self.task.logger
            self.estimator.logger = self.logger
        else:
            self.logger = logger

        self.store = SimulationStore()
        self.losses = []
        
        # Log initialization
        self.logger.info(f"RoundManager initialized with device: {device}")
        self.logger.info(f"Task: {task.__class__.__name__}")
        self.logger.info(f"Estimator: {estimator.__class__.__name__}")


    def run_round(self, round_id, sims_per_round):

        # Log round start
        self.logger.info(f"Starting round {round_id} with {sims_per_round} simulations")
        
        # sample parameters
        self.logger.debug(f"Proposal distribution: {self.proposal}")

        if round_id == 0:
            # theta = self.proposal.sample((sims_per_round, *self.proposal_params.x_0.shape[:-1]), device=self.device)
            theta = self.clean_sample((sims_per_round, *self.proposal_params.x_0.shape[:-1]))
        else:
            # theta = self.proposal.sample((sims_per_round, ), device=self.device)
            theta = self.clean_sample((sims_per_round, ))

        self.logger.debug(f"x_0 shape {self.proposal_params.x_0.shape[:-1]}")
        
        self.logger.debug(f"Sampled {sims_per_round} parameters with shape {theta.shape}")

        # simulate data
        # TODO: сделать очистку от nan
        x = self.task.simulator.simulate(theta)
        self.logger.debug(f"Simulated data with shape: {x.shape}")

        # summary statistics
        features = self.task.summary(x)
        self.logger.debug(f"Computed summary statistics with shape: {features.shape}")

        # store simulations
        self.store.add(theta, features, round_id)
        self.logger.info(f"Round {round_id} completed - stored {sims_per_round} simulations")

    def train_estimator(self, rounds=None, **train_kwargs):

        dataset = RoundDataset(self.store, rounds)
        
        num_samples = len(dataset)
        self.logger.info(f"Training estimator on {num_samples} samples from rounds: {rounds}")
        
        # Extract training parameters for logging
        epochs = train_kwargs.get('epochs', 'default')
        self.logger.info(f"Training parameters: epochs={epochs}")
        
        # Train the estimator
        loss_stats = self.estimator.train(dataset, **train_kwargs)
        
        self.logger.info(f"Estimator training completed")
        return loss_stats
    
    def build_posterior(self):
        return self.estimator.build_posterior(self.proposal_params)
    
    def update_proposal(self, posterior):

        self.logger.debug(f"Updating proposal distribution")
        self.logger.debug(f"Old proposal: {self.proposal}")
        self.proposal = posterior
        self.logger.debug(f"New proposal: {posterior}")
    

    def run_sequential(
        self,
        num_rounds,
        sims_per_round,
        **train_kwargs,
    ):
        
        self.logger.info(f"Starting sequential training with {num_rounds} rounds")
        self.logger.info(f"Simulations per round: {sims_per_round}")
        self.logger.info(f"Training kwargs: {train_kwargs}")

        for r in range(num_rounds):

            self.logger.info(f"--- Round {r}/{num_rounds} ---")

            self.run_round(r, sims_per_round)

            self.losses += self.train_estimator([r], **train_kwargs)
            # if torch.isnan(torch.tensor(self.losses[-1])):
            #     self.logger.error(f"Loss is nan, stopping execution...")
            #     return -1
            posterior = self.build_posterior()
            self.logger.debug(f"Built posterior: {posterior}")

            # self.validator.validate(r, posterior)

            self.update_proposal(posterior)
            
            # Log progress
            progress = (r + 1) / num_rounds * 100
            self.logger.info(f"Round {r} completed - Progress: {progress:.1f}%")

        self.logger.info(f"Sequential training completed successfully")
        self.logger.info(f"Total simulations: {num_rounds * sims_per_round}")


    def clean_sample(self, shape):
        if self.task.check_support is None:
            return self.proposal.sample(shape, device=self.device)

        self.buffer = []          # список чистых тензоров (возможно, разной длины)
        self.buffer_len = 0       # общее количество чистых образцов в буфере
        self.sample_shape = self.task.theta_dim    # форма одного образца, например (data_dim,) или (H, W) для изображений

        def _add_clean(data: torch.Tensor) -> None:
            """Отфильтровать и добавить чистые образцы в буфер."""
            if data.numel() == 0:
                return
            mask = self.task.check_support(data)
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
                

            while total > 0:
                raw = self.proposal.sample(shape, device=self.device)
                raw_shape = raw.shape
                _add_clean(raw)
                take = min(total, self.buffer_len)
                # self.logger.debug(f"third, {raw.shape}")
                
                if take > 0:
                    result_parts.append(_pop_from_buffer(take))
                    total -= take
                    # self.logger.debug("fourth")
                # self.logger.info(f"Generated clean samples {len(result_parts)} in total")
                

            # Склеиваем все части
            self.logger.debug(f"{len(result_parts)}, {result_parts[0].shape}")
            flat_result = torch.cat(result_parts, dim=0) if len(result_parts) > 1 else result_parts[0]
            # Придаём нужную многомерную форму
            # self.logger.debug(f"flat_result {flat_result}, want shape {(*shape, self.sample_shape)}")
            # self.logger.debug(f"flat_result return {flat_result.view(*shape, self.sample_shape)}")

            # TODO: плохо, что мы извлекаем нужную нам разменость через raw.shape
            return flat_result.view(*raw_shape)
        
        return sample(shape)
