import torch
from abc import ABC, abstractmethod
from torch.distributions import MultivariateNormal, Normal as UnivariateNormal, Chi2



class Distribution(ABC):
    def __init__(self):
        self.params = None

    @abstractmethod
    def sample(self, size, **kwargs) -> torch.Tensor:
        pass

    def sample_like(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return tensor
    
    def log_prob(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.Tensor([-torch.inf])
    
    def to(self, device: torch.device) -> None:
        pass

class Uniform(Distribution):
    def __init__(self, dim=None, low=0, high=1):
        self.low = low
        self.high = high
        self.dim = dim
        self.dist = torch.distributions.Uniform(low, high) 

    def sample(self, size, **kwargs) -> torch.Tensor:
        if self.dim is not None:
            size = (*size, self.dim)
        return self.dist.sample(size)
    
    def sample_like(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.low + (self.high - self.low) * torch.rand_like(tensor)

    def log_prob(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.dist.log_prob(value)


# WARNING AI-SLOP
import torch
from torch.distributions import MultivariateNormal, Normal as UnivariateNormal, Chi2

class Normal:
    def __init__(self, dim=1, device='cpu'):
        self.dim = dim
        self.device = device
        self.dist = MultivariateNormal(
            torch.zeros(dim, device=device),
            covariance_matrix=torch.eye(dim, device=device)
        )
        self._dist_1d = UnivariateNormal(0.0, 1.0) if dim == 1 else None
        self._chi2 = Chi2(df=dim) if dim > 1 else None
        # кэш для пороговых значений R²_max по quantile
        self._R2_max_cache = {}

    def _chi2_ppf(self, p, tol=1e-5):
        """Обратная функция распределения χ² через бинарный поиск по cdf."""
        # p – скалярный float
        left, right = 0.0, 1.0
        # ищем верхнюю границу, пока cdf не станет >= p
        while self._chi2.cdf(torch.tensor(right)) < p:
            right *= 2.0
        for _ in range(40):  # 40 итераций хватит для точности 2^-40
            mid = (left + right) / 2
            if self._chi2.cdf(torch.tensor(mid)) < p:
                left = mid
            else:
                right = mid
        return (left + right) / 2

    def _get_R2_max(self, quantile):
        """Возвращает R²_max для заданного quantile (с кэшированием)."""
        if quantile not in self._R2_max_cache:
            with torch.no_grad():
                self._R2_max_cache[quantile] = self._chi2_ppf(quantile)
        return self._R2_max_cache[quantile]

    def sample(self, size, quantile=None, **kwargs):
        """
        size : int или tuple (размер батча без последнего измерения признаков).
        quantile : float 0<q<1 – доля центральной вероятностной массы.
                   Если None, возвращается обычная выборка.
        """
        if quantile is None:
            return self.dist.sample(size)

        if not (0 < quantile < 1):
            raise ValueError("quantile должен быть между 0 и 1")

        # Определим общее количество точек N и финальную форму
        if isinstance(size, int):
            N = size
            out_shape = (size, self.dim)
        else:
            N = int(torch.prod(torch.tensor(size)))
            out_shape = tuple(size) + (self.dim,)

        # ---- Одномерный случай: симметричный интервал ----
        if self.dim == 1:
            # здесь icdf у Normal обычно есть, но на всякий случай используем erf
            a = self._dist_1d.icdf(torch.tensor((1 + quantile) / 2, device=self.device))
            low_cdf = self._dist_1d.cdf(-a)
            high_cdf = self._dist_1d.cdf(a)
            u = low_cdf + (high_cdf - low_cdf) * torch.rand(N, device=self.device)
            samples = self._dist_1d.icdf(u)
            return samples.reshape(out_shape)

        # ---- Многомерный случай (dim >= 2) ----
        R2_max = self._get_R2_max(quantile)
        samples_list = []
        remaining = N
        while remaining > 0:
            # генерируем чуть больше, чем remaining (с запасом)
            batch_size = max(remaining, 10)
            # сэмплируем направления (нормальный вектор)
            dirs = torch.randn(batch_size, self.dim, device=self.device)
            dirs = dirs / dirs.norm(dim=1, keepdim=True)
            # сэмплируем радиусы (через χ²)
            chi2_samples = self._chi2.sample((batch_size,))
            # отбрасываем слишком большие радиусы
            mask = chi2_samples <= R2_max
            accepted_chi2 = chi2_samples[mask][:remaining]
            accepted_dirs = dirs[mask][:remaining]
            if len(accepted_chi2) > 0:
                r = torch.sqrt(accepted_chi2)
                points = r.unsqueeze(1) * accepted_dirs
                samples_list.append(points)
                remaining -= points.shape[0]
        samples = torch.cat(samples_list, dim=0)
        return samples.reshape(out_shape)

    def sample_like(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        assert tensor.shape[-1] == self.dim
        return self.dist.sample(tensor.shape[:-1])

    def log_prob(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.dist.log_prob(value.to(self.device))

    def to(self, device: torch.device):
        self.device = device
        self.dist = MultivariateNormal(
            torch.zeros(self.dim, device=device),
            covariance_matrix=torch.eye(self.dim, device=device)
        )
        if self.dim == 1:
            self._dist_1d = UnivariateNormal(0.0, 1.0)
        if self.dim > 1:
            self._chi2 = Chi2(df=self.dim)
        self._R2_max_cache.clear()


# class Normal(Distribution):
#     def __init__(self, dim=1, device='cpu'):
#         self.dim = dim
#         self.device = device
#         self.dist = torch.distributions.MultivariateNormal(torch.zeros(dim, device=device), covariance_matrix=torch.eye(dim, device=device))
    
#     def sample(self, size, **kwargs) -> torch.Tensor:
#         return self.dist.sample(size)
    
#     def sample_like(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
#         assert tensor.shape[-1] == self.dim
#         return self.dist.sample(sample_shape=tensor.shape[:-1])

#     def log_prob(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
#         return self.dist.log_prob(value.to(self.device), **kwargs)



