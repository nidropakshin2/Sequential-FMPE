import torch
from torch.utils.data import Dataset


class RoundDataset(Dataset):
    """
    PyTorch dataset built from SimulationStore
    """

    def __init__(self, store, rounds=None):
        """
        rounds:
            None - use all rounds
            list[int] - selected rounds
        """

        theta, x, round_id = store.get_all()

        if rounds is not None:
            mask = torch.zeros_like(round_id, dtype=torch.bool)
            for r in rounds:
                mask |= round_id == r

            theta = theta[mask]
            x = x[mask]

        self.theta = theta
        self.x = x

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, idx):

        return (self.theta[idx], self.x[idx])

# import bisect
# import torch
# from collections.abc import Sequence
# from torch.utils.data import Dataset

# class RoundDataset(Dataset):
#     def __init__(self, store, rounds=None):
#         """
#         rounds : None (все раунды), int или список int – раунды для включения
#         """
#         if rounds is None:
#             chunks = store.get_all_chunks()
#         else:
#             if isinstance(rounds, int):
#                 rounds = [rounds]
#             chunks = []
#             for r in rounds:
#                 chunks.extend(store.get_round_chunks(r))
#         self.chunks = chunks

#         # предварительно считаем границы чанков для быстрого поиска
#         self.cum_sizes = []
#         total = 0
#         for ch in self.chunks:
#             total += ch["n_samples"]
#             self.cum_sizes.append(total)

#         # кэш для ускорения последовательного доступа
#         self._cached_chunk_idx = None
#         self._cached_data = None

#     def __len__(self):
#         return self.cum_sizes[-1] if self.cum_sizes else 0

#     def __getitem__(self, idx):
#         # в каком чанке лежит индекс
#         chunk_idx = bisect.bisect_right(self.cum_sizes, idx)
#         local_idx = idx - (self.cum_sizes[chunk_idx - 1] if chunk_idx > 0 else 0)

#         if self._cached_chunk_idx != chunk_idx:
#             self._cached_data = torch.load(
#                 self.chunks[chunk_idx]["path"], weights_only=False
#             )
#             self._cached_chunk_idx = chunk_idx

#         return self._cached_data["theta"][local_idx], self._cached_data["x"][local_idx]

#     def get_all(self):
#         """Вернуть все данные текущего датасета (theta, x)."""
#         thetas, xs = [], []
#         for ch in self.chunks:
#             data = torch.load(ch["path"], weights_only=False)
#             thetas.append(data["theta"])
#             xs.append(data["x"])
#         return (torch.cat(thetas, dim=0),
#                 torch.cat(xs, dim=0))

#     def get_round(self, rounds):
#         """
#         Вернуть данные (theta, x) для указанных раундов (int или list[int]),
#         загружая их напрямую из хранилища.
#         """
#         if isinstance(rounds, int):
#             rounds = [rounds]
#         chunks = []
#         for r in rounds:
#             chunks.extend(self.store.get_round_chunks(r))
#         thetas, xs = [], []
#         for ch in chunks:
#             data = torch.load(ch["path"], weights_only=False)
#             thetas.append(data["theta"])
#             xs.append(data["x"])
#         return (torch.cat(thetas, dim=0),
#                 torch.cat(xs, dim=0))