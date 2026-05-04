import torch
import pandas as pd

class SimulationStore:
    """
    Storage for all simulations across sequential rounds
    """

    def __init__(self, df_path):
        # self.df = pd.DataFrame(columns=["theta", "x", "round_id"])
        # self.df_path = df_path
        self.theta = []
        self.x = []
        self.round_id = []


    def add(self, theta, x, round_id):
        """
        Add simulations to store

        theta : (B, N, theta_dim) or (N, theta_dim) 
        x : (B, N, x_dim) or (N, x_dim)
        """
        # if len(theta.shape) > 2:
        #     print("Theta", theta.shape, x.shape)
        #     # theta = theta.view(-1, theta.shape[-1])
        #     # x = x.view(-1, x.shape[-1])
        #     print("Theta after reshape", theta.shape, x.shape)
        #     return 0
        
        self.theta.append(theta.detach())
        self.x.append(x.detach())
        self.round_id.append(
            torch.full((*theta.shape[:-1], ), round_id, device=theta.device)
        )
        # print("SimStore1: ", self.theta[-1].shape, self.x[-1].shape, self.round_id[-1].shape)

    def get_all(self):
        """
        Return all stored simulations
        """
        # print("SimStore2: ", self.theta[0].shape, self.x[0].shape, self.round_id[0].shape)
        # print("SimStore3: ", self.theta[-1].shape, self.x[-1].shape, self.round_id[-1].shape)
        
        return (
            torch.cat(self.theta, dim=0),
            torch.cat(self.x, dim=0),
            torch.cat(self.round_id, dim=0),
        ) 

    def get_round(self, round_id):
        """
        Return simulations from specific round
        """

        theta, x, r = self.get_all()

        mask = r == round_id

        return theta[mask], x[mask]

    def size(self):
        if len(self.theta) == 0:
            return 0
        return sum(t.shape[0] for t in self.theta)

    def rounds(self):
        return self.round_id.unique()

# import torch
# from pathlib import Path
# from collections.abc import Sequence
# import bisect

# class SimulationStore:
#     def __init__(self, storage_dir: str, chunk_prefix: str = "chunk"):
#         self.storage_dir = Path(storage_dir)
#         self.storage_dir.mkdir(parents=True, exist_ok=True)
#         self.chunk_prefix = chunk_prefix
#         self.chunks = []          # каждый элемент: dict(round_id=int, n_samples=int, path=str)
#         self._next_chunk_id = 0

#     def add(self, theta: torch.Tensor, x: torch.Tensor, round_id: int):
#         """
#         theta: (*B, theta_dim) или (theta_dim,)  — батч параметров
#         x:     (*B, x_dim) или (x_dim,)          — соответствующие наблюдения
#         """
#         theta = theta.detach().cpu()
#         x = x.detach().cpu()

#         # если на входе один сэмпл без батчевого измерения – добавим его
#         if theta.ndim == 1:
#             theta = theta.unsqueeze(0)
#             x = x.unsqueeze(0)

#         n_samples = theta.shape[0]
#         chunk_path = self.storage_dir / f"{self.chunk_prefix}_{self._next_chunk_id:06d}.pt"
#         torch.save({"theta": theta, "x": x, "round_id": round_id}, chunk_path)
#         self.chunks.append({
#             "round_id": round_id,
#             "n_samples": n_samples,
#             "path": str(chunk_path)
#         })
#         self._next_chunk_id += 1

#     def get_all_chunks(self):
#         """Возвращает список всех чанков (для использования в Dataset)."""
#         return self.chunks

#     def get_round_chunks(self, round_id):
#         """Возвращает чанки, относящиеся к конкретному раунду."""
#         return [ch for ch in self.chunks if ch["round_id"] == round_id]
    
#     def get_all(self):
#         """Вернуть все симуляции в виде тензоров (theta, x, round_id)."""
#         thetas, xs, rids = [], [], []
#         for ch in self.chunks:
#             data = torch.load(ch["path"], weights_only=False)
#             thetas.append(data["theta"])
#             xs.append(data["x"])
#             # round_id хранится как int в чанке, создаём тензор для каждого сэмпла
#             rids.append(torch.full((data["theta"].shape[0],), ch["round_id"], dtype=torch.int))
#         return (torch.cat(thetas, dim=0),
#                 torch.cat(xs, dim=0),
#                 torch.cat(rids, dim=0))

#     def get_round(self, round_id):
#         """Вернуть симуляции конкретного раунда (theta, x)."""
#         thetas, xs = [], []
#         for ch in self.get_round_chunks(round_id):
#             data = torch.load(ch["path"], weights_only=False)
#             thetas.append(data["theta"])
#             xs.append(data["x"])
#         return (torch.cat(thetas, dim=0),
#                 torch.cat(xs, dim=0))

#     def size(self):
#         return sum(ch["n_samples"] for ch in self.chunks)

#     def rounds(self):
#         return sorted({ch["round_id"] for ch in self.chunks})