import os
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.utilities import rank_zero_only

class ITPEnvironment(ClusterEnvironment):

    def creates_children(self) -> bool:
        # return True if the cluster is managed (you don't launch processes yourself)
        return True

    def world_size(self) -> int:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])

    def set_world_size(self, size: int) -> None:
        self._world_size = size
    
    def set_global_rank(self, rank: int) -> None:
        self._global_rank = rank
        rank_zero_only.rank = rank    

    def global_rank(self) -> int:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])

    def local_rank(self) -> int:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

    def node_rank(self) -> int:
        return int(os.environ["NODE_RANK"])

    def master_address(self) -> str:
        return os.environ['MASTER_ADDR']

    def master_port(self) -> int:
        return os.environ['MASTER_PORT']
