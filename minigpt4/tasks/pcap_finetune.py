from minigpt4.common.registry import registry
from minigpt4.tasks.base_task import BaseTask


@registry.register_task("pcap_finetune")
class PCapFinetuneTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass
    
    
    # TODO: find out which part is responsible for playing w/ prompts
    # and modify it to include personality explicitly 