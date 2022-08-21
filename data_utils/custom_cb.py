import requests
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.distributed import rank_zero_only


class ClientLogger(Logger):
    def __init__(self, model_path, log_url=None, task_id=None, max_epochs=None, response_url=None, ocr_type=None):
        super().__init__()
        self.log_url = log_url
        self.task_id = task_id
        self.max_epochs = max_epochs
        self.last_metrics = {}
        self.response_url = response_url
        self.model_path = model_path
        self.ocr_type = ocr_type

    @property
    def name(self):
        return "ClientLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        print(f"Logging step: {step}")
        metrics['step'] = step
        metrics['task_id'] = self.task_id
        metrics['ocr_type'] = self.ocr_type
        if self.max_epochs is not None:
            metrics['is_finished'] = metrics['epoch'] >= self.max_epochs
        print(f"Logging metrics: {metrics}")
        if self.log_url:
            requests.post(self.log_url, json=metrics)
        self.last_metrics = metrics

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        print(f"Finalizing with status: {status}, last metrics: {self.last_metrics}")
        if self.response_url:
            requests.post(self.log_url, json={'status': status, 'task_id': self.task_id, 'is_finished': True,
                                              'ocr_type': self.ocr_type,
                                              'metrics': self.last_metrics,
                                              'model_path': self.model_path
                                              }
                          )
