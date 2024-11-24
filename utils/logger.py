import json
import os
import yaml
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, config_path):
        '''
        log_dir is the path to the experiments directory
        config_path is the path to the relevant config file.
        '''
        # verify that the log directory exists, if not, create it.
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # automatically populate the config experiment's logging path
        self.config_path = config_path
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # extract the model name
        model_name = config['model']['name']
        print("model name: ", model_name)
        print("log_dir: ", log_dir)
        # Dynamically construct the logging path
        self.experiment_path = os.path.join("./",log_dir, model_name)
        print("self.experiment_path: ", self.experiment_path)

        if not os.path.exists(self.experiment_path):
            os.makedirs(self.experiment_path)
            # save the modified config file
            config['logging']['path'] = self.experiment_path

            with open(config_path, 'w') as file:
                yaml.safe_dump(config, file)

        else:
            # if it's already been made, we have a problem since we don't want duplicates
            raise Exception("This model name has already been used for an experiment")

        self.writer = SummaryWriter(log_dir=self.experiment_path)

        # self.logs = {}

        print(f"Logger initialized, config saved with updated logging path:
              {self.experiment_path}")

    def get_parameter_save_path(self): # returns the path to save the parameters of the model to #TODO implement versioning for early stopping etc.
        return self.experiment_path + "/model_parameters.pth"

    def log_scalar(self, key, value, step): # e.g. loss or accuracy, for each training step or epoch.
        """Logs a scalar value to TensorBoard."""
        self.writer.add_scalar(key, value, step)

    def log_histogram(self, key, values, step=None): # for weights, gradients or other tensor values.
        """Logs a histogram of values to TensorBoard."""
        self.writer.add_histogram(key, values, step)

    def log_text(self, tag, text, step):
        """Logs text to TensorBoard."""
        self.writer.add_text(tag, text, step)

    def log_image(self, tag, images, step):
        """Logs images to TensorBoard."""
        self.writer.add_images(tag, images, step)

    def save_logs(self): # ensures that all logs are flushed
        """Flushes the logs to TensorBoard."""
        self.writer.flush()

    def close(self): # terminates the SummaryWriter afte rall training is completed.
        """Closes the TensorBoard writer."""
        self.writer.close()

    # old functions for the simple dictionary version of logging

    # def log(self, key, value):
    #     if key not in self.logs:
    #         self.logs[key] = []
    #     self.logs[key].append(value)

    # def save_logs(self):
    #     with open(os.path.join(self.log_dir, 'logs.json'), 'w') as f:
    #         json.dump(self.logs, f)

# Usage: logger.log('accuracy', acc_value)

# TODO should the experiment path automatically increment a counter as a suffix if someone runs the model with the same name over again.
# TODO get teh dataloader to use a tensorboard writer instead of a simple dictionary
