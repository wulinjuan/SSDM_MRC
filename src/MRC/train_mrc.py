import logging
import pickle
import sys
sys.path.append("./src/MRC/")

import time
from datetime import datetime
from os.path import join

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

import util
from evaluate_mlqa import evaluate as mlqa_evaluate
from evaluate_squad import evaluate as squad_evaluate
from model_mrc import TransformerQa
from squad import SquadResult
from squad_metrics import compute_predictions_logits
from src.SSDM import models, data_utils
from src.SSDM import train_helper
from tensorize_mrc import QaDataProcessor

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()

# model_types_wo_token_type_ids = ["xlm", "roberta", "distilbert", "camembert", "bart", "mt5"]


class QaRunner:
    def __init__(self, config_name, gpu_id=0, seed=None):
        self.name = config_name
        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id
        self.seed = seed

        # Set up config
        self.config = util.initialize_config(config_name)

        # Set up logger
        log_path = join(self.config['log_dir'], 'log_' + self.name_suffix + '.txt')
        logger.addHandler(logging.FileHandler(log_path, 'a'))
        logger.info(f'Log file path: {log_path}')

        # Set up seed
        if seed:
            util.set_seed(seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id is None else f'cuda:{gpu_id}')

        # Set up data
        self.data = QaDataProcessor(self.config)

        conf = self.config
        save_dict = torch.load(
            conf.syntax_model_dir,
            map_location=lambda storage,
                                loc: storage)
        config = save_dict['config']
        checkpoint = save_dict['state_dict']
        config.debug = True
        Token = data_utils.MODEL_CLASSES[config.ml_type]
        vocab = Token.from_pretrained(config.ml_token)
        with train_helper.experiment(config, config.save_file_path) as e:
            self.model_syntax = models.vgvae(
                vocab_size=len(vocab.vocab),
                embed_dim=e.config.edim,
                experiment=e,
                tags_vocab_size=18)
            self.model_syntax.eval()
            self.model_syntax.load(checkpointed_state_dict=checkpoint)
            # e.log.info(model_syntax)

    def initialize_model(self, saved_suffix=None):
        model = TransformerQa(self.config)
        if saved_suffix:
            self.load_model_checkpoint(model, saved_suffix)
        return model

    def prepare_inputs(self, batch, is_training):
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'dis_model': self.model_syntax
        }
        if is_training:
            inputs.update({
                'start_positions': batch[3],
                'end_positions': batch[4]
            })
        return inputs

    def train(self, model):
        conf = self.config
        logger.info(conf)
        epochs, batch_size, grad_accum = conf['num_epochs'], conf['batch_size'], conf['gradient_accumulation_steps']

        model.to(self.device)

        # Set up tensorboard
        tb_path = join(conf['tb_dir'], self.name + '_' + self.name_suffix)
        tb_writer = SummaryWriter(tb_path, flush_secs=30)
        logger.info(f'Tensorboard summary path: {tb_path}')

        # Set up data
        train_dataset = self.data.get_source(conf['train_dataset'], 'train', only_dataset=True)
        dev_examples, dev_features, dev_dataset = self.data.get_source(conf['train_dataset'], 'dev')
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset),
                                      batch_size=batch_size, drop_last=False)

        # Set up optimizer and scheduler
        total_update_steps = len(train_dataloader) * epochs // grad_accum
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer, total_update_steps)
        trained_params = model.parameters()

        # Start training
        logger.info('*******************Training*******************')
        logger.info('Num samples: %d' % len(train_dataset))
        logger.info('Num epochs: %d' % epochs)
        logger.info('Gradient accumulation steps: %d' % grad_accum)
        logger.info('Total update steps: %d' % total_update_steps)

        loss_during_accum = []  # To compute effective loss at each update
        loss_during_report = 0.0  # Effective loss during logging step
        loss_history = []  # Full history of effective loss; length equals total update steps
        max_f1 = 0
        start_time = time.time()
        model.zero_grad()
        for epo in range(epochs):
            print("epoch: ", epo)
            for batch in train_dataloader:
                model.train()
                inputs = self.prepare_inputs(batch, is_training=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                loss, _ = outputs = model(**inputs)
                if grad_accum > 1:
                    loss /= grad_accum
                loss.backward()
                loss_during_accum.append(loss.item())

                # Update
                if len(loss_during_accum) % grad_accum == 0:
                    if conf['max_grad_norm']:
                        torch.nn.utils.clip_grad_norm_(trained_params, conf['max_grad_norm'])
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()

                    # Compute effective loss
                    effective_loss = np.sum(loss_during_accum).item()
                    loss_during_accum = []
                    loss_during_report += effective_loss
                    loss_history.append(effective_loss)

                    # Report
                    if len(loss_history) % conf['report_frequency'] == 0:
                        # Show avg loss during last report interval
                        avg_loss = loss_during_report / conf['report_frequency']
                        loss_during_report = 0.0
                        end_time = time.time()
                        logger.info('Step %d: avg loss %.2f; steps/sec %.2f' %
                                    (len(loss_history), avg_loss, conf['report_frequency'] / (end_time - start_time)))
                        start_time = end_time

                        tb_writer.add_scalar('Training_Loss', avg_loss, len(loss_history))
                        tb_writer.add_scalar('Learning_Rate_Bert', scheduler.get_last_lr()[0], len(loss_history))

                    # Evaluate
                    if len(loss_history) > 3000 and len(loss_history) % conf['eval_frequency'] == 0:
                        metrics, _ = self.evaluate(model, dev_examples, dev_features, dev_dataset, len(loss_history), tb_writer)
                        if metrics['f1'] > max_f1:
                            max_f1 = metrics['f1']
                            self.save_model_checkpoint(model, len(loss_history))
                        logger.info(f'Eval max f1: {max_f1:.2f}')
                        start_time = time.time()

        logger.info('**********Finished training**********')
        logger.info('Actual update steps: %d' % len(loss_history))

        # Eval at the end
        metrics, _ = self.evaluate(model, dev_examples, dev_features, dev_dataset, len(loss_history), tb_writer)
        if metrics['f1'] > max_f1:
            max_f1 = metrics['f1']
            self.save_model_checkpoint(model, len(loss_history))
        logger.info(f'Eval max f1: {max_f1:.2f}')

        # Wrap up
        tb_writer.close()
        return loss_history

    def evaluate(self, model, examples, features, dataset, step=0, tb_writer=None, output_results_file=None,
                 output_prediction_file=None, output_nbest_file=None, output_null_log_odds_file=None,
                 dataset_name=None, lang=None, verbose_logging=False):
        conf = self.config
        logger.info(f'Step {step}: evaluating on {len(dataset)} samples...')
        dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=conf['batch_size'],
                                pin_memory=False, num_workers=0, persistent_workers=False)

        model.eval()
        model.to(self.device)
        results = []
        for batch in dataloader:
            feature_indices = batch[3]  # To identify feature in batch eval
            inputs = self.prepare_inputs(batch, is_training=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # Build results from batch output
            for i, feature_idx in enumerate(feature_indices):
                feature = features[feature_idx.item()]
                feature_unique_id = int(feature.unique_id)
                feature_output = [output[i].tolist() for output in outputs]

                start_logits, end_logits = feature_output[:2]
                result = SquadResult(feature_unique_id, start_logits, end_logits)
                results.append(result)

        if output_results_file:
            with open(output_results_file, 'wb') as f:
                pickle.dump(results, f)

        # Evaluate
        metrics, predictions = self.evaluate_from_results(examples, features, results, output_prediction_file,
                                                          output_nbest_file, output_null_log_odds_file,
                                                          dataset_name, lang, verbose_logging)
        if tb_writer:
            for name, val in metrics.items():
                tb_writer.add_scalar(f'Train_Eval_{name}', val, step)
        return metrics, predictions

    def evaluate_from_results(self, examples, features, results, output_prediction_file=None, output_nbest_file=None,
                              output_null_log_odds_file=None, dataset_name=None, lang=None, verbose_logging=False):
        conf = self.config
        predictions = compute_predictions_logits(examples, features, results, conf['n_best_predictions'],
                                                 conf['max_answer_len'], False, output_prediction_file,
                                                 output_nbest_file, output_null_log_odds_file,
                                                 conf['version_2_with_negative'], conf['null_score_diff_threshold'],
                                                 self.data.get_tokenizer(), verbose_logging)
        if dataset_name == 'mlqa':
            metrics = mlqa_evaluate(examples, predictions, lang)
        else:
            metrics = squad_evaluate(examples, predictions)
        return metrics, predictions

    def get_optimizer(self, model):
        no_decay = ['bias', 'LayerNorm.weight']
        grouped_param = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config['adam_weight_decay']
            }, {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = AdamW(grouped_param, lr=self.config['bert_learning_rate'], eps=self.config['adam_eps'])
        return optimizer

    def get_scheduler(self, optimizer, total_update_steps):
        if self.config['model_type'] == 'mt5':
            # scheduler = get_constant_schedule(optimizer)
            cooldown_start = int(total_update_steps * 0.7)

            def lr_lambda(current_step: int):
                return 1 if current_step < cooldown_start else 0.3
            return LambdaLR(optimizer, lr_lambda, -1)
        else:
            warmup_steps = int(total_update_steps * self.config['warmup_ratio'])
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                        num_training_steps=total_update_steps)
        return scheduler

    def save_model_checkpoint(self, model, step):
        path_ckpt = join(self.config['log_dir'], f'model_{self.name_suffix}.bin')
        torch.save(model.state_dict(), path_ckpt)
        logger.info('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, model, suffix):
        path_ckpt = join(self.config['log_dir'], f'model_{suffix}.bin')
        model.load_state_dict(torch.load(path_ckpt, map_location=torch.device('cpu')), strict=False)
        logger.info('Loaded model from %s' % path_ckpt)


if __name__ == '__main__':
    config_name, gpu_id = sys.argv[1], int(sys.argv[2])
    runner = QaRunner(config_name, gpu_id)
    model = runner.initialize_model()

    runner.train(model)
