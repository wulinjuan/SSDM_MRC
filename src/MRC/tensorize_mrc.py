import os
import sys
from os.path import join
import pickle
import logging
from squad import SquadProcessor, squad_convert_examples_to_features
import util

logger = logging.getLogger(__name__)


class QaDataProcessor:
    def __init__(self, config):
        self.config = config

        self.max_seg_len = config['max_segment_len']
        self.doc_stride = config['doc_stride']
        self.max_query_len = config['max_query_len']

        self.tokenizer = None  # Lazy loading

    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = util.get_bert_tokenizer(self.config)
        return self.tokenizer

    def _get_data(self, dataset_name, partition, lang, data_dir, data_file):
        cache_feature_path = self.get_cache_feature_path(dataset_name, partition, lang)
        cache_dataset_path = self.get_cache_dataset_path(dataset_name, partition, lang)
        if os.path.exists(cache_feature_path):
            with open(cache_feature_path, 'rb') as f:
                examples, features = pickle.load(f)
            with open(cache_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            to_return = (examples, features, dataset)
            logger.info('Loaded features and dataset from cache')
        else:
            logger.info(f'Getting {dataset_name}-{partition}-{lang}; results will be cached')
            processor = SquadProcessor()
            examples = processor.get_train_examples(data_dir, data_file) if partition == 'train' else \
                processor.get_dev_or_test_examples(data_dir, data_file)  # For train, only use first answer as gold
            features, dataset = squad_convert_examples_to_features(examples, self.get_tokenizer(),
                                                                   max_seq_length=self.max_seg_len,
                                                                   doc_stride=self.doc_stride,
                                                                   max_query_length=self.max_query_len,
                                                                   config=self.config,
                                                                   is_training=(partition == 'train'),
                                                                   return_dataset=True)
            with open(cache_feature_path, 'wb') as f:
                pickle.dump((examples, features), f, protocol=4)
            max_bytes = 2 ** 31 - 1
            bytes_out = pickle.dumps(dataset)
            n_bytes = sys.getsizeof(bytes_out)
            with open(cache_dataset_path, 'wb') as f:
                for idx in range(0, n_bytes, max_bytes):
                    f.write(bytes_out[idx:idx + max_bytes])
                # pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info('Saved features and dataset to cache')
            to_return = (examples, features, dataset)
        return to_return

    def get_source(self, dataset_name, partition, only_dataset=False):
        assert dataset_name in ['squad', 'tydiqa']
        if dataset_name == 'squad':
            data_dir = join(self.config['download_dir'], dataset_name)
            data_file = f'{partition}-v1.1.json'
        else:
            data_dir = join(self.config['download_dir'], dataset_name, f'tydiqa-goldp-v1.1-{partition}')
            data_file = f'tydiqa.en.{partition}.json'
        cache_dataset_path = self.get_cache_dataset_path(dataset_name, partition, 'en')

        if only_dataset and os.path.exists(cache_dataset_path):
            with open(cache_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            logger.info('Loaded dataset from cache')
            return dataset

        examples, features, dataset = self._get_data(dataset_name, partition, 'en', data_dir, data_file)
        return dataset if only_dataset else (examples, features, dataset)

    def get_target(self, dataset_name, partition, lang, only_dataset=False):
        assert dataset_name in ['xquad', 'mlqa', 'tydiqa']
        if dataset_name == 'xquad':
            data_dir = join(self.config['download_dir'], 'xquad')
            data_file = f'xquad.{lang}.json'
        elif dataset_name == 'mlqa':
            data_dir = join(self.config['download_dir'], f'mlqa/MLQA_V1/{partition}')
            data_file = f'{partition}-context-{lang}-question-{lang}.json'
        else:
            data_dir = join(self.config['download_dir'], dataset_name, f'tydiqa-goldp-v1.1-dev')
            data_file = f'tydiqa.{lang}.dev.json'
        cache_dataset_path = self.get_cache_dataset_path(dataset_name, partition, lang)

        if only_dataset and os.path.exists(cache_dataset_path):
            with open(cache_dataset_path, 'rb') as f:
                dataset = pickle.load(f)
            logger.info('Loaded dataset from cache')
            return dataset

        examples, features, dataset = self._get_data(dataset_name, partition, lang, data_dir, data_file)
        return dataset if only_dataset else (examples, features, dataset)

    def get_cache_feature_path(self, dataset_name, partition, lang):
        cache_dir = join(self.config['data_dir'], 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        model_type = self.config['model_type']
        cache_name = f'{dataset_name}.{partition}.{lang}.{self.max_seg_len}.{model_type}'
        cache_path = join(cache_dir, f'{cache_name}.bin')
        return cache_path

    def get_cache_dataset_path(self, dataset_name, partition, lang):
        cache_path = self.get_cache_feature_path(dataset_name, partition, lang)
        cache_path = cache_path[:-4] + '.dataset'
        return cache_path
