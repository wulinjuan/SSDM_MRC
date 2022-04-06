import warnings

warnings.filterwarnings("ignore")

from src.MRC.cross_lingual_mrc_test import Evaluator
from src.MRC.train_mrc import QaRunner
from src.SSDM import train_helper
from src.SSDM.train_ssdm import run, best_dev_res, test_bm_res, test_avg_res
import torch
import config


if __name__ == '__main__':
    # ssdm
    args = config.get_base_parser().parse_args()
    args.use_cuda = torch.cuda.is_available()

    def exit_handler(*args):
        print(args)
        print("best dev result: {:.4f}, "
              "STSBenchmark result: {:.4f}, "
              "test average result: {:.4f}"
              .format(best_dev_res, test_bm_res, test_avg_res))
        exit()


    train_helper.register_exit_handler(exit_handler)

    with train_helper.experiment(args, args.save_file_path) as e:
        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(e)

    if args.use_cuda:
        gpu_id = 0
    else:
        gpu_id = None
    # mrc
    config_name = "{}_zero_shot".format(args.ml_type)
    runner = QaRunner(config_name, gpu_id)
    model = runner.initialize_model()

    runner.train(model)

    # cross-lingual mrc test
    config_name, saved_suffix = "{}_zero_shot".format(args.ml_type), runner.name_suffix
    dataset_name = "xquad"
    evaluator = Evaluator(config_name, saved_suffix, gpu_id)
    evaluator.evaluate_task(dataset_name)

    dataset_name = "mlqa"
    evaluator = Evaluator(config_name, saved_suffix, gpu_id)
    evaluator.evaluate_task(dataset_name)

    dataset_name = "tydiqa"
    evaluator = Evaluator(config_name, saved_suffix, gpu_id)
    evaluator.evaluate_task(dataset_name)
