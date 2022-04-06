import os
import pickle
import torch
import sys
sys.path.append("./src/")
import config
import data_utils
import train_helper
from tree import ud_to_list, ud_to_pos
from models import vgvae

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

best_dev_res = test_bm_res = test_avg_res = best_distance = 0

emb_config = {
    'mbert': 768,
    'xlm': 1280,
    'xlmr': 768
}

def run(e):
    global best_dev_res, test_bm_res, test_avg_res, best_distance

    e.config.edim = emb_config[e.config.ml_type]

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    tags = ["<pad>"]
    train_pos_1, tags_1 = ud_to_pos.load_conll_dataset("./data/SSDM_data/DP/sentence1_tree.txt")
    train_pos_2, tags_2 = ud_to_pos.load_conll_dataset("./data/SSDM_data/DP/sentence2_tree.txt")
    tags += tags_1
    for t in tags_2:
        if t not in tags:
            tags.append(t)
    tag2idx = {tag: idx for idx, tag in enumerate(tags)}

    if not os.path.exists("./data/SSDM_data/SSDM_train_{}.pkl".format(e.config.ml_type)):
        train_dp_1 = ud_to_list.load_conll_dataset("./data/SSDM_data/DP/sentence1_tree.txt")
        train_dp_2 = ud_to_list.load_conll_dataset("./data/SSDM_data/DP/sentence2_tree.txt")
        dp = data_utils.data_processor(
            dp_1=train_dp_1,
            dp_2=train_dp_2,
            pos_path1=train_pos_1,
            pos_path2=train_pos_2,
            pos_id=tag2idx,
            experiment=e)
        data, tokenizer = dp.process()
        output_hal = open("./data/SSDM_data/SSDM_train_{}.pkl".format(e.config.ml_type), 'wb')
        str = pickle.dumps(data)
        output_hal.write(str)
        output_hal.close()
    else:
        with open("./data/SSDM_data/SSDM_train_{}.pkl".format(e.config.ml_type), 'rb') as file:
            data = pickle.loads(file.read())

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    model = vgvae(
        vocab_size=len(data.vocab),
        embed_dim=e.config.edim,
        experiment=e,
        tags_vocab_size=len(tag2idx))

    start_epoch = true_it = 0
    if e.config.resume:
        try:
            start_epoch, _, best_dev_res, test_avg_res = \
                model.load(name="latest")
        except:
            start_epoch = 0
        if e.config.use_cuda:
            model.cuda()
            e.log.info("transferred model to gpu")
        e.log.info(
            "resumed from previous checkpoint: start epoch: {}, "
            "iteration: {}, best dev res: {:.3f}, test avg res: {:.3f}"
                .format(start_epoch, true_it, best_dev_res, test_avg_res))

    e.log.info(model)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    minibatcher = data_utils.minibatcher

    train_batch = minibatcher(
        data1=data.train_data[0],
        data2=data.train_data[1],
        dp1=data.train_data[2],
        dp2=data.train_data[3],
        pos=data.train_data[4],
        vocab_size=len(data.vocab),
        batch_size=e.config.batch_size,
        score_func=model.score,
        shuffle=False,
        mega_batch=e.config.mb,
        p_scramble=e.config.ps)

    evaluator = train_helper.evaluator(model, e)

    e.log.info("Training start ...")
    train_stats = train_helper.tracker(["loss", "vmf_kl", "gauss_kl",
                                        "RL", "CRL",
                                        "WPL", "SDL", "POS_loss", "STL"])
    no_new = 0
    stop_early = False

    e.log.info("*" * 25 + " TEST EVAL: mBERT " + "*" * 25)

    for epoch in range(start_epoch, e.config.n_epoch):
        if stop_early:
            break
        if epoch > 1 and train_batch.mega_batch != e.config.mb:
            train_batch.mega_batch = e.config.mb
            train_batch._reset()
        e.log.info("current mega batch: {}".format(train_batch.mega_batch))

        for it, (s1, m1, s2, m2, t1, tm1, t2, tm2,
                 n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2, tree_tags, y, _) in \
                enumerate(train_batch):
            true_it = it + 1 + epoch * len(train_batch)
            loss, vkl, gkl, rec_logloss, para_logloss, ploss, sdl, pos_loss, stl = \
                model(s1, m1, s2, m2, t1, tm1, t2, tm2,
                      n1, nm1, nt1, ntm1, n2, nm2, nt2, ntm2,
                      e.config.vmkl, e.config.gmkl, tree_tags, y,
                      e.config.mb > 1, true_it=true_it)

            model.optimize(loss)

            train_stats.update(
                {"loss": loss, "vmf_kl": vkl, "gauss_kl": gkl,
                 "RL": para_logloss, "CRL": rec_logloss,
                 "WPL": ploss, "SDL": sdl, "POS_loss": pos_loss, "STL": stl},
                len(s1))

            if (true_it + 1) % e.config.print_every == 0 or \
                    (true_it + 1) % len(train_batch) == 0:
                summarization = train_stats.summarize(
                    "epoch: {}, it: {} (max: {}), kl_temp: {:.2E}|{:.2E}"
                        .format(epoch, it, len(train_batch),
                                e.config.vmkl, e.config.gmkl))
                e.log.info(summarization)
                train_stats.reset()

            if (true_it + 1) % e.config.eval_every == 0 or \
                    (true_it + 1) % len(train_batch) == 0:

                e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)
                dev_stats, _, dev_res, _ = evaluator.evaluate(
                    data.dev_data, 'pred')
                e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

                e.log.info("*" * 25 + " TEST EVAL: SEMANTICS " + "*" * 25)
                test_stats, test_bm_res, test_avg_res, test_avg_s = \
                    evaluator.evaluate(data.test_data, 'pred')
                e.log.info("*" * 25 + " TEST EVAL: SEMANTICS " + "*" * 25)

                e.log.info("*" * 25 + " TEST EVAL: SYNTAX " + "*" * 25)
                tz_stats, tz_bm_res, tz_avg_res, tz_avg_s = \
                    evaluator.evaluate(data.test_data, 'predz')
                e.log.info("*" * 25 + " TEST EVAL: SYNTAX " + "*" * 25)

                distance = abs(test_avg_res - tz_avg_res)
                if best_dev_res < dev_res:
                    no_new = 0
                    best_dev_res = dev_res

                    model.save(
                        dev_avg=best_dev_res,
                        dev_perf=dev_stats,
                        test_avg=test_avg_res,
                        test_perf=test_stats,
                        iteration=true_it,
                        epoch=epoch)

                    if distance > best_distance:
                        best_distance = distance
                elif distance > best_distance:
                    best_distance = distance
                    model.save(
                        dev_avg=best_dev_res,
                        dev_perf=dev_stats,
                        test_avg=test_avg_res,
                        test_perf=test_stats,
                        iteration=true_it,
                        epoch=epoch,
                        name="distant")
                    continue
                else:
                    no_new += 1

                train_stats.reset()
                e.log.info("best dev result: {:.4f}, "
                           "longest distance: {:.4f}, "
                           .format(best_dev_res, best_distance))
            if no_new == 15:
                if best_dev_res:
                    e.log.info("*" * 25 + "stop early!" + "*" * 25)
                    stop_early = True
                    break
                else:
                    no_new = 0
            it += 1

        model.save(
            dev_avg=best_dev_res,
            dev_perf=dev_stats,
            test_avg=test_avg_res,
            test_perf=test_stats,
            iteration=true_it,
            epoch=epoch + 1,
            name="latest")

    e.log.info("*" * 25 + " TEST EVAL: SEMANTICS " + "*" * 25)
    test_stats, test_bm_res, test_avg_res, test_avg_s = \
        evaluator.evaluate(data.test_data, 'pred')
    e.log.info("*" * 25 + " TEST EVAL: SEMANTICS " + "*" * 25)

    e.log.info("*" * 25 + " TEST EVAL: SYNTAX " + "*" * 25)
    tz_stats, tz_bm_res, tz_avg_res, tz_avg_s = \
        evaluator.evaluate(data.test_data, 'predz')
    e.log.info("*" * 25 + " TEST EVAL: SYNTAX " + "*" * 25)


if __name__ == '__main__':
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