import os
import yaml
from argparse import Namespace


def parse_yml(path):
    if not os.path.exists(path):
        return None

    f = open(path ,"r")
    config = yaml.safe_load(f)
    return config


def combine(args, config):
    dict_args = vars(args)
    for k, v in config.items():
        if k not in dict_args.keys():
            continue  #     print(f"`{k}` not in terminal arguments")
        elif dict_args[k] != v:
            print(f"[CONFLICT]: `{k}` command-line: `{dict_args[k]}` configuration file:`{v}`\nIgnore value in command line")

    # overwrite command line argument or configuration file
    overwrite = args.overwrite
    if overwrite == "command-line":
        dict_args.update(config)
        comb_dict = dict_args
    else:
        config.update(dict_args)
        comb_dict = config

    # for k, v in comb_dict.items():
    #     if k == "dataset":
    #         for sk, sv in comb_dict["dataset"].items():
    #             if isinstance(sv, dict):
    #                 comb_dict["dataset"][sk] = Namespace(**sv)
            
    #         comb_dict["dataset"] = Namespace(**comb_dict["dataset"])

    comb_dict["dataset"] = Namespace(**comb_dict["dataset"])
    comb_args = Namespace(**comb_dict)

    return comb_args