import logging


def get_logger(name="my_logger"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    return logging.getLogger(name)


def append_dict(dictionary, history):
    for metric in dictionary:
        dictionary[metric].append(history[metric])


def append_dict_from_list(dictionary, history):
    for i, metric in enumerate(dictionary):
        dictionary[metric].append(history[i])


def check_format(dictionary):
    for metric in dictionary:
        dictionary[metric] = unsqueeze_list(dictionary[metric])


def unsqueeze_list(list_metric):
    return [
        item if not isinstance(item, list) else item[0] for item in list_metric
    ]