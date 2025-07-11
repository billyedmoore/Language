from .api import RNNAPI

from common.argument_parsing import get_argparser, execute_by_arguments
from common.get_torch_device import get_torch_device


def main(input_length, device, dataset_size):
    api = RNNAPI(device, input_length, "shakespeare", dataset_size)
    parser = get_argparser("RNN Model")
    args = parser.parse_args()
    execute_by_arguments(args, api)


if __name__ == "__main__":
    main(200, get_torch_device(silent=True), 10000)
