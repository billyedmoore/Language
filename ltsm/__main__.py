from .api import LTSMAPI

from common.argument_parsing import get_argparser, execute_by_arguments
from common.get_torch_device import get_torch_device


def main(input_length, device, dataset_size, hidden_layer_size):
    api = LTSMAPI(
        device, input_length, "Emma", dataset_size, hidden_layer_size=hidden_layer_size
    )
    parser = get_argparser("LTSM Model")
    args = parser.parse_args()
    execute_by_arguments(args, api)


if __name__ == "__main__":
    main(100, get_torch_device(silent=True), 8000, 256)
