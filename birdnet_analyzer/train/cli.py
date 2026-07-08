from birdnet_analyzer.utils import runtime_error_handler


@runtime_error_handler
def main():
    from birdnet_analyzer import cli
    from birdnet_analyzer.train.core import train

    # Parse arguments
    parser = cli.train_parser()

    args = parser.parse_args()

    train(**vars(args))
