from fairlens.cli import (
    build_parser,
)


def test_parser():

    parser = build_parser()

    args = parser.parse_args([
        "audit",
        "--data",
        "adult.csv",
        "--target",
        "income",
        "--prediction",
        "prediction",
        "--protected",
        "sex",
    ])

    assert args.command == "audit"

    assert args.target == "income"

    assert args.protected == "sex"