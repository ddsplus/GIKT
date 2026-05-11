from infer import build_parser, evaluate


def main():
    parser = build_parser()
    parser.set_defaults(noise_levels="0,0.1,0.2,0.3,0.4,0.5")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()