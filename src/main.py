from experiments import backbone_sweep


def main():
    backbone_sweep(finetune_mode="head_only")


if __name__ == "__main__":
    main()
