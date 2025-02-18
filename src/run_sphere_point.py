import pathlib
import sys
import argparse
import pickle

sys.path.append(r'C:\Users\Vladimir\PycharmProjects\EEGPatternRecognition\src')
import run_experiment

def run_point(exp_idx, point_idx):
    print(f"Running for: {exp_idx=}\t{point_idx=}")

    base_dir = pathlib.Path(rf'C:\Users\Vladimir\PycharmProjects\EEGPatternRecognition\dumps\view_on_sphere\exp_{exp_idx}')
    cfg_path = base_dir / 'cfgs' / f'point_{point_idx}.yaml'
    cfg_path = str(cfg_path)

    train_res = run_experiment.run(cfg_path)
    
    with open(base_dir / 'results' / f'point_{point_idx}.bin', 'wb') as f:
        pickle.dump(train_res, file=f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_idx')
    parser.add_argument('--point_idx')

    args = parser.parse_args()
    exp_idx = args.exp_idx
    point_idx = args.point_idx

    run_point(exp_idx, point_idx)


if __name__ == '__main__':
    main()
