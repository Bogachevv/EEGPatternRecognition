import sys
sys.path.append('./src')

from src.run_experimet import run

def main():
    if len(sys.argv) == 1:
        print('Usage:')
        print('python3 run.py confg_path.yaml')
        exit(1)

    cfg_path = sys.argv[1]
    run(cfg_path=cfg_path)


if __name__ == '__main__':
    main()
