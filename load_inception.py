from calc_inception import load_patched_inception_v3
from time import time

if __name__ == "__main__":
    s = time()
    model = load_patched_inception_v3()
    model.eval()
    print(f"It takes {time() - s} sec to load inception model")