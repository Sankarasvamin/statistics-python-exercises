import pandas as pd
import numpy as np


def main():
    print("[INFO] script started")

    print("[INFO] pandas version:", pd.__version__)
    print("[INFO] numpy version:", np.__version__)

    x = 6
    print("[RESULT] x =", x)

    print("[INFO] script finished")


if __name__ == "__main__":
    main()

import fastapi