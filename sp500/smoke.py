"""Run pretrained inference on the checked-in upstream regression fixture."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import Kronos, KronosPredictor, KronosTokenizer


def main() -> None:
    """Download public small weights and assert a finite CPU forecast."""
    torch.set_num_threads(4)
    torch.manual_seed(42)
    fixture = Path(__file__).resolve().parents[1] / "tests/data/regression_input.csv"
    data = pd.read_csv(fixture, parse_dates=["timestamps"])
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base").eval()
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").eval()
    predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
    result = predictor.predict(
        data.iloc[:64], data.timestamps.iloc[:64].reset_index(drop=True),
        data.timestamps.iloc[64:66].reset_index(drop=True), pred_len=2, verbose=False,
    )
    assert result.shape == (2, 6) and np.isfinite(result.to_numpy()).all()
    print(result.to_string())
    print("Pretrained CPU smoke test passed.")


if __name__ == "__main__":
    main()