import os
import json
import tempfile
import pytest
import torch
import torch.nn as nn

from apa import APAConfig, APAManager, APALinear, APAConv2d
from apa.diagnostics import FrobeniusTelemetry


class SimpleSequentialModel(nn.Module):
    def __init__(self, config: APAConfig):
        super().__init__()
        self.fc1 = APALinear(32, 32, config=config)
        self.fc2 = APALinear(32, 32, config=config)
        self.fc3 = APALinear(32, 32, config=config)
        self.fc4 = APALinear(32, 10, config=config)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class MixedConvModel(nn.Module):
    def __init__(self, config: APAConfig):
        super().__init__()
        self.conv1 = APAConv2d(3, 16, kernel_size=3, padding=1, config=config)
        self.conv2 = APAConv2d(16, 16, kernel_size=3, padding=1, config=config)
        self.fc = APALinear(16 * 8 * 8, 10, config=config)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# 1. Test Inisialisasi & Weight Variance Disparity
# ---------------------------------------------------------------------------

def test_frobenius_initialization_balanced():
    """Test evaluating model weights at initialization with balanced scaling."""
    config = APAConfig(enable_frobenius_telemetry=True, device='cpu')
    model = SimpleSequentialModel(config)
    telemetry = FrobeniusTelemetry(config=config, model=model)

    res = telemetry.evaluate_at_initialization(model)
    assert res["step"] == 0
    assert res["event"] == "frobenius_initialization"
    assert res["num_layers"] == 4
    assert res["weight_disparity"] >= 1.0
    # Balanced default PyTorch init disparity should be small (< 100)
    assert res["weight_disparity"] < 1e3
    assert res["weight_disparity_warning"] is False


def test_frobenius_initialization_disparity_warning():
    """Test that extreme disparity at initialization triggers disparity_warning flag."""
    config = APAConfig(enable_frobenius_telemetry=True, device='cpu')
    model = SimpleSequentialModel(config)
    
    # Artificially inject extreme scale into fc4 weights (e.g. factor 1e6)
    with torch.no_grad():
        model.fc4.weight_master.mul_(1e6)
        if model.fc4.weight_work is not None:
            model.fc4.weight_work.mul_(1e6)

    telemetry = FrobeniusTelemetry(config=config, model=model)
    res = telemetry.evaluate_at_initialization(model)
    assert res["weight_disparity"] > 1e5
    assert res["weight_disparity_warning"] is True


# ---------------------------------------------------------------------------
# 2. Test Gradient Slope (Vanishing & Exploding)
# ---------------------------------------------------------------------------

def test_frobenius_gradient_slope_vanishing():
    """Slope > +0.2 indicates gradual vanishing gradient towards input (small at input, large at output)."""
    telemetry = FrobeniusTelemetry()
    # Sequence of 4 layers from input (l=1) to output (l=4)
    # Input has small gradient (1e-6), output has larger gradient (1e-1)
    mock_metrics = [
        {"name": "fc1", "weight_norm": 10.0, "grad_norm": 1e-6, "update_ratio": 1e-7, "level": 0},
        {"name": "fc2", "weight_norm": 10.0, "grad_norm": 1e-4, "update_ratio": 1e-5, "level": 0},
        {"name": "fc3", "weight_norm": 10.0, "grad_norm": 1e-2, "update_ratio": 1e-3, "level": 0},
        {"name": "fc4", "weight_norm": 10.0, "grad_norm": 1.0,  "update_ratio": 1e-1, "level": 0},
    ]
    res = telemetry.evaluate_instability(step=1, layer_metrics=mock_metrics)
    assert res["grad_slope"] > 0.2
    assert res["slope_status"] == "GRADUAL_VANISHING_GRADIENT"


def test_frobenius_gradient_slope_exploding():
    """Slope < -0.2 indicates gradual exploding gradient towards input (large at input, small at output)."""
    telemetry = FrobeniusTelemetry()
    # Sequence of 4 layers from input (l=1) to output (l=4)
    # Input has large gradient (100.0), output has small gradient (1e-4)
    mock_metrics = [
        {"name": "fc1", "weight_norm": 10.0, "grad_norm": 100.0, "update_ratio": 10.0, "level": 0},
        {"name": "fc2", "weight_norm": 10.0, "grad_norm": 1.0,   "update_ratio": 0.1,  "level": 0},
        {"name": "fc3", "weight_norm": 10.0, "grad_norm": 1e-2,  "update_ratio": 1e-3, "level": 0},
        {"name": "fc4", "weight_norm": 10.0, "grad_norm": 1e-4,  "update_ratio": 1e-5, "level": 0},
    ]
    res = telemetry.evaluate_instability(step=1, layer_metrics=mock_metrics)
    assert res["grad_slope"] < -0.2
    assert res["slope_status"] == "GRADUAL_EXPLODING_GRADIENT"


def test_frobenius_gradient_slope_stable():
    """Relatively uniform gradient norms yield STABLE status."""
    telemetry = FrobeniusTelemetry()
    mock_metrics = [
        {"name": "fc1", "weight_norm": 10.0, "grad_norm": 0.50, "update_ratio": 0.05, "level": 0},
        {"name": "fc2", "weight_norm": 10.0, "grad_norm": 0.48, "update_ratio": 0.048, "level": 0},
        {"name": "fc3", "weight_norm": 10.0, "grad_norm": 0.52, "update_ratio": 0.052, "level": 0},
        {"name": "fc4", "weight_norm": 10.0, "grad_norm": 0.51, "update_ratio": 0.051, "level": 0},
    ]
    res = telemetry.evaluate_instability(step=1, layer_metrics=mock_metrics)
    assert -0.2 <= res["grad_slope"] <= 0.2
    assert res["slope_status"] == "STABLE"


# ---------------------------------------------------------------------------
# 3. Test Layer Outlier Detection (Amplifier & Cliff)
# ---------------------------------------------------------------------------

def test_frobenius_layer_outlier_detection():
    """Detects amplifier (> 10^4) and cliff (< 10^-4) anomalies and tracks consecutive counts."""
    telemetry = FrobeniusTelemetry()

    # Step 1: fc2 has a sudden amplifier surge (ratio = 100000 / 1 = 10^5 > 1e4)
    # fc3 has a sudden cliff drop (ratio = 0.0001 / 100000 = 10^-9 < 1e-4)
    step1_metrics = [
        {"name": "fc1", "weight_norm": 10.0, "grad_norm": 1.0,     "update_ratio": 0.1,  "level": 0},
        {"name": "fc2", "weight_norm": 10.0, "grad_norm": 100000.0,"update_ratio": 1e4,  "level": 0},
        {"name": "fc3", "weight_norm": 10.0, "grad_norm": 0.0001,  "update_ratio": 1e-5, "level": 0},
        {"name": "fc4", "weight_norm": 10.0, "grad_norm": 0.0001,  "update_ratio": 1e-5, "level": 0},
    ]
    res1 = telemetry.evaluate_instability(step=1, layer_metrics=step1_metrics)
    outliers1 = {o["layer"]: o for o in res1["outliers"]}

    assert "fc2" in outliers1
    assert outliers1["fc2"]["type"] == "ANOMALY_AMPLIFIER"
    assert outliers1["fc2"]["consecutive"] == 1
    assert outliers1["fc2"]["is_persistent"] is False

    assert "fc3" in outliers1
    assert outliers1["fc3"]["type"] == "ANOMALY_CLIFF"
    assert outliers1["fc3"]["consecutive"] == 1

    # Step 2: fc2 remains an amplifier in the next step -> consecutive reaches 2 -> persistent!
    res2 = telemetry.evaluate_instability(step=2, layer_metrics=step1_metrics)
    outliers2 = {o["layer"]: o for o in res2["outliers"]}
    assert outliers2["fc2"]["consecutive"] == 2
    assert outliers2["fc2"]["is_persistent"] is True

    # Check query helper
    info = telemetry.get_module_outlier_info("fc2")
    assert info is not None
    assert info["is_persistent"] is True

    # Step 3: fc2 returns to normal -> consecutive resets to 0
    normal_metrics = [
        {"name": "fc1", "weight_norm": 10.0, "grad_norm": 1.0, "update_ratio": 0.1, "level": 0},
        {"name": "fc2", "weight_norm": 10.0, "grad_norm": 1.0, "update_ratio": 0.1, "level": 0},
        {"name": "fc3", "weight_norm": 10.0, "grad_norm": 1.0, "update_ratio": 0.1, "level": 0},
        {"name": "fc4", "weight_norm": 10.0, "grad_norm": 1.0, "update_ratio": 0.1, "level": 0},
    ]
    res3 = telemetry.evaluate_instability(step=3, layer_metrics=normal_metrics)
    assert len(res3["outliers"]) == 0
    assert telemetry.get_module_outlier_info("fc2") is None


# ---------------------------------------------------------------------------
# 4. Test JSONL Logging & Type Serialization
# ---------------------------------------------------------------------------

def test_frobenius_jsonl_serialization():
    """Verify JSONL writing and that all data types are valid native Python types (no PyTorch/NumPy serialization crashes)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "frobenius_test.jsonl")
        telemetry = FrobeniusTelemetry(log_file=log_file)

        mock_metrics = [
            {"name": "fc1", "weight_norm": float(torch.tensor(5.234)), "grad_norm": float(torch.tensor(0.123)), "update_ratio": 0.001, "level": 0},
            {"name": "fc2", "weight_norm": float(torch.tensor(10.567)), "grad_norm": float(torch.tensor(0.456)), "update_ratio": 0.002, "level": 1},
        ]
        eval_res = telemetry.evaluate_instability(step=4, layer_metrics=mock_metrics)
        telemetry.log_step(step=4, eval_result=eval_res)

        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert record["event"] == "frobenius_telemetry"
            assert record["step"] == 4
            assert "grad_slope" in record
            assert len(record["layers"]) == 2
            assert record["layers"][0]["name"] == "fc1"


# ---------------------------------------------------------------------------
# 5. Test Integration with APAManager & Optional Flag
# ---------------------------------------------------------------------------

def test_frobenius_optional_disabled_by_default():
    """Verify that Frobenius telemetry is strictly OPTIONAL and disabled by default."""
    config = APAConfig(enable_frobenius_telemetry=False, device='cpu')
    model = SimpleSequentialModel(config)
    manager = APAManager(model, config=config)

    assert manager.frobenius_telemetry is None
    # Training step should run without creating any frobenius log
    x = torch.randn(4, 32)
    y = model(x).sum()
    y.backward()
    accepted = manager.post_backward_sync_and_eval()
    assert accepted is True
    assert getattr(config, "frobenius_log_file", None) is None


def test_frobenius_manager_lifecycle_and_escalation_link():
    """Verify full APAManager lifecycle with Frobenius telemetry enabled, including escalation context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        apa_log = os.path.join(tmpdir, "apa_run.jsonl")
        frob_log = os.path.join(tmpdir, "apa_run_frobenius.jsonl")

        config = APAConfig(
            enable_frobenius_telemetry=True,
            frobenius_check_interval=1,
            log_file=apa_log,
            frobenius_log_file=frob_log,
            device='cpu'
        )
        model = SimpleSequentialModel(config)
        manager = APAManager(model, config=config)

        assert manager.frobenius_telemetry is not None
        # Step 0 initialization log should exist
        assert os.path.exists(frob_log)

        # Step 1: Normal forward & backward
        manager.pre_step()
        x = torch.randn(4, 32)
        out = model(x).sum()
        out.backward()
        manager.post_backward_sync_and_eval()

        # Check frobenius log has entries
        with open(frob_log, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f.readlines()]
            assert len(lines) >= 2
            assert lines[0]["event"] == "frobenius_initialization"
            assert lines[1]["event"] == "frobenius_telemetry"

        # Simulate persistent outlier anomaly on fc2
        manager.frobenius_telemetry.consecutive_outliers["fc2"] = 3
        manager.frobenius_telemetry.last_outliers["fc2"] = {
            "layer": "fc2",
            "type": "ANOMALY_AMPLIFIER",
            "ratio": 50000.0,
            "consecutive": 3,
            "is_persistent": True
        }

        # Trigger escalation on fc2
        manager._escalate_module("fc2", model.fc2, "OVERFLOW", trigger_value=1000.0)

        # Verify that apa_run.jsonl records the frobenius anomaly context
        assert os.path.exists(apa_log)
        with open(apa_log, "r", encoding="utf-8") as f:
            apa_records = [json.loads(l) for l in f.readlines()]
            escalations = [r for r in apa_records if r.get("event") == "escalation"]
            assert len(escalations) == 1
            esc = escalations[0]
            assert esc["module"] == "fc2"
            assert "frobenius_anomaly" in esc
            assert esc["frobenius_anomaly"]["type"] == "ANOMALY_AMPLIFIER"
