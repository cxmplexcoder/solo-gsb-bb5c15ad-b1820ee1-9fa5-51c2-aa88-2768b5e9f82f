"""POST /startup-order 的端到端测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def base_payload(**overrides):
    """与 test_evaluate 相同的三层小树：

    source --s1(0.1Ω)--> a --s2(0.2Ω)--> b
                          |
                          +--s3(0.05Ω)--> c

    负载：a=2A, b=5A, c=8A
    安全基线：段电流 s1=15, s2=5, s3=8；节点电压 a=46.5, b=45.5, c=46.1
    """
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 40.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 20.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.2, "rated_current": 10.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.05, "rated_current": 15.0},
        ],
        "loads": [
            {"node": "a", "current": 2.0},
            {"node": "b", "current": 5.0},
            {"node": "c", "current": 8.0},
        ],
        "startup_loads": [],
    }
    payload.update(overrides)
    return payload


# ---------- 基本可行顺序 ----------


def test_single_load_feasible_with_step_metrics():
    # b 点新负载：稳态 2A、启动 4A
    # 启动瞬间：s1=19(.95), s2=9(.9), s3=8(.533)；电压 a=46.1, b=44.3, c=45.7
    payload = base_payload(
        startup_loads=[{"node": "b", "steady_current": 2.0, "startup_current": 4.0}]
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "feasible"
    assert data["order"] == [0]
    assert len(data["steps"]) == 1
    step = data["steps"][0]
    assert step["step"] == 1
    assert step["load_index"] == 0
    assert step["node"] == "b"
    assert step["steady_current"] == pytest.approx(2.0)
    assert step["startup_current"] == pytest.approx(4.0)
    assert step["min_voltage"] == pytest.approx(44.3)
    assert step["min_voltage_node"] == "b"
    assert step["max_loading_rate"] == pytest.approx(0.95)
    assert step["max_loading_segment_id"] == "s1"


def test_two_load_order_metrics_previous_load_at_steady_current():
    # s1 额定 25A：
    # 顺序 [0,1]：步 1 b 启动 10A -> s1=25 恰好满载；
    # 步 2 c 启动 6A，此时 b 已回落为稳态 4A -> s1=15+4+6=25 恰好满载。
    # 若此前投运负载仍按启动电流计入，步 2 将为 31A 越限。
    payload = base_payload(
        segments=[
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 25.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.2, "rated_current": 20.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.05, "rated_current": 20.0},
        ],
        startup_loads=[
            {"node": "b", "steady_current": 4.0, "startup_current": 10.0},
            {"node": "c", "steady_current": 3.0, "startup_current": 6.0},
        ],
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "feasible"
    assert data["order"] == [0, 1]

    s1, s2 = data["steps"]
    assert s1["min_voltage"] == pytest.approx(42.5)
    assert s1["min_voltage_node"] == "b"
    assert s1["max_loading_rate"] == pytest.approx(1.0)
    assert s1["max_loading_segment_id"] == "s1"

    # 步 2：s2=9(.45), s3=14(.7), s1=25(1.0)；电压 a=45.5, b=43.7, c=44.8
    assert s2["min_voltage"] == pytest.approx(43.7)
    assert s2["min_voltage_node"] == "b"
    assert s2["max_loading_rate"] == pytest.approx(1.0)
    assert s2["max_loading_segment_id"] == "s1"


def test_not_started_loads_do_not_count():
    # 三条负载均放 b 点、启动电流 1A；任意时刻只有已投运 / 正在投运的负载计入
    payload = base_payload(
        startup_loads=[
            {"node": "b", "steady_current": 1.0, "startup_current": 1.0},
            {"node": "b", "steady_current": 1.0, "startup_current": 1.0},
            {"node": "b", "steady_current": 1.0, "startup_current": 1.0},
        ]
    )
    data = client.post("/startup-order", json=payload).json()
    assert data["status"] == "feasible"
    assert data["order"] == [0, 1, 2]
    # s1 基线 15A、额定 20A：逐步仅 +1A -> 0.8, 0.85, 0.9（未投运负载不计入）
    rates = [s["max_loading_rate"] for s in data["steps"]]
    assert rates == [pytest.approx(0.8), pytest.approx(0.85), pytest.approx(0.9)]
    assert all(s["max_loading_segment_id"] == "s1" for s in data["steps"])


def test_all_orders_feasible_returns_lexicographically_smallest():
    payload = base_payload(
        startup_loads=[
            {"node": "b", "steady_current": 0.1, "startup_current": 0.2},
            {"node": "c", "steady_current": 0.1, "startup_current": 0.2},
            {"node": "a", "steady_current": 0.1, "startup_current": 0.2},
        ]
    )
    data = client.post("/startup-order", json=payload).json()
    assert data["status"] == "feasible"
    assert data["order"] == [0, 1, 2]
    assert [s["load_index"] for s in data["steps"]] == [0, 1, 2]
    assert [s["step"] for s in data["steps"]] == [1, 2, 3]


# ---------- 恰好触限允许 ----------


def test_exact_rated_current_and_exact_min_voltage_allowed():
    # b 启动 5A：s2=10 恰好额定、s1=20 恰好额定；
    # 电源 44V：s1 压降 2V -> a=42；s2 压降 2V -> b=40 恰好为最低允许电压。
    payload = base_payload(
        source_voltage=44.0,
        startup_loads=[{"node": "b", "steady_current": 5.0, "startup_current": 5.0}],
    )
    data = client.post("/startup-order", json=payload).json()
    assert data["status"] == "feasible"
    assert data["order"] == [0]
    step = data["steps"][0]
    assert step["min_voltage"] == pytest.approx(40.0)
    assert step["max_loading_rate"] == pytest.approx(1.0)


def test_loading_rate_tie_picks_first_input_segment():
    # 小电阻网络（忽略电压约束）：步 2 s1=31/31 与 s2=15/15 同时满载，取输入顺序首个 s1
    payload = {
        "source_voltage": 48.0,
        "min_voltage": 10.0,
        "segments": [
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.001, "rated_current": 31.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.001, "rated_current": 15.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.001, "rated_current": 20.0},
        ],
        "loads": [
            {"node": "a", "current": 2.0},
            {"node": "b", "current": 5.0},
            {"node": "c", "current": 8.0},
        ],
        "startup_loads": [
            {"node": "b", "steady_current": 8.0, "startup_current": 10.0},
            {"node": "c", "steady_current": 6.0, "startup_current": 10.0},
        ],
    }
    data = client.post("/startup-order", json=payload).json()
    # [0,1] 步 2 s1=15+8+10=33>31 不可行；[1,0] 步 2 s1=15+6+10=31 可行 -> 回溯得 [1,0]
    assert data["status"] == "feasible"
    assert data["order"] == [1, 0]

    first, second = data["steps"]
    assert first["load_index"] == 1
    assert first["min_voltage_node"] == "c"
    assert first["min_voltage"] == pytest.approx(47.957)
    assert first["max_loading_segment_id"] == "s3"  # 18/20 = 0.9 最大
    assert first["max_loading_rate"] == pytest.approx(0.9)

    assert second["load_index"] == 0
    assert second["min_voltage_node"] == "b"
    assert second["min_voltage"] == pytest.approx(47.954)
    assert second["max_loading_segment_id"] == "s1"  # s1、s2 同为 1.0，取输入顺序首个
    assert second["max_loading_rate"] == pytest.approx(1.0)


def test_node_level_min_voltage_coverage_respected():
    # c 点节点级最低允许电压 45.1V，放宽额定电流与全局阈值使其仅受节点电压约束
    # （该网络基线 c=45.7V）：c 启动 12A 经 s3（0.05Ω）与 s1（0.1Ω，兄弟支路同受）
    # 压降 12×0.15=1.8V，c=45.7-1.8=43.9 < 45.1 不可行；
    # 启动 4A 时 c=45.7-0.6=45.1 恰好触限，可行。
    payload = base_payload(
        segments=[
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 40.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.2, "rated_current": 40.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.05, "rated_current": 40.0},
        ],
        loads=[
            {"node": "a", "current": 2.0},
            {"node": "b", "current": 5.0},
            {"node": "c", "current": 8.0, "min_voltage": 45.1},
        ],
        startup_loads=[{"node": "c", "steady_current": 4.0, "startup_current": 12.0}],
    )
    data = client.post("/startup-order", json=payload).json()
    assert data["status"] == "infeasible"

    payload["startup_loads"] = [{"node": "c", "steady_current": 4.0, "startup_current": 4.0}]
    data = client.post("/startup-order", json=payload).json()
    assert data["status"] == "feasible"
    assert data["steps"][0]["min_voltage"] == pytest.approx(45.1)
    # b=45.5-4×0.1=45.1 与 c=45.7-4×0.15=45.1 并列，BFS 顺序根在前取 b
    assert data["steps"][0]["min_voltage_node"] == "b"


# ---------- 无解 ----------


def test_infeasible_when_first_step_always_over_rated():
    # b 启动 6A 即使第一步也使 s1=21 > 20，无任何可投运的第一步
    payload = base_payload(
        startup_loads=[{"node": "b", "steady_current": 6.0, "startup_current": 6.0}]
    )
    data = client.post("/startup-order", json=payload).json()
    assert data["status"] == "infeasible"
    assert data["order"] is None
    assert data["steps"] is None


def test_infeasible_when_only_second_step_fails():
    # 两条 b 点负载各稳态/启动 6A：s2 额定 16A，单投运 s2=11 可行，
    # 投第二条时 s2=5+6+6=17 > 16；两种顺序均失败 -> 无完整顺序
    payload = base_payload(
        segments=[
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.01, "rated_current": 40.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.01, "rated_current": 16.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.01, "rated_current": 40.0},
        ],
        startup_loads=[
            {"node": "b", "steady_current": 6.0, "startup_current": 6.0},
            {"node": "b", "steady_current": 6.0, "startup_current": 6.0},
        ],
    )
    data = client.post("/startup-order", json=payload).json()
    assert data["status"] == "infeasible"


def test_infeasible_due_to_voltage_dip():
    # 电源 43V：b 点基线 40.5V；新增启动电流 3A 经 s1+s2（0.3Ω）再降 0.9V -> 39.6V 欠压
    payload = base_payload(
        source_voltage=43.0,
        startup_loads=[{"node": "b", "steady_current": 3.0, "startup_current": 3.0}],
    )
    data = client.post("/startup-order", json=payload).json()
    assert data["status"] == "infeasible"


# ---------- 基线越限（400 baseline_violation） ----------


def test_baseline_over_current_returns_400():
    payload = base_payload(
        segments=[
            {"id": "s1", "parent": "source", "child": "a", "resistance": 0.1, "rated_current": 14.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.2, "rated_current": 10.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.05, "rated_current": 15.0},
        ],
        startup_loads=[{"node": "b", "steady_current": 1.0, "startup_current": 1.0}],
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "baseline_violation"
    assert any("s1" in d["message"] for d in error["details"])


def test_baseline_under_voltage_returns_400():
    payload = base_payload(
        segments=[
            {"id": "s1", "parent": "source", "child": "a", "resistance": 1.0, "rated_current": 20.0},
            {"id": "s2", "parent": "a", "child": "b", "resistance": 0.2, "rated_current": 10.0},
            {"id": "s3", "parent": "a", "child": "c", "resistance": 0.05, "rated_current": 15.0},
        ],
        startup_loads=[{"node": "b", "steady_current": 1.0, "startup_current": 1.0}],
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "baseline_violation"


# ---------- 输入校验 ----------


def test_startup_current_below_steady_rejected():
    payload = base_payload(
        startup_loads=[{"node": "b", "steady_current": 5.0, "startup_current": 4.0}]
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 422
    assert any("启动电流不能小于稳态电流" in d["message"] for d in resp.json()["error"]["details"])


def test_startup_current_equal_to_steady_accepted():
    payload = base_payload(
        startup_loads=[{"node": "b", "steady_current": 2.0, "startup_current": 2.0}]
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "feasible"


def test_empty_startup_loads_rejected():
    resp = client.post("/startup-order", json=base_payload())
    assert resp.status_code == 422
    assert any("不能为空" in d["message"] for d in resp.json()["error"]["details"])


def test_eleven_startup_loads_rejected():
    payload = base_payload(
        startup_loads=[
            {"node": "a", "steady_current": 0.0, "startup_current": 0.0} for _ in range(11)
        ]
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 422
    assert any("10" in d["message"] for d in resp.json()["error"]["details"])


def test_ten_startup_loads_accepted():
    payload = base_payload(
        startup_loads=[
            {"node": "a", "steady_current": 0.0, "startup_current": 0.0} for _ in range(10)
        ]
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["order"] == list(range(10))
    assert len(data["steps"]) == 10


def test_unknown_startup_node_rejected():
    payload = base_payload(
        startup_loads=[{"node": "ghost", "steady_current": 1.0, "startup_current": 1.0}]
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "invalid_input"
    assert any("ghost" in d["message"] for d in error["details"])


def test_source_node_startup_load_accepted():
    # source 节点负载不流经任何线段、不改变任何节点电压
    payload = base_payload(
        startup_loads=[{"node": "source", "steady_current": 100.0, "startup_current": 100.0}]
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "feasible"
    assert data["steps"][0]["max_loading_rate"] == pytest.approx(0.75)  # s1=15/20 与基线相同


def test_negative_current_rejected():
    payload = base_payload(
        startup_loads=[{"node": "b", "steady_current": -1.0, "startup_current": 1.0}]
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 422
    assert any("稳态电流" in d["message"] for d in resp.json()["error"]["details"])

    payload = base_payload(
        startup_loads=[{"node": "b", "steady_current": 1.0, "startup_current": -1.0}]
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 422
    assert any("启动电流" in d["message"] for d in resp.json()["error"]["details"])


def test_non_finite_current_rejected():
    # 以原始 JSON 发送 NaN（标准 json 不允许 NaN 出现在请求体中），由服务端 pydantic 拒绝
    import json as _json

    payload = base_payload(
        startup_loads=[{"node": "b", "steady_current": 1.0, "startup_current": 0.0}]
    )
    payload["startup_loads"][0]["startup_current"] = float("nan")
    resp = client.post(
        "/startup-order",
        content=_json.dumps(payload),  # Python json 输出非标准 NaN 字面量
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422

    payload = base_payload(
        startup_loads=[{"node": "b", "steady_current": 0.0, "startup_current": 1.0}]
    )
    payload["startup_loads"][0]["steady_current"] = float("inf")
    resp = client.post(
        "/startup-order",
        content=_json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_existing_tree_validation_still_applies():
    payload = base_payload(
        loads=[{"node": "ghost", "current": 1.0}],
        startup_loads=[{"node": "b", "steady_current": 1.0, "startup_current": 1.0}],
    )
    resp = client.post("/startup-order", json=payload)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_input"
