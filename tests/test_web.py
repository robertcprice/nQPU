"""Comprehensive tests for the QKD Network Planning API.

Uses FastAPI's TestClient (synchronous, no live server) to exercise every
endpoint, validate status codes, and verify response payloads.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from nqpu.web.qkd_api import _networks, app

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_networks():
    """Ensure each test starts with a clean network store."""
    _networks.clear()
    yield
    _networks.clear()


# ---------------------------------------------------------------------------
# Helper: create a network and return its ID
# ---------------------------------------------------------------------------


def _create_network(seed: int = 42) -> str:
    resp = client.post("/networks", json={"seed": seed})
    assert resp.status_code == 200
    return resp.json()["network_id"]


def _add_node(network_id: str, node_id: str, x: float = 0.0, y: float = 0.0):
    resp = client.post(
        f"/networks/{network_id}/nodes",
        json={"node_id": node_id, "x": x, "y": y, "is_trusted_relay": True},
    )
    assert resp.status_code == 200
    return resp.json()


def _add_link(network_id: str, node_a: str, node_b: str, error_rate: float = 0.02):
    resp = client.post(
        f"/networks/{network_id}/links",
        json={"node_a": node_a, "node_b": node_b, "error_rate": error_rate},
    )
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# 1. Network CRUD
# ---------------------------------------------------------------------------


def test_create_network():
    """POST /networks returns 200 with a valid network_id."""
    resp = client.post("/networks", json={"seed": 42})
    assert resp.status_code == 200
    data = resp.json()
    assert "network_id" in data
    assert isinstance(data["network_id"], str)
    assert len(data["network_id"]) > 0


def test_create_network_no_body():
    """POST /networks with no JSON body still succeeds (defaults)."""
    resp = client.post("/networks")
    assert resp.status_code == 200
    assert "network_id" in resp.json()


def test_get_network_not_found():
    """GET /networks/<bogus> returns 404."""
    resp = client.get("/networks/does-not-exist")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 2. Nodes
# ---------------------------------------------------------------------------


def test_add_node():
    """Adding a node returns the correct fields."""
    nid = _create_network()
    data = _add_node(nid, "alice", x=10.0, y=20.0)
    assert data["node_id"] == "alice"
    assert data["x"] == 10.0
    assert data["y"] == 20.0
    assert data["is_trusted_relay"] is True


def test_add_duplicate_node():
    """Adding the same node_id twice returns 409."""
    nid = _create_network()
    _add_node(nid, "alice")
    resp = client.post(
        f"/networks/{nid}/nodes",
        json={"node_id": "alice", "x": 0.0, "y": 0.0},
    )
    assert resp.status_code == 409


# ---------------------------------------------------------------------------
# 3. Links
# ---------------------------------------------------------------------------


def test_add_link():
    """Adding a link returns node endpoints and a distance."""
    nid = _create_network()
    _add_node(nid, "alice", x=0.0, y=0.0)
    _add_node(nid, "bob", x=3.0, y=4.0)
    data = _add_link(nid, "alice", "bob")
    assert data["node_a"] == "alice"
    assert data["node_b"] == "bob"
    assert data["distance_km"] == pytest.approx(5.0, abs=0.01)


def test_add_link_duplicate():
    """Adding the same link twice returns 409."""
    nid = _create_network()
    _add_node(nid, "alice")
    _add_node(nid, "bob")
    _add_link(nid, "alice", "bob")
    resp = client.post(
        f"/networks/{nid}/links",
        json={"node_a": "alice", "node_b": "bob"},
    )
    assert resp.status_code == 409


def test_add_link_missing_node():
    """Linking a non-existent node returns 404."""
    nid = _create_network()
    _add_node(nid, "alice")
    resp = client.post(
        f"/networks/{nid}/links",
        json={"node_a": "alice", "node_b": "ghost"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 4. Key establishment
# ---------------------------------------------------------------------------


def _setup_two_node_network(seed: int = 42) -> str:
    """Helper: create a network with alice--bob link."""
    nid = _create_network(seed=seed)
    _add_node(nid, "alice", x=0.0, y=0.0)
    _add_node(nid, "bob", x=10.0, y=0.0)
    _add_link(nid, "alice", "bob", error_rate=0.02)
    return nid


def test_establish_key_bb84():
    """BB84 key establishment produces a secure key."""
    nid = _setup_two_node_network()
    resp = client.post(
        f"/networks/{nid}/establish-key",
        json={"node_a": "alice", "node_b": "bob", "protocol": "BB84", "n_bits": 10000},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["protocol"] == "BB84"
    assert data["secure"] is True
    assert data["key_length"] > 0
    assert 0.0 <= data["qber"] <= 0.11
    assert isinstance(data["final_key_hex"], str)
    assert data["key_rate"] > 0.0


def test_establish_key_e91():
    """E91 key establishment produces a secure key."""
    nid = _setup_two_node_network()
    resp = client.post(
        f"/networks/{nid}/establish-key",
        json={"node_a": "alice", "node_b": "bob", "protocol": "E91", "n_bits": 10000},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["protocol"] == "E91"
    assert data["secure"] is True
    assert data["key_length"] > 0
    assert data["key_rate"] > 0.0


def test_establish_key_b92():
    """B92 key establishment produces a secure key."""
    nid = _setup_two_node_network()
    resp = client.post(
        f"/networks/{nid}/establish-key",
        json={"node_a": "alice", "node_b": "bob", "protocol": "B92", "n_bits": 10000},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["protocol"] == "B92"
    assert data["secure"] is True
    assert data["key_length"] > 0


def test_establish_key_relay():
    """Key establishment via a relay node succeeds."""
    nid = _create_network(seed=42)
    _add_node(nid, "alice", x=0.0, y=0.0)
    _add_node(nid, "relay", x=5.0, y=0.0)
    _add_node(nid, "bob", x=10.0, y=0.0)
    _add_link(nid, "alice", "relay", error_rate=0.02)
    _add_link(nid, "relay", "bob", error_rate=0.02)

    resp = client.post(
        f"/networks/{nid}/establish-key",
        json={
            "node_a": "alice",
            "node_b": "bob",
            "protocol": "BB84",
            "n_bits": 10000,
            "via_relay": ["relay"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "relay" in data["protocol"].lower() or "relay" in data["protocol"]
    assert data["secure"] is True
    assert data["key_length"] > 0


def test_establish_key_missing_link():
    """Key establishment on a missing link returns 404."""
    nid = _create_network()
    _add_node(nid, "alice")
    _add_node(nid, "bob")
    # No link added.
    resp = client.post(
        f"/networks/{nid}/establish-key",
        json={"node_a": "alice", "node_b": "bob"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 5. Topology generation
# ---------------------------------------------------------------------------


def _make_nodes(n: int, spacing: float = 10.0) -> list[dict]:
    return [
        {"node_id": f"n{i}", "x": i * spacing, "y": 0.0, "is_trusted_relay": True}
        for i in range(n)
    ]


def test_star_topology():
    """Star topology creates n-1 links with the centre connected to all."""
    nid = _create_network()
    nodes = _make_nodes(5)
    resp = client.post(
        f"/networks/{nid}/topology/star",
        json={
            "topology_type": "star",
            "nodes": nodes,
            "center_node_id": "n0",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 5
    assert len(data["links"]) == 4  # n-1 links


def test_line_topology():
    """Line topology creates n-1 links chained sequentially."""
    nid = _create_network()
    nodes = _make_nodes(4)
    resp = client.post(
        f"/networks/{nid}/topology/line",
        json={"topology_type": "line", "nodes": nodes},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 4
    assert len(data["links"]) == 3


def test_mesh_topology():
    """Mesh topology creates n*(n-1)/2 links."""
    nid = _create_network()
    n = 4
    nodes = _make_nodes(n)
    resp = client.post(
        f"/networks/{nid}/topology/mesh",
        json={"topology_type": "mesh", "nodes": nodes},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == n
    expected_links = n * (n - 1) // 2
    assert len(data["links"]) == expected_links


def test_unknown_topology():
    """Requesting an unknown topology type returns 400."""
    nid = _create_network()
    resp = client.post(
        f"/networks/{nid}/topology/ring",
        json={"topology_type": "ring", "nodes": _make_nodes(3)},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 6. Standalone simulation
# ---------------------------------------------------------------------------


def test_simulate_protocol():
    """Standalone BB84 simulation returns a valid result."""
    resp = client.post(
        "/simulate/protocol",
        json={
            "protocol": "BB84",
            "n_bits": 10000,
            "error_rate": 0.02,
            "loss_probability": 0.1,
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["protocol"] == "BB84"
    assert data["secure"] is True
    assert data["key_length"] > 0
    assert data["n_bits_sent"] == 10000


def test_simulate_protocol_with_eavesdropper():
    """Eavesdropper raises the QBER compared to a clean channel."""
    # Clean channel
    resp_clean = client.post(
        "/simulate/protocol",
        json={
            "protocol": "BB84",
            "n_bits": 10000,
            "error_rate": 0.01,
            "loss_probability": 0.05,
            "eavesdropper": False,
            "seed": 100,
        },
    )
    assert resp_clean.status_code == 200
    qber_clean = resp_clean.json()["qber"]

    # Eavesdropper present
    resp_eve = client.post(
        "/simulate/protocol",
        json={
            "protocol": "BB84",
            "n_bits": 10000,
            "error_rate": 0.01,
            "loss_probability": 0.05,
            "eavesdropper": True,
            "eavesdropper_rate": 1.0,
            "seed": 100,
        },
    )
    assert resp_eve.status_code == 200
    qber_eve = resp_eve.json()["qber"]

    # Eavesdropper should cause higher QBER.
    assert qber_eve > qber_clean


# ---------------------------------------------------------------------------
# 7. Network report
# ---------------------------------------------------------------------------


def test_network_report():
    """Report endpoint returns per-link analysis."""
    nid = _create_network(seed=42)
    _add_node(nid, "alice", x=0.0, y=0.0)
    _add_node(nid, "bob", x=5.0, y=0.0)
    _add_link(nid, "alice", "bob", error_rate=0.02)

    resp = client.get(f"/networks/{nid}/report")
    assert resp.status_code == 200
    data = resp.json()

    assert data["num_nodes"] == 2
    assert data["num_links"] == 1
    assert len(data["links"]) == 1

    link_info = data["links"][0]
    assert link_info["node_a"] in ("alice", "bob")
    assert link_info["node_b"] in ("alice", "bob")
    assert "qber" in link_info
    assert "key_rate" in link_info
    assert "secure" in link_info


# ---------------------------------------------------------------------------
# 8. Get network details round-trip
# ---------------------------------------------------------------------------


def test_get_network_details():
    """GET /networks/<id> returns all nodes and links that were added."""
    nid = _create_network()
    _add_node(nid, "alice", x=1.0, y=2.0)
    _add_node(nid, "bob", x=4.0, y=6.0)
    _add_link(nid, "alice", "bob")

    resp = client.get(f"/networks/{nid}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["network_id"] == nid
    assert len(data["nodes"]) == 2
    assert len(data["links"]) == 1

    node_ids = {n["node_id"] for n in data["nodes"]}
    assert node_ids == {"alice", "bob"}
