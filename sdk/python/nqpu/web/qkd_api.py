"""QKD Network Planning API.

FastAPI application providing REST endpoints for quantum key distribution
network planning, topology generation, and protocol simulation.  Networks
are stored in-memory (one process, no persistence) -- suitable for demos,
integration testing, and lightweight SaaS deployments behind a gateway.

Endpoints
---------
POST   /networks                              Create an empty network.
GET    /networks/{id}                         Retrieve network details.
POST   /networks/{id}/nodes                   Add a node.
POST   /networks/{id}/links                   Add a quantum link.
POST   /networks/{id}/establish-key           Run QKD between two nodes.
POST   /networks/{id}/topology/{type}         Generate a standard topology.
GET    /networks/{id}/report                  Full per-link key-rate report.
POST   /simulate/protocol                     Standalone protocol simulation.
"""

from __future__ import annotations

import math
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nqpu.qkd import (
    BB84Protocol,
    B92Protocol,
    E91Protocol,
    EavesdropperConfig,
    QKDNetwork,
    QKDNode,
    QuantumChannel,
)

# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class NodeCreate(BaseModel):
    """Payload for adding a node to a network."""

    node_id: str
    x: float = 0.0
    y: float = 0.0
    is_trusted_relay: bool = True


class LinkCreate(BaseModel):
    """Payload for adding a quantum link between two nodes."""

    node_a: str
    node_b: str
    error_rate: float = 0.02


class EstablishKeyRequest(BaseModel):
    """Parameters for a QKD key-establishment run."""

    node_a: str
    node_b: str
    protocol: str = "BB84"  # BB84, E91, B92
    n_bits: int = 10000
    via_relay: list[str] | None = None  # intermediate relay node IDs


class TopologyRequest(BaseModel):
    """Specification for generating a standard network topology."""

    topology_type: str  # "star", "line", "mesh"
    nodes: list[NodeCreate]
    center_node_id: str | None = None  # required for star topology


class ProtocolSimRequest(BaseModel):
    """Standalone protocol simulation (no network required)."""

    protocol: str = "BB84"
    n_bits: int = 10000
    error_rate: float = 0.02
    loss_probability: float = 0.1
    eavesdropper: bool = False
    eavesdropper_rate: float = 1.0
    seed: int | None = None


class NetworkCreateRequest(BaseModel):
    """Optional body when creating a new network."""

    seed: int | None = 42


# -- Responses ---------------------------------------------------------------


class NodeResponse(BaseModel):
    """Serialised representation of a QKD node."""

    node_id: str
    x: float
    y: float
    is_trusted_relay: bool


class LinkResponse(BaseModel):
    """Serialised representation of a quantum link."""

    node_a: str
    node_b: str
    distance_km: float


class KeyResult(BaseModel):
    """Result of a single QKD protocol execution."""

    protocol: str
    n_bits_sent: int
    secure: bool
    qber: float
    final_key_hex: str
    key_length: int
    key_rate: float


class NetworkResponse(BaseModel):
    """Top-level network state."""

    network_id: str
    nodes: list[NodeResponse]
    links: list[LinkResponse]


class NetworkReport(BaseModel):
    """Comprehensive per-link analysis of a network."""

    network_id: str
    num_nodes: int
    num_links: int
    nodes: list[NodeResponse]
    links: list[dict]  # per-link key rate analysis
    topology_type: str


# ---------------------------------------------------------------------------
# In-memory storage
# ---------------------------------------------------------------------------

_networks: dict[str, dict] = {}
# Each entry:
#   {
#       "network": QKDNetwork,
#       "nodes": list[NodeResponse],
#       "links": list[LinkResponse],
#       "seed": int | None,
#   }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _key_bits_to_hex(key: list[int]) -> str:
    """Convert a list of binary ints (0/1) to a hex string."""
    if not key:
        return ""
    bitstring = "".join(str(b) for b in key)
    return hex(int(bitstring, 2))[2:]


def _get_network(network_id: str) -> dict:
    """Retrieve a network entry or raise 404."""
    if network_id not in _networks:
        raise HTTPException(status_code=404, detail=f"Network {network_id!r} not found")
    return _networks[network_id]


def _make_key_result(result, protocol_name: str | None = None) -> KeyResult:
    """Build a ``KeyResult`` from a ``QKDResult``."""
    proto = protocol_name or result.protocol
    return KeyResult(
        protocol=proto,
        n_bits_sent=result.n_bits_sent,
        secure=result.secure,
        qber=result.qber,
        final_key_hex=_key_bits_to_hex(result.final_key),
        key_length=len(result.final_key),
        key_rate=len(result.final_key) / max(1, result.n_bits_sent),
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="nQPU QKD Network Planner",
    description="Quantum Key Distribution network planning and simulation API",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/networks", response_model=dict)
def create_network(body: NetworkCreateRequest | None = None):
    """Create a new empty QKD network.

    Optionally pass ``{"seed": 42}`` for deterministic protocol runs.
    """
    seed = body.seed if body is not None else 42
    network_id = str(uuid.uuid4())
    net = QKDNetwork(seed=seed)
    _networks[network_id] = {
        "network": net,
        "nodes": [],
        "links": [],
        "seed": seed,
    }
    return {"network_id": network_id}


@app.get("/networks/{network_id}", response_model=NetworkResponse)
def get_network(network_id: str):
    """Return node and link details for a network."""
    entry = _get_network(network_id)
    return NetworkResponse(
        network_id=network_id,
        nodes=entry["nodes"],
        links=entry["links"],
    )


@app.post("/networks/{network_id}/nodes", response_model=NodeResponse)
def add_node(network_id: str, body: NodeCreate):
    """Add a node to an existing network."""
    entry = _get_network(network_id)
    net: QKDNetwork = entry["network"]

    # Check for duplicates before hitting the library (cleaner error).
    if body.node_id in net._nodes:
        raise HTTPException(
            status_code=409,
            detail=f"Node {body.node_id!r} already exists in network {network_id!r}",
        )

    node = QKDNode(
        node_id=body.node_id,
        position=(body.x, body.y),
        is_trusted_relay=body.is_trusted_relay,
    )
    net.add_node(node)

    resp = NodeResponse(
        node_id=body.node_id,
        x=body.x,
        y=body.y,
        is_trusted_relay=body.is_trusted_relay,
    )
    entry["nodes"].append(resp)
    return resp


@app.post("/networks/{network_id}/links", response_model=LinkResponse)
def add_link(network_id: str, body: LinkCreate):
    """Add a quantum link between two existing nodes."""
    entry = _get_network(network_id)
    net: QKDNetwork = entry["network"]

    # Validate nodes exist.
    for nid in (body.node_a, body.node_b):
        if nid not in net._nodes:
            raise HTTPException(
                status_code=404,
                detail=f"Node {nid!r} not found in network {network_id!r}",
            )

    link_key = tuple(sorted([body.node_a, body.node_b]))
    if link_key in net._links:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Link between {body.node_a!r} and {body.node_b!r} "
                f"already exists in network {network_id!r}"
            ),
        )

    channel = QuantumChannel(error_rate=body.error_rate)
    net.add_link(body.node_a, body.node_b, channel=channel)

    distance = net._links[link_key].distance_km

    resp = LinkResponse(
        node_a=body.node_a,
        node_b=body.node_b,
        distance_km=distance,
    )
    entry["links"].append(resp)
    return resp


@app.post("/networks/{network_id}/establish-key", response_model=KeyResult)
def establish_key(network_id: str, body: EstablishKeyRequest):
    """Run a QKD protocol on a direct link or relay path."""
    entry = _get_network(network_id)
    net: QKDNetwork = entry["network"]
    seed = entry.get("seed")

    # ---- Relay path --------------------------------------------------------
    if body.via_relay:
        path = [body.node_a, *body.via_relay, body.node_b]
        result = net.establish_key_via_relay(path, n_bits=body.n_bits)
        return _make_key_result(result, protocol_name=body.protocol + "-relay")

    # ---- Direct link -------------------------------------------------------
    link_key = tuple(sorted([body.node_a, body.node_b]))
    if link_key not in net._links:
        raise HTTPException(
            status_code=404,
            detail=f"No direct link between {body.node_a!r} and {body.node_b!r}",
        )

    protocol_name = body.protocol.upper()

    if protocol_name == "BB84":
        result = net.establish_key(body.node_a, body.node_b, n_bits=body.n_bits)
        return _make_key_result(result)

    # For E91 / B92 we grab the link's channel and run the protocol directly.
    link = net._links[link_key]
    proto_seed = net._next_seed()

    if protocol_name == "E91":
        protocol = E91Protocol(seed=proto_seed)
        result = protocol.generate_key(n_pairs=body.n_bits, channel=link.channel)
    elif protocol_name == "B92":
        protocol = B92Protocol(seed=proto_seed)
        result = protocol.generate_key(n_bits=body.n_bits, channel=link.channel)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown protocol {body.protocol!r}. Supported: BB84, E91, B92.",
        )

    return _make_key_result(result)


@app.post(
    "/networks/{network_id}/topology/{topology_type}",
    response_model=NetworkResponse,
)
def generate_topology(network_id: str, topology_type: str, body: TopologyRequest):
    """Replace the network with a generated topology (star / line / mesh)."""
    entry = _get_network(network_id)
    seed = entry.get("seed")

    qkd_nodes = [
        QKDNode(
            node_id=n.node_id,
            position=(n.x, n.y),
            is_trusted_relay=n.is_trusted_relay,
        )
        for n in body.nodes
    ]
    node_map = {n.node_id: n for n in qkd_nodes}

    topo = topology_type.lower()
    if topo == "star":
        if body.center_node_id is None:
            raise HTTPException(
                status_code=400,
                detail="center_node_id is required for star topology",
            )
        if body.center_node_id not in node_map:
            raise HTTPException(
                status_code=400,
                detail=f"center_node_id {body.center_node_id!r} not in provided nodes",
            )
        center = node_map[body.center_node_id]
        leaves = [n for n in qkd_nodes if n.node_id != body.center_node_id]
        new_net = QKDNetwork.star_topology(center, leaves, seed=seed)

    elif topo == "line":
        new_net = QKDNetwork.line_topology(qkd_nodes, seed=seed)

    elif topo == "mesh":
        new_net = QKDNetwork.mesh_topology(qkd_nodes, seed=seed)

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown topology {topology_type!r}. Supported: star, line, mesh.",
        )

    # Rebuild bookkeeping.
    node_responses = [
        NodeResponse(
            node_id=n.node_id,
            x=n.position[0],
            y=n.position[1],
            is_trusted_relay=n.is_trusted_relay,
        )
        for n in new_net.get_nodes()
    ]

    link_responses = []
    for lk, link_obj in new_net._links.items():
        link_responses.append(
            LinkResponse(
                node_a=link_obj.node_a_id,
                node_b=link_obj.node_b_id,
                distance_km=link_obj.distance_km,
            )
        )

    _networks[network_id] = {
        "network": new_net,
        "nodes": node_responses,
        "links": link_responses,
        "seed": seed,
    }

    return NetworkResponse(
        network_id=network_id,
        nodes=node_responses,
        links=link_responses,
    )


@app.post("/simulate/protocol", response_model=KeyResult)
def simulate_protocol(body: ProtocolSimRequest):
    """Run a protocol simulation on an ad-hoc channel (no network needed)."""
    eve_cfg = None
    if body.eavesdropper:
        eve_cfg = EavesdropperConfig(interception_rate=body.eavesdropper_rate)

    channel = QuantumChannel(
        error_rate=body.error_rate,
        loss_probability=body.loss_probability,
        eavesdropper=eve_cfg,
    )

    seed = body.seed if body.seed is not None else 42
    protocol_name = body.protocol.upper()

    if protocol_name == "BB84":
        proto = BB84Protocol(seed=seed)
        result = proto.generate_key(n_bits=body.n_bits, channel=channel)
    elif protocol_name == "E91":
        proto = E91Protocol(seed=seed)
        result = proto.generate_key(n_pairs=body.n_bits, channel=channel)
    elif protocol_name == "B92":
        proto = B92Protocol(seed=seed)
        result = proto.generate_key(n_bits=body.n_bits, channel=channel)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown protocol {body.protocol!r}. Supported: BB84, E91, B92.",
        )

    return _make_key_result(result)


@app.get("/networks/{network_id}/report", response_model=NetworkReport)
def network_report(network_id: str):
    """Generate a full per-link key-rate analysis report."""
    entry = _get_network(network_id)
    net: QKDNetwork = entry["network"]

    link_analyses: list[dict] = []
    for lk, link_obj in net._links.items():
        try:
            result = net.establish_key(link_obj.node_a_id, link_obj.node_b_id)
        except Exception:
            # If key establishment fails (e.g. too few bits), record failure.
            link_analyses.append(
                {
                    "node_a": link_obj.node_a_id,
                    "node_b": link_obj.node_b_id,
                    "distance_km": link_obj.distance_km,
                    "qber": None,
                    "key_rate": 0.0,
                    "key_length": 0,
                    "secure": False,
                }
            )
            continue

        link_analyses.append(
            {
                "node_a": link_obj.node_a_id,
                "node_b": link_obj.node_b_id,
                "distance_km": link_obj.distance_km,
                "qber": result.qber,
                "key_rate": len(result.final_key) / max(1, result.n_bits_sent),
                "key_length": len(result.final_key),
                "secure": result.secure,
            }
        )

    # Guess topology type from link count.
    n = len(net._nodes)
    n_links = len(net._links)
    if n_links == n * (n - 1) // 2 and n > 2:
        topo = "mesh"
    elif n_links == n - 1:
        # Could be star or line -- heuristic: star has one node with degree n-1.
        max_degree = max(len(adj) for adj in net._adjacency.values()) if net._adjacency else 0
        topo = "star" if max_degree == n - 1 else "line"
    else:
        topo = "custom"

    return NetworkReport(
        network_id=network_id,
        num_nodes=n,
        num_links=n_links,
        nodes=entry["nodes"],
        links=link_analyses,
        topology_type=topo,
    )
