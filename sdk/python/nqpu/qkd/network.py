"""Multi-node QKD network simulation.

Models a network of quantum nodes connected by fibre-optic links,
supporting:
  - Direct point-to-point key establishment via BB84
  - Trusted relay key chaining for nodes without direct quantum links
  - Distance-dependent fibre loss on each link
  - Common network topologies (star, line, mesh)

In a trusted-relay network, intermediate nodes establish pairwise keys
with their neighbours and XOR-chain them to extend the key across
multiple hops. This requires trusting the relay nodes.

References:
    - Peev et al., New J. Phys. 11, 075001 (2009) [SECOQC network]
    - Sasaki et al., Opt. Express 19, 10387 (2011) [Tokyo QKD network]
    - Chen et al., Nature 589, 214 (2021) [integrated space-ground QKD]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .bb84 import BB84Protocol, QKDResult
from .channel import QuantumChannel


@dataclass
class QKDNode:
    """A node in a QKD network.

    Parameters
    ----------
    node_id : str
        Unique identifier for the node.
    position : tuple[float, float]
        (x, y) coordinates in kilometres for distance calculations.
    is_trusted_relay : bool
        Whether this node can act as a trusted relay.
    """

    node_id: str
    position: Tuple[float, float] = (0.0, 0.0)
    is_trusted_relay: bool = True

    def distance_to(self, other: "QKDNode") -> float:
        """Compute Euclidean distance to another node in km.

        Parameters
        ----------
        other : QKDNode
            The other node.

        Returns
        -------
        float
            Distance in kilometres.
        """
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return math.sqrt(dx * dx + dy * dy)


@dataclass
class _Link:
    """Internal: a quantum link between two nodes."""

    node_a_id: str
    node_b_id: str
    distance_km: float
    channel: QuantumChannel


class QKDNetwork:
    """Multi-node QKD network with topology management and key establishment.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility across all protocol runs.
    attenuation_db_per_km : float
        Default fibre attenuation for automatically-configured links.
    n_bits_per_link : int
        Number of qubits to send per BB84 key establishment on each link.
    channel_error_rate : float
        Base channel error rate for all links.

    Examples
    --------
    >>> from nqpu.qkd import QKDNode, QKDNetwork
    >>> net = QKDNetwork(seed=42)
    >>> net.add_node(QKDNode("alice", (0, 0)))
    >>> net.add_node(QKDNode("relay", (50, 0)))
    >>> net.add_node(QKDNode("bob", (100, 0)))
    >>> net.add_link("alice", "relay")
    >>> net.add_link("relay", "bob")
    >>> result = net.establish_key_via_relay(["alice", "relay", "bob"])
    >>> assert result.secure
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        attenuation_db_per_km: float = 0.2,
        n_bits_per_link: int = 5000,
        channel_error_rate: float = 0.02,
    ) -> None:
        self._nodes: Dict[str, QKDNode] = {}
        self._links: Dict[Tuple[str, str], _Link] = {}
        self._adjacency: Dict[str, Set[str]] = {}
        self.seed = seed
        self._seed_counter = 0
        self.attenuation_db_per_km = attenuation_db_per_km
        self.n_bits_per_link = n_bits_per_link
        self.channel_error_rate = channel_error_rate

    # ------------------------------------------------------------------
    # Topology management
    # ------------------------------------------------------------------

    def add_node(self, node: QKDNode) -> None:
        """Add a node to the network.

        Parameters
        ----------
        node : QKDNode
            Node to add.

        Raises
        ------
        ValueError
            If a node with the same ID already exists.
        """
        if node.node_id in self._nodes:
            raise ValueError(f"Node {node.node_id!r} already exists")
        self._nodes[node.node_id] = node
        self._adjacency[node.node_id] = set()

    def add_link(
        self,
        node_a_id: str,
        node_b_id: str,
        channel: Optional[QuantumChannel] = None,
    ) -> None:
        """Add a quantum link between two nodes.

        If no channel is specified, one is automatically created with
        distance-dependent fibre loss computed from node positions.

        Parameters
        ----------
        node_a_id : str
            First node ID.
        node_b_id : str
            Second node ID.
        channel : QuantumChannel, optional
            Custom channel configuration. If None, auto-configured.

        Raises
        ------
        ValueError
            If either node does not exist or the link already exists.
        """
        if node_a_id not in self._nodes:
            raise ValueError(f"Node {node_a_id!r} not found")
        if node_b_id not in self._nodes:
            raise ValueError(f"Node {node_b_id!r} not found")

        link_key = tuple(sorted([node_a_id, node_b_id]))
        if link_key in self._links:
            raise ValueError(
                f"Link between {node_a_id!r} and {node_b_id!r} already exists"
            )

        node_a = self._nodes[node_a_id]
        node_b = self._nodes[node_b_id]
        distance = node_a.distance_to(node_b)

        if channel is None:
            channel = QuantumChannel(
                error_rate=self.channel_error_rate,
                distance_km=distance,
                attenuation_db_per_km=self.attenuation_db_per_km,
            )

        link = _Link(
            node_a_id=node_a_id,
            node_b_id=node_b_id,
            distance_km=distance,
            channel=channel,
        )
        self._links[link_key] = link
        self._adjacency[node_a_id].add(node_b_id)
        self._adjacency[node_b_id].add(node_a_id)

    def get_nodes(self) -> List[QKDNode]:
        """Return all nodes in the network."""
        return list(self._nodes.values())

    def get_neighbors(self, node_id: str) -> Set[str]:
        """Return the set of node IDs directly connected to a node."""
        return self._adjacency.get(node_id, set())

    # ------------------------------------------------------------------
    # Topology generators
    # ------------------------------------------------------------------

    @classmethod
    def star_topology(
        cls,
        center: QKDNode,
        leaves: List[QKDNode],
        seed: Optional[int] = None,
        **kwargs,
    ) -> "QKDNetwork":
        """Create a star-topology network.

        Parameters
        ----------
        center : QKDNode
            Central hub node.
        leaves : list[QKDNode]
            Leaf nodes connected to the hub.
        seed : int, optional
            Random seed.

        Returns
        -------
        QKDNetwork
            Configured star network.
        """
        net = cls(seed=seed, **kwargs)
        net.add_node(center)
        for leaf in leaves:
            net.add_node(leaf)
            net.add_link(center.node_id, leaf.node_id)
        return net

    @classmethod
    def line_topology(
        cls,
        nodes: List[QKDNode],
        seed: Optional[int] = None,
        **kwargs,
    ) -> "QKDNetwork":
        """Create a line-topology network.

        Parameters
        ----------
        nodes : list[QKDNode]
            Nodes in order from one end to the other.
        seed : int, optional
            Random seed.

        Returns
        -------
        QKDNetwork
            Configured line network.
        """
        net = cls(seed=seed, **kwargs)
        for node in nodes:
            net.add_node(node)
        for i in range(len(nodes) - 1):
            net.add_link(nodes[i].node_id, nodes[i + 1].node_id)
        return net

    @classmethod
    def mesh_topology(
        cls,
        nodes: List[QKDNode],
        seed: Optional[int] = None,
        **kwargs,
    ) -> "QKDNetwork":
        """Create a fully-connected mesh network.

        Parameters
        ----------
        nodes : list[QKDNode]
            All nodes to interconnect.
        seed : int, optional
            Random seed.

        Returns
        -------
        QKDNetwork
            Configured mesh network.
        """
        net = cls(seed=seed, **kwargs)
        for node in nodes:
            net.add_node(node)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                net.add_link(nodes[i].node_id, nodes[j].node_id)
        return net

    # ------------------------------------------------------------------
    # Key establishment
    # ------------------------------------------------------------------

    def establish_key(
        self,
        node_a_id: str,
        node_b_id: str,
        n_bits: Optional[int] = None,
    ) -> QKDResult:
        """Establish a key between two directly connected nodes using BB84.

        Parameters
        ----------
        node_a_id : str
            First node ID.
        node_b_id : str
            Second node ID.
        n_bits : int, optional
            Number of qubits to send. Defaults to ``self.n_bits_per_link``.

        Returns
        -------
        QKDResult
            Result of the BB84 protocol run.

        Raises
        ------
        ValueError
            If the nodes are not directly connected.
        """
        link_key = tuple(sorted([node_a_id, node_b_id]))
        if link_key not in self._links:
            raise ValueError(
                f"No direct link between {node_a_id!r} and {node_b_id!r}"
            )

        link = self._links[link_key]
        if n_bits is None:
            n_bits = self.n_bits_per_link

        seed = self._next_seed()
        protocol = BB84Protocol(seed=seed)
        return protocol.generate_key(n_bits, link.channel)

    def establish_key_via_relay(
        self,
        path: List[str],
        n_bits: Optional[int] = None,
    ) -> QKDResult:
        """Establish a key between endpoints via trusted relay nodes.

        Each adjacent pair in the path establishes a key via BB84.
        The relay nodes XOR-chain the segment keys to create an
        end-to-end key that only the endpoints know (assuming relay
        nodes are trusted and delete their segment keys).

        Parameters
        ----------
        path : list[str]
            Ordered list of node IDs from source to destination.
            Intermediate nodes are treated as trusted relays.
        n_bits : int, optional
            Qubits per link. Defaults to ``self.n_bits_per_link``.

        Returns
        -------
        QKDResult
            End-to-end result. The final_key is the XOR-chained key.

        Raises
        ------
        ValueError
            If the path has fewer than 2 nodes or any link is missing.
        """
        if len(path) < 2:
            raise ValueError("Path must have at least 2 nodes")

        # Establish keys on each link
        segment_results: List[QKDResult] = []
        for i in range(len(path) - 1):
            result = self.establish_key(path[i], path[i + 1], n_bits)
            segment_results.append(result)
            if not result.secure:
                # If any segment fails, the whole path fails
                return QKDResult(
                    protocol="BB84-relay",
                    n_bits_sent=sum(r.n_bits_sent for r in segment_results),
                    secure=False,
                    qber=result.qber,
                )

        # XOR-chain the segment keys
        # End-to-end key length is limited by the shortest segment key
        min_len = min(len(r.final_key) for r in segment_results)
        if min_len == 0:
            return QKDResult(
                protocol="BB84-relay",
                n_bits_sent=sum(r.n_bits_sent for r in segment_results),
                secure=True,
                qber=max(r.qber for r in segment_results),
            )

        # XOR all segment keys together
        end_to_end_key = list(segment_results[0].final_key[:min_len])
        for seg_result in segment_results[1:]:
            for i in range(min_len):
                end_to_end_key[i] ^= seg_result.final_key[i]

        total_sent = sum(r.n_bits_sent for r in segment_results)
        max_qber = max(r.qber for r in segment_results)

        return QKDResult(
            protocol="BB84-relay",
            n_bits_sent=total_sent,
            n_bits_received=sum(r.n_bits_received for r in segment_results),
            final_key=end_to_end_key,
            qber=max_qber,
            key_rate=len(end_to_end_key) / total_sent if total_sent > 0 else 0.0,
            secure=True,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_seed(self) -> int:
        """Generate a deterministic sub-seed for each protocol run."""
        if self.seed is None:
            return np.random.randint(0, 2**31)
        seed = (self.seed + self._seed_counter * 7919) % (2**31)
        self._seed_counter += 1
        return seed

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"QKDNetwork(nodes={len(self._nodes)}, "
            f"links={len(self._links)})"
        )
