#!/usr/bin/env python3
"""
NQPU Platform Dashboard - Real-time Visualization and Control

A comprehensive web dashboard for the NQPU quantum simulation platform featuring:
- 3D Neural Tissue Visualization with Three.js
- Digital Terrarium with ASCII art rendering
- Quantum State Visualization
- Real-time metrics and control panel

Run with: python3 nqpu_dashboard.py
Then open: http://localhost:8050
"""

from __future__ import annotations

import sys
import os
import json
import time
import asyncio
import random
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field, asdict

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Import NQPU modules
try:
    from organic_neural_network import (
        OrganicNeuralNetwork, OrganicNeuron, OrganicSynapse,
        NeuronState, EmergenceTracker
    )
    from quantum_terrarium import (
        QuantumTerrarium, DigitalOrganism, Genome,
        QuantumState, Bacteria, Algae, Predator
    )
    from quantum_organism import QuantumDNA, QuantumOrganism
    NQPU_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import NQPU modules: {e}")
    NQPU_AVAILABLE = False


# ============================================================================
# SIMULATION STATE MANAGER
# ============================================================================

class SimulationState:
    """Manages the state of all simulations with thread-safe access."""

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.tick = 0

        # Neural tissue
        self.neural_network: Optional[OrganicNeuralNetwork] = None
        self.neural_history: deque = deque(maxlen=100)

        # Quantum terrarium
        self.terrarium: Optional[QuantumTerrarium] = None
        self.terrarium_history: deque = deque(maxlen=100)

        # Metrics
        self.metrics = {
            'population': [],
            'avg_energy': [],
            'neurogenesis_events': [],
            'quantum_coherence': [],
            'emergence_events': [],
            'timestamps': []
        }

        # Parameters
        self.params = {
            'energy_supply': 2.0,
            'temperature': 300.0,
            'mutation_rate': 0.01,
            'simulation_speed': 50  # ms per tick
        }

        # Connected clients
        self.clients: List[WebSocket] = []

    def initialize_simulations(self):
        """Initialize or reset simulations."""
        with self.lock:
            # Create neural tissue
            if NQPU_AVAILABLE:
                self.neural_network = OrganicNeuralNetwork(
                    size=(10.0, 10.0, 5.0),
                    initial_neurons=30,
                    energy_supply=self.params['energy_supply']
                )
                self.terrarium = QuantumTerrarium(width=60, height=30)
                self.terrarium.populate(n_bacteria=15, n_algae=8, n_predators=3)
                self.terrarium.seed_food(30)
            else:
                self.neural_network = None
                self.terrarium = None

            self.tick = 0
            self.neural_history.clear()
            self.terrarium_history.clear()
            self.metrics = {k: [] for k in self.metrics}

    def step(self):
        """Advance simulation by one tick."""
        with self.lock:
            if not self.running:
                return

            self.tick += 1

            # Step neural network
            if self.neural_network:
                self.neural_network.step(dt=0.5)

                # Apply periodic stimulation
                if self.tick % 20 == 0:
                    self.neural_network.stimulate(
                        position=(5.0, 5.0, 2.5),
                        intensity=15.0,
                        radius=3.0
                    )

                # Record neural history
                alive = [n for n in self.neural_network.neurons.values() if n.alive]
                if alive:
                    self.neural_history.append({
                        'tick': self.tick,
                        'neuron_count': len(alive),
                        'synapse_count': len(self.neural_network.synapses),
                        'avg_energy': np.mean([n.energy for n in alive]),
                        'active_count': sum(1 for n in alive if n.state == NeuronState.ACTIVE),
                        'quantum_count': sum(1 for n in alive if n.state in (NeuronState.SUPERPOSITION, NeuronState.ENTANGLED))
                    })

            # Step terrarium
            if self.terrarium:
                self.terrarium.step()

                # Record terrarium history
                if self.terrarium.organisms:
                    self.terrarium_history.append({
                        'tick': self.tick,
                        'population': len(self.terrarium.organisms),
                        'avg_energy': np.mean([o.energy for o in self.terrarium.organisms]),
                        'food_count': len(self.terrarium.food),
                        'quantum_states': sum(1 for o in self.terrarium.organisms if o.quantum_state != QuantumState.GROUND)
                    })

            # Update metrics
            self._update_metrics()

    def _update_metrics(self):
        """Update aggregated metrics."""
        timestamp = datetime.utcnow().isoformat()

        # Neural metrics
        if self.neural_history:
            latest = self.neural_history[-1]
            self.metrics['neurogenesis_events'].append(
                self.neural_network.neurogenesis_events if self.neural_network else 0
            )
            self.metrics['quantum_coherence'].append(
                latest['quantum_count'] / max(1, latest['neuron_count'])
            )

        # Terrarium metrics
        if self.terrarium_history:
            latest = self.terrarium_history[-1]
            self.metrics['population'].append(latest['population'])
            self.metrics['avg_energy'].append(latest['avg_energy'])
            self.metrics['emergence_events'].append(
                self.terrarium.speciations if self.terrarium else 0
            )

        self.metrics['timestamps'].append(timestamp)

        # Keep metrics bounded
        max_points = 200
        for key in self.metrics:
            if len(self.metrics[key]) > max_points:
                self.metrics[key] = self.metrics[key][-max_points:]

    def get_neural_data(self) -> Dict:
        """Get current neural network state for visualization."""
        with self.lock:
            if not self.neural_network:
                return {'neurons': [], 'synapses': [], 'stats': {}}

            neurons = []
            for nid, n in self.neural_network.neurons.items():
                if n.alive:
                    state_map = {
                        NeuronState.RESTING: 'resting',
                        NeuronState.ACTIVE: 'active',
                        NeuronState.REFRACTORY: 'refractory',
                        NeuronState.SUPERPOSITION: 'superposition',
                        NeuronState.ENTANGLED: 'entangled'
                    }
                    neurons.append({
                        'id': nid,
                        'x': n.x,
                        'y': n.y,
                        'z': n.z,
                        'energy': n.energy,
                        'state': state_map.get(n.state, 'resting'),
                        'age': n.age,
                        'generation': n.generation,
                        'membrane_potential': n.membrane_potential
                    })

            synapses = []
            for (pre, post), s in list(self.neural_network.synapses.items())[:500]:
                synapses.append({
                    'pre': pre,
                    'post': post,
                    'weight': s.weight,
                    'strength': s.strength
                })

            stats = {
                'neuron_count': len(neurons),
                'synapse_count': len(self.neural_network.synapses),
                'neurogenesis_events': self.neural_network.neurogenesis_events,
                'pruning_events': self.neural_network.pruning_events,
                'entanglement_events': self.neural_network.entanglement_events,
                'tick': self.tick,
                'time': self.neural_network.time
            }

            return {'neurons': neurons, 'synapses': synapses, 'stats': stats}

    def get_terrarium_data(self) -> Dict:
        """Get current terrarium state for visualization."""
        with self.lock:
            if not self.terrarium:
                return {'organisms': [], 'food': [], 'grid': [], 'stats': {}}

            organisms = []
            for org in self.terrarium.organisms:
                state_map = {
                    QuantumState.GROUND: 'ground',
                    QuantumState.SUPERPOSITION: 'superposition',
                    QuantumState.ENTANGLED: 'entangled'
                }
                org_type = 'bacteria'
                if isinstance(org, Algae):
                    org_type = 'algae'
                elif isinstance(org, Predator):
                    org_type = 'predator'

                organisms.append({
                    'id': id(org),
                    'x': org.x,
                    'y': org.y,
                    'energy': org.energy,
                    'age': org.age,
                    'type': org_type,
                    'state': state_map.get(org.quantum_state, 'ground'),
                    'size': org.size,
                    'generation': org.generation
                })

            food = [{'x': fx, 'y': fy, 'energy': e} for fx, fy, e in self.terrarium.food]

            # Generate ASCII grid
            grid = self._generate_ascii_grid()

            stats = {
                'population': len(organisms),
                'food_count': len(food),
                'tick': self.terrarium.tick,
                'extinctions': self.terrarium.extinctions,
                'speciations': self.terrarium.speciations
            }

            return {'organisms': organisms, 'food': food, 'grid': grid, 'stats': stats}

    def _generate_ascii_grid(self) -> List[List[str]]:
        """Generate ASCII representation of terrarium."""
        if not self.terrarium:
            return []

        width, height = 60, 25
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Add food
        for fx, fy, energy in self.terrarium.food:
            x, y = int(fx) % width, int(fy) % height
            grid[y][x] = '.' if energy < 20 else '*'

        # Add organisms
        for org in self.terrarium.organisms:
            x, y = int(org.x) % width, int(org.y) % height

            if isinstance(org, Algae):
                base = '~'
            elif isinstance(org, Predator):
                base = '>'
            elif org.size < 0.3:
                base = 'o'
            elif org.size < 0.6:
                base = 'O'
            else:
                base = '0'

            # State modifier
            if org.quantum_state == QuantumState.SUPERPOSITION:
                base = '?'
            elif org.quantum_state == QuantumState.ENTANGLED:
                base = '&'
            elif org.energy < 30:
                base = '.'

            grid[y][x] = base

        return grid

    def get_quantum_data(self) -> Dict:
        """Get quantum state visualization data."""
        with self.lock:
            quantum_states = []

            # From neural network
            if self.neural_network:
                for nid, n in self.neural_network.neurons.items():
                    if n.alive and n.state in (NeuronState.SUPERPOSITION, NeuronState.ENTANGLED):
                        quantum_states.append({
                            'source': 'neural',
                            'id': nid,
                            'x': n.x,
                            'y': n.y,
                            'z': n.z,
                            'state': 'superposition' if n.state == NeuronState.SUPERPOSITION else 'entangled',
                            'entangled_with': n.entangled_with
                        })

            # From terrarium
            if self.terrarium:
                for org in self.terrarium.organisms:
                    if org.quantum_state != QuantumState.GROUND:
                        quantum_states.append({
                            'source': 'terrarium',
                            'id': id(org),
                            'x': org.x,
                            'y': org.y,
                            'z': 0,
                            'state': 'superposition' if org.quantum_state == QuantumState.SUPERPOSITION else 'entangled',
                            'entangled_with': id(org.entangled_with) if org.entangled_with else None
                        })

            # Calculate coherence metrics
            total_entities = 0
            quantum_count = 0
            entangled_pairs = 0

            if self.neural_network:
                alive = [n for n in self.neural_network.neurons.values() if n.alive]
                total_entities += len(alive)
                quantum_count += sum(1 for n in alive if n.state in (NeuronState.SUPERPOSITION, NeuronState.ENTANGLED))
                entangled_pairs += sum(1 for n in alive if n.state == NeuronState.ENTANGLED) // 2

            if self.terrarium:
                total_entities += len(self.terrarium.organisms)
                quantum_count += sum(1 for o in self.terrarium.organisms if o.quantum_state != QuantumState.GROUND)

            coherence = quantum_count / max(1, total_entities)

            return {
                'quantum_states': quantum_states,
                'coherence': coherence,
                'entangled_pairs': entangled_pairs,
                'total_quantum': quantum_count,
                'total_entities': total_entities
            }

    def get_metrics_data(self) -> Dict:
        """Get time-series metrics for charts."""
        with self.lock:
            return {
                'population': list(self.metrics['population'])[-100:],
                'avg_energy': list(self.metrics['avg_energy'])[-100:],
                'neurogenesis': list(self.metrics['neurogenesis_events'])[-100:],
                'coherence': list(self.metrics['quantum_coherence'])[-100:],
                'emergence': list(self.metrics['emergence_events'])[-100:],
                'timestamps': list(self.metrics['timestamps'])[-100:]
            }

    def apply_stimulation(self, x: float, y: float, z: float, intensity: float, radius: float):
        """Apply stimulation to neural tissue."""
        with self.lock:
            if self.neural_network:
                self.neural_network.stimulate((x, y, z), intensity, radius)

    def update_params(self, params: Dict):
        """Update simulation parameters."""
        with self.lock:
            self.params.update(params)
            if self.neural_network:
                self.neural_network.energy_supply = self.params.get('energy_supply', 2.0)


# Global state
state = SimulationState()


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="NQPU Dashboard",
    description="Real-time visualization dashboard for NQPU quantum simulations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HTML TEMPLATE (Embedded)
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NQPU Dashboard - Quantum Simulation Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #131a2a;
            --bg-card: #1a2235;
            --text-primary: #e4e8f1;
            --text-secondary: #8b92a5;
            --accent-blue: #4a9eff;
            --accent-cyan: #00d4ff;
            --accent-purple: #a855f7;
            --accent-green: #22c55e;
            --accent-yellow: #eab308;
            --accent-red: #ef4444;
            --border-color: #2a3548;
        }

        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Segoe UI', system-ui, sans-serif;
            min-height: 100vh;
        }

        .navbar {
            background: var(--bg-secondary) !important;
            border-bottom: 1px solid var(--border-color);
            padding: 0.75rem 1.5rem;
        }

        .navbar-brand {
            font-weight: 700;
            font-size: 1.4rem;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .status-running {
            background: rgba(34, 197, 94, 0.2);
            color: var(--accent-green);
        }

        .status-stopped {
            background: rgba(239, 68, 68, 0.2);
            color: var(--accent-red);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        .status-running .status-dot {
            background: var(--accent-green);
        }

        .status-stopped .status-dot {
            background: var(--accent-red);
            animation: none;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .main-container {
            padding: 1.5rem;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }

        .card-header {
            background: transparent;
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-title {
            font-size: 1rem;
            font-weight: 600;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card-body {
            padding: 1.25rem;
        }

        .viz-container {
            width: 100%;
            height: 400px;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
            position: relative;
        }

        #neural-3d-container, #quantum-viz-container {
            width: 100%;
            height: 100%;
        }

        .terrarium-display {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 10px;
            line-height: 1.2;
            color: var(--text-primary);
            background: var(--bg-primary);
            padding: 10px;
            border-radius: 8px;
            overflow: auto;
            white-space: pre;
        }

        .metric-card {
            background: linear-gradient(135deg, var(--bg-card), var(--bg-secondary));
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
            border: 1px solid var(--border-color);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .metric-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        .metric-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
            opacity: 0.7;
        }

        .btn-nqpu {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border: none;
            color: white;
            padding: 0.5rem 1.25rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .btn-nqpu:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(74, 158, 255, 0.3);
            color: white;
        }

        .btn-nqpu-outline {
            background: transparent;
            border: 1px solid var(--accent-blue);
            color: var(--accent-blue);
            padding: 0.5rem 1.25rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .btn-nqpu-outline:hover {
            background: rgba(74, 158, 255, 0.1);
            color: var(--accent-blue);
        }

        .control-group {
            margin-bottom: 1rem;
        }

        .control-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            display: block;
        }

        .form-range {
            width: 100%;
        }

        .form-range::-webkit-slider-thumb {
            background: var(--accent-blue);
        }

        .form-range::-webkit-slider-runnable-track {
            background: var(--border-color);
        }

        .param-value {
            font-size: 0.85rem;
            color: var(--accent-cyan);
            float: right;
        }

        .chart-container {
            width: 100%;
            height: 250px;
        }

        .legend-item {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-right: 1rem;
            font-size: 0.85rem;
        }

        .legend-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }

        .log-container {
            height: 200px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            background: var(--bg-primary);
            border-radius: 8px;
            padding: 10px;
        }

        .log-entry {
            padding: 2px 0;
            border-bottom: 1px solid var(--border-color);
        }

        .log-time {
            color: var(--text-secondary);
            margin-right: 0.5rem;
        }

        .log-event {
            color: var(--accent-cyan);
        }

        .log-emergence {
            color: var(--accent-yellow);
        }

        .log-quantum {
            color: var(--accent-purple);
        }

        .tab-content {
            padding-top: 1rem;
        }

        .nav-tabs {
            border-bottom: 1px solid var(--border-color);
        }

        .nav-tabs .nav-link {
            color: var(--text-secondary);
            border: none;
            padding: 0.75rem 1.25rem;
            font-weight: 500;
        }

        .nav-tabs .nav-link.active {
            color: var(--accent-cyan);
            background: transparent;
            border-bottom: 2px solid var(--accent-cyan);
        }

        .nav-tabs .nav-link:hover {
            color: var(--text-primary);
            border: none;
        }

        .tooltip-nqpu {
            position: absolute;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            display: none;
        }

        .organism-hover {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid var(--accent-cyan);
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
            pointer-events: none;
            z-index: 100;
        }

        @media (max-width: 1200px) {
            .viz-container {
                height: 300px;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg">
        <div class="container-fluid">
            <span class="navbar-brand">
                <i class="bi bi-cpu-fill"></i> NQPU Dashboard
            </span>
            <div class="d-flex align-items-center gap-3">
                <div id="status-indicator" class="status-indicator status-stopped">
                    <span class="status-dot"></span>
                    <span id="status-text">Stopped</span>
                </div>
                <span class="text-secondary" style="font-size: 0.9rem;">
                    Tick: <span id="tick-counter">0</span>
                </span>
            </div>
        </div>
    </nav>

    <div class="container-fluid main-container">
        <!-- Metrics Row -->
        <div class="row mb-4">
            <div class="col-md-2 col-sm-4 mb-3">
                <div class="metric-card">
                    <div class="metric-icon"><i class="bi bi-bug-fill"></i></div>
                    <div class="metric-value" id="metric-population">0</div>
                    <div class="metric-label">Population</div>
                </div>
            </div>
            <div class="col-md-2 col-sm-4 mb-3">
                <div class="metric-card">
                    <div class="metric-icon"><i class="bi bi-lightning-fill"></i></div>
                    <div class="metric-value" id="metric-energy">0</div>
                    <div class="metric-label">Avg Energy</div>
                </div>
            </div>
            <div class="col-md-2 col-sm-4 mb-3">
                <div class="metric-card">
                    <div class="metric-icon"><i class="bi bi-plus-circle-fill"></i></div>
                    <div class="metric-value" id="metric-neurogenesis">0</div>
                    <div class="metric-label">Neurogenesis</div>
                </div>
            </div>
            <div class="col-md-2 col-sm-4 mb-3">
                <div class="metric-card">
                    <div class="metric-icon"><i class="bi bi-diamond-half"></i></div>
                    <div class="metric-value" id="metric-coherence">0%</div>
                    <div class="metric-label">Quantum Coherence</div>
                </div>
            </div>
            <div class="col-md-2 col-sm-4 mb-3">
                <div class="metric-card">
                    <div class="metric-icon"><i class="bi bi-stars"></i></div>
                    <div class="metric-value" id="metric-emergence">0</div>
                    <div class="metric-label">Emergence Events</div>
                </div>
            </div>
            <div class="col-md-2 col-sm-4 mb-3">
                <div class="metric-card">
                    <div class="metric-icon"><i class="bi bi-diagram-3-fill"></i></div>
                    <div class="metric-value" id="metric-synapses">0</div>
                    <div class="metric-label">Synapses</div>
                </div>
            </div>
        </div>

        <div class="row">
            <!-- Main Visualizations -->
            <div class="col-lg-8">
                <!-- 3D Neural Tissue -->
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title">
                            <i class="bi bi-box" style="color: var(--accent-cyan);"></i>
                            3D Neural Tissue
                        </h5>
                        <div>
                            <span class="legend-item">
                                <span class="legend-dot" style="background: #22c55e;"></span> Resting
                            </span>
                            <span class="legend-item">
                                <span class="legend-dot" style="background: #ef4444;"></span> Active
                            </span>
                            <span class="legend-item">
                                <span class="legend-dot" style="background: #a855f7;"></span> Quantum
                            </span>
                        </div>
                    </div>
                    <div class="card-body p-0">
                        <div class="viz-container">
                            <div id="neural-3d-container"></div>
                        </div>
                    </div>
                </div>

                <!-- Digital Terrarium -->
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title">
                            <i class="bi bi-globe" style="color: var(--accent-green);"></i>
                            Digital Terrarium
                        </h5>
                        <div>
                            <span class="legend-item">
                                <span class="legend-dot" style="background: #22c55e;"></span> Bacteria
                            </span>
                            <span class="legend-item">
                                <span class="legend-dot" style="background: #22c55e;"></span> Algae ~
                            </span>
                            <span class="legend-item">
                                <span class="legend-dot" style="background: #ef4444;"></span> Predator >
                            </span>
                            <span class="legend-item">
                                <span class="legend-dot" style="background: #a855f7;"></span> Quantum ?
                            </span>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="terrarium-display" id="terrarium-display">
                            Loading terrarium...
                        </div>
                    </div>
                </div>

                <!-- Charts -->
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title">
                            <i class="bi bi-graph-up" style="color: var(--accent-yellow);"></i>
                            Real-time Metrics
                        </h5>
                    </div>
                    <div class="card-body">
                        <ul class="nav nav-tabs" role="tablist">
                            <li class="nav-item">
                                <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#chart-population">Population</button>
                            </li>
                            <li class="nav-item">
                                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#chart-energy">Energy</button>
                            </li>
                            <li class="nav-item">
                                <button class="nav-link" data-bs-toggle="tab" data-bs-target="#chart-quantum">Quantum</button>
                            </li>
                        </ul>
                        <div class="tab-content">
                            <div class="tab-pane fade show active" id="chart-population">
                                <div class="chart-container" id="population-chart"></div>
                            </div>
                            <div class="tab-pane fade" id="chart-energy">
                                <div class="chart-container" id="energy-chart"></div>
                            </div>
                            <div class="tab-pane fade" id="chart-quantum">
                                <div class="chart-container" id="quantum-chart"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Sidebar -->
            <div class="col-lg-4">
                <!-- Control Panel -->
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title">
                            <i class="bi bi-sliders" style="color: var(--accent-blue);"></i>
                            Control Panel
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="d-grid gap-2 mb-3">
                            <button id="btn-start" class="btn btn-nqpu" onclick="toggleSimulation()">
                                <i class="bi bi-play-fill"></i> Start Simulation
                            </button>
                        </div>
                        <div class="d-grid gap-2 mb-3">
                            <button class="btn btn-nqpu-outline" onclick="resetSimulation()">
                                <i class="bi bi-arrow-clockwise"></i> Reset
                            </button>
                        </div>

                        <hr style="border-color: var(--border-color);">

                        <div class="control-group">
                            <label class="control-label">
                                Energy Supply <span class="param-value" id="val-energy">2.0</span>
                            </label>
                            <input type="range" class="form-range" id="param-energy" min="0.5" max="5" step="0.1" value="2.0" onchange="updateParam('energy_supply', this.value)">
                        </div>

                        <div class="control-group">
                            <label class="control-label">
                                Temperature <span class="param-value" id="val-temp">300</span>
                            </label>
                            <input type="range" class="form-range" id="param-temp" min="250" max="400" step="10" value="300" onchange="updateParam('temperature', this.value)">
                        </div>

                        <div class="control-group">
                            <label class="control-label">
                                Mutation Rate <span class="param-value" id="val-mutation">0.01</span>
                            </label>
                            <input type="range" class="form-range" id="param-mutation" min="0" max="0.1" step="0.005" value="0.01" onchange="updateParam('mutation_rate', this.value)">
                        </div>

                        <div class="control-group">
                            <label class="control-label">
                                Simulation Speed <span class="param-value" id="val-speed">50ms</span>
                            </label>
                            <input type="range" class="form-range" id="param-speed" min="10" max="200" step="10" value="50" onchange="updateParam('simulation_speed', this.value)">
                        </div>

                        <hr style="border-color: var(--border-color);">

                        <h6 style="color: var(--text-secondary); font-size: 0.85rem; margin-bottom: 0.75rem;">Stimulation</h6>
                        <div class="row g-2">
                            <div class="col-4">
                                <input type="number" class="form-control form-control-sm" id="stim-x" placeholder="X" value="5.0">
                            </div>
                            <div class="col-4">
                                <input type="number" class="form-control form-control-sm" id="stim-y" placeholder="Y" value="5.0">
                            </div>
                            <div class="col-4">
                                <input type="number" class="form-control form-control-sm" id="stim-z" placeholder="Z" value="2.5">
                            </div>
                        </div>
                        <div class="row g-2 mt-2">
                            <div class="col-6">
                                <input type="number" class="form-control form-control-sm" id="stim-intensity" placeholder="Intensity" value="15.0">
                            </div>
                            <div class="col-6">
                                <button class="btn btn-nqpu-outline btn-sm w-100" onclick="applyStimulation()">
                                    <i class="bi bi-lightning"></i> Stimulate
                                </button>
                            </div>
                        </div>

                        <hr style="border-color: var(--border-color);">

                        <div class="d-grid gap-2">
                            <button class="btn btn-nqpu-outline btn-sm" onclick="exportData('json')">
                                <i class="bi bi-download"></i> Export JSON
                            </button>
                            <button class="btn btn-nqpu-outline btn-sm" onclick="exportData('csv')">
                                <i class="bi bi-filetype-csv"></i> Export CSV
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Quantum State -->
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title">
                            <i class="bi bi-diamond" style="color: var(--accent-purple);"></i>
                            Quantum State
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="viz-container" style="height: 200px;">
                            <canvas id="quantum-canvas"></canvas>
                        </div>
                        <div class="mt-3">
                            <div class="row text-center">
                                <div class="col-6">
                                    <div class="metric-value" style="font-size: 1.5rem;" id="quantum-coherence">0%</div>
                                    <div class="metric-label">Coherence</div>
                                </div>
                                <div class="col-6">
                                    <div class="metric-value" style="font-size: 1.5rem;" id="entangled-pairs">0</div>
                                    <div class="metric-label">Entangled Pairs</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Event Log -->
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title">
                            <i class="bi bi-terminal" style="color: var(--accent-cyan);"></i>
                            Event Log
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="log-container" id="event-log">
                            <div class="log-entry">
                                <span class="log-time">[00:00:00]</span>
                                <span class="log-event">System initialized</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // WebSocket connection
        let ws = null;
        let simulationRunning = false;
        let simulationSpeed = 50;

        // Three.js scene
        let scene, camera, renderer, neurons = {}, synapses = [];
        let quantumCtx, quantumCanvas;

        // Initialize Three.js scene
        function initThreeJS() {
            const container = document.getElementById('neural-3d-container');
            const width = container.clientWidth;
            const height = container.clientHeight;

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0e17);

            camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
            camera.position.set(15, 15, 15);
            camera.lookAt(5, 5, 2.5);

            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height);
            container.appendChild(renderer.domElement);

            // Add lights
            const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(10, 20, 10);
            scene.add(directionalLight);

            // Add grid helper
            const gridHelper = new THREE.GridHelper(10, 10, 0x2a3548, 0x1a2235);
            gridHelper.position.set(5, 0, 2.5);
            scene.add(gridHelper);

            // Animation loop
            function animate() {
                requestAnimationFrame(animate);
                renderer.render(scene, camera);
            }
            animate();

            // Handle resize
            window.addEventListener('resize', () => {
                const width = container.clientWidth;
                const height = container.clientHeight;
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height);
            });

            // Mouse rotation
            let isDragging = false;
            let previousMousePosition = { x: 0, y: 0 };

            container.addEventListener('mousedown', (e) => { isDragging = true; });
            container.addEventListener('mouseup', () => { isDragging = false; });
            container.addEventListener('mouseleave', () => { isDragging = false; });
            container.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    const deltaX = e.clientX - previousMousePosition.x;
                    const deltaY = e.clientY - previousMousePosition.y;

                    const spherical = new THREE.Spherical();
                    spherical.setFromVector3(camera.position);
                    spherical.theta -= deltaX * 0.01;
                    spherical.phi -= deltaY * 0.01;
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

                    camera.position.setFromSpherical(spherical);
                    camera.lookAt(5, 5, 2.5);
                }
                previousMousePosition = { x: e.clientX, y: e.clientY };
            });
        }

        // Update 3D visualization
        function updateNeuralViz(data) {
            if (!scene) return;

            // Update neurons
            const currentIds = new Set(data.neurons.map(n => n.id));

            // Remove dead neurons
            Object.keys(neurons).forEach(id => {
                if (!currentIds.has(parseInt(id))) {
                    scene.remove(neurons[id]);
                    delete neurons[id];
                }
            });

            // Add or update neurons
            data.neurons.forEach(n => {
                let mesh = neurons[n.id];
                const color = getNeuronColor(n.state);
                const size = 0.1 + (n.energy / 200) * 0.3;

                if (!mesh) {
                    const geometry = new THREE.SphereGeometry(size, 16, 16);
                    const material = new THREE.MeshPhongMaterial({
                        color: color,
                        emissive: color,
                        emissiveIntensity: n.state === 'active' ? 0.5 : 0.1
                    });
                    mesh = new THREE.Mesh(geometry, material);
                    scene.add(mesh);
                    neurons[n.id] = mesh;
                }

                mesh.position.set(n.x, n.z, n.y);
                mesh.material.color.setHex(color);
                mesh.material.emissive.setHex(color);
                mesh.material.emissiveIntensity = n.state === 'active' ? 0.5 : 0.1;
            });

            // Clear old synapses and add new ones (limit for performance)
            synapses.forEach(s => scene.remove(s));
            synapses = [];

            const synapseSample = data.synapses.slice(0, 200);
            synapseSample.forEach(s => {
                const preMesh = neurons[s.pre];
                const postMesh = neurons[s.post];
                if (preMesh && postMesh) {
                    const points = [preMesh.position, postMesh.position];
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const material = new THREE.LineBasicMaterial({
                        color: 0x4a9eff,
                        transparent: true,
                        opacity: Math.min(1, s.weight * 0.5)
                    });
                    const line = new THREE.Line(geometry, material);
                    scene.add(line);
                    synapses.push(line);
                }
            });
        }

        function getNeuronColor(state) {
            switch (state) {
                case 'active': return 0xef4444;
                case 'superposition': return 0xa855f7;
                case 'entangled': return 0x00d4ff;
                case 'refractory': return 0xeab308;
                default: return 0x22c55e;
            }
        }

        // Update terrarium display
        function updateTerrarium(data) {
            const display = document.getElementById('terrarium-display');
            const border = '+' + '-'.repeat(60) + '+';
            let html = border + '\\n';

            data.grid.forEach(row => {
                html += '|' + row.join('') + '|\\n';
            });

            html += border;
            display.textContent = html;
        }

        // Initialize charts
        let populationChart, energyChart, quantumChart;

        function initCharts() {
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { t: 20, r: 20, b: 40, l: 50 },
                font: { color: '#8b92a5' },
                xaxis: { gridcolor: '#2a3548' },
                yaxis: { gridcolor: '#2a3548' }
            };

            const config = { responsive: true, displayModeBar: false };

            Plotly.newPlot('population-chart', [{
                y: [],
                name: 'Neural Neurons',
                line: { color: '#4a9eff' }
            }, {
                y: [],
                name: 'Terrarium Pop',
                line: { color: '#22c55e' }
            }], layout, config);

            Plotly.newPlot('energy-chart', [{
                y: [],
                name: 'Avg Energy',
                line: { color: '#eab308' }
            }], layout, config);

            Plotly.newPlot('quantum-chart', [{
                y: [],
                name: 'Coherence %',
                line: { color: '#a855f7' }
            }], layout, config);
        }

        function updateCharts(data) {
            Plotly.extendTraces('population-chart', {
                y: [[data.neural_count || 0], [data.terrarium_pop || 0]]
            }, [0, 1], 100);

            Plotly.extendTraces('energy-chart', {
                y: [[data.avg_energy || 0]]
            }, [0], 100);

            Plotly.extendTraces('quantum-chart', {
                y: [[(data.coherence || 0) * 100]]
            }, [0], 100);
        }

        // Initialize quantum canvas
        function initQuantumCanvas() {
            quantumCanvas = document.getElementById('quantum-canvas');
            quantumCtx = quantumCanvas.getContext('2d');
            quantumCanvas.width = quantumCanvas.parentElement.clientWidth;
            quantumCanvas.height = quantumCanvas.parentElement.clientHeight;
        }

        function updateQuantumViz(data) {
            if (!quantumCtx) return;

            const width = quantumCanvas.width;
            const height = quantumCanvas.height;

            // Clear
            quantumCtx.fillStyle = '#0a0e17';
            quantumCtx.fillRect(0, 0, width, height);

            // Draw quantum states
            data.quantum_states.forEach((q, i) => {
                const x = (q.x / 10) * width;
                const y = (q.y / 30) * height;

                // Pulsing effect
                const pulse = Math.sin(Date.now() / 200 + i) * 0.3 + 0.7;
                const radius = 8 * pulse;

                if (q.state === 'superposition') {
                    // Draw superposition as layered circles
                    for (let j = 3; j >= 0; j--) {
                        quantumCtx.beginPath();
                        quantumCtx.arc(x, y, radius + j * 4, 0, Math.PI * 2);
                        quantumCtx.fillStyle = `rgba(168, 85, 247, ${0.1 + j * 0.1})`;
                        quantumCtx.fill();
                    }
                } else if (q.state === 'entangled' && q.entangled_with) {
                    // Draw entanglement beam
                    const partner = data.quantum_states.find(p => p.id === q.entangled_with);
                    if (partner) {
                        const px = (partner.x / 10) * width;
                        const py = (partner.y / 30) * height;

                        quantumCtx.beginPath();
                        quantumCtx.moveTo(x, y);
                        quantumCtx.lineTo(px, py);
                        quantumCtx.strokeStyle = `rgba(0, 212, 255, ${0.3 * pulse})`;
                        quantumCtx.lineWidth = 2;
                        quantumCtx.stroke();
                    }
                }

                // Draw core
                quantumCtx.beginPath();
                quantumCtx.arc(x, y, radius * 0.5, 0, Math.PI * 2);
                quantumCtx.fillStyle = q.state === 'superposition' ? '#a855f7' : '#00d4ff';
                quantumCtx.fill();
            });

            // Update metrics
            document.getElementById('quantum-coherence').textContent =
                (data.coherence * 100).toFixed(1) + '%';
            document.getElementById('entangled-pairs').textContent = data.entangled_pairs;
        }

        // Update metrics display
        function updateMetrics(data) {
            document.getElementById('metric-population').textContent = data.neural_stats?.neuron_count || 0;
            document.getElementById('metric-energy').textContent =
                (data.neural_history?.slice(-1)[0]?.avg_energy || 0).toFixed(1);
            document.getElementById('metric-neurogenesis').textContent = data.neural_stats?.neurogenesis_events || 0;
            document.getElementById('metric-coherence').textContent =
                ((data.quantum?.coherence || 0) * 100).toFixed(0) + '%';
            document.getElementById('metric-emergence').textContent = data.terrarium_stats?.speciations || 0;
            document.getElementById('metric-synapses').textContent = data.neural_stats?.synapse_count || 0;
            document.getElementById('tick-counter').textContent = data.tick || 0;
        }

        // Add log entry
        function addLog(message, type = 'event') {
            const log = document.getElementById('event-log');
            const time = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-${type}">${message}</span>`;
            log.insertBefore(entry, log.firstChild);

            // Keep only last 50 entries
            while (log.children.length > 50) {
                log.removeChild(log.lastChild);
            }
        }

        // WebSocket handlers
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

            ws.onopen = () => {
                addLog('Connected to simulation server', 'event');
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'state') {
                    updateNeuralViz(data.neural);
                    updateTerrarium(data.terrarium);
                    updateQuantumViz(data.quantum);
                    updateMetrics(data);

                    // Update charts periodically
                    if (data.tick % 5 === 0) {
                        updateCharts({
                            neural_count: data.neural_stats?.neuron_count,
                            terrarium_pop: data.terrarium_stats?.population,
                            avg_energy: data.neural_history?.slice(-1)[0]?.avg_energy,
                            coherence: data.quantum?.coherence
                        });
                    }
                } else if (data.type === 'event') {
                    addLog(data.message, data.event_type);
                }
            };

            ws.onclose = () => {
                addLog('Disconnected from server', 'event');
                setTimeout(connectWebSocket, 3000);
            };

            ws.onerror = (error) => {
                addLog('WebSocket error', 'event');
            };
        }

        // Control functions
        function toggleSimulation() {
            simulationRunning = !simulationRunning;
            const btn = document.getElementById('btn-start');
            const indicator = document.getElementById('status-indicator');
            const statusText = document.getElementById('status-text');

            if (simulationRunning) {
                btn.innerHTML = '<i class="bi bi-pause-fill"></i> Pause Simulation';
                indicator.className = 'status-indicator status-running';
                statusText.textContent = 'Running';
                ws.send(JSON.stringify({ action: 'start' }));
                addLog('Simulation started', 'event');
            } else {
                btn.innerHTML = '<i class="bi bi-play-fill"></i> Start Simulation';
                indicator.className = 'status-indicator status-stopped';
                statusText.textContent = 'Paused';
                ws.send(JSON.stringify({ action: 'stop' }));
                addLog('Simulation paused', 'event');
            }
        }

        function resetSimulation() {
            ws.send(JSON.stringify({ action: 'reset' }));
            addLog('Simulation reset', 'event');
        }

        function updateParam(param, value) {
            const displayIds = {
                'energy_supply': 'val-energy',
                'temperature': 'val-temp',
                'mutation_rate': 'val-mutation',
                'simulation_speed': 'val-speed'
            };

            const displayValue = param === 'simulation_speed' ? value + 'ms' : value;
            document.getElementById(displayIds[param]).textContent = displayValue;

            ws.send(JSON.stringify({
                action: 'update_param',
                param: param,
                value: parseFloat(value)
            }));
        }

        function applyStimulation() {
            const x = parseFloat(document.getElementById('stim-x').value);
            const y = parseFloat(document.getElementById('stim-y').value);
            const z = parseFloat(document.getElementById('stim-z').value);
            const intensity = parseFloat(document.getElementById('stim-intensity').value);

            ws.send(JSON.stringify({
                action: 'stimulate',
                x: x,
                y: y,
                z: z,
                intensity: intensity,
                radius: 3.0
            }));

            addLog(`Stimulation applied at (${x}, ${y}, ${z})`, 'event');
        }

        function exportData(format) {
            ws.send(JSON.stringify({
                action: 'export',
                format: format
            }));
            addLog(`Exporting data as ${format.toUpperCase()}`, 'event');
        }

        // Animation loop for quantum visualization
        function animateQuantum() {
            if (simulationRunning) {
                // Trigger re-render of quantum canvas
            }
            requestAnimationFrame(animateQuantum);
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', () => {
            initThreeJS();
            initCharts();
            initQuantumCanvas();
            connectWebSocket();
            animateQuantum();
        });
    </script>
</body>
</html>
"""


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard page."""
    return HTML_TEMPLATE


@app.get("/api/state")
async def get_state():
    """Get current simulation state."""
    return JSONResponse({
        'tick': state.tick,
        'running': state.running,
        'neural': state.get_neural_data(),
        'terrarium': state.get_terrarium_data(),
        'quantum': state.get_quantum_data(),
        'metrics': state.get_metrics_data(),
        'params': state.params
    })


@app.post("/api/start")
async def start_simulation():
    """Start the simulation."""
    state.running = True
    return {'status': 'started'}


@app.post("/api/stop")
async def stop_simulation():
    """Stop the simulation."""
    state.running = False
    return {'status': 'stopped'}


@app.post("/api/reset")
async def reset_simulation():
    """Reset the simulation."""
    state.running = False
    state.initialize_simulations()
    return {'status': 'reset'}


@app.post("/api/stimulate")
async def stimulate(request: Request):
    """Apply stimulation to neural tissue."""
    data = await request.json()
    state.apply_stimulation(
        x=data.get('x', 5.0),
        y=data.get('y', 5.0),
        z=data.get('z', 2.5),
        intensity=data.get('intensity', 15.0),
        radius=data.get('radius', 3.0)
    )
    return {'status': 'stimulated'}


@app.post("/api/params")
async def update_params(request: Request):
    """Update simulation parameters."""
    data = await request.json()
    state.update_params(data)
    return {'status': 'updated', 'params': state.params}


@app.get("/api/export/{format}")
async def export_data(format: str):
    """Export simulation data."""
    if format == 'json':
        data = {
            'neural': state.get_neural_data(),
            'terrarium': state.get_terrarium_data(),
            'metrics': state.get_metrics_data()
        }
        return JSONResponse(data)
    elif format == 'csv':
        # Generate CSV from metrics
        metrics = state.get_metrics_data()
        lines = ['timestamp,population,avg_energy,coherence']
        for i in range(len(metrics['timestamps'])):
            lines.append(','.join([
                metrics['timestamps'][i] if i < len(metrics['timestamps']) else '',
                str(metrics['population'][i]) if i < len(metrics['population']) else '0',
                str(metrics['avg_energy'][i]) if i < len(metrics['avg_energy']) else '0',
                str(metrics['coherence'][i]) if i < len(metrics['coherence']) else '0'
            ]))
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse('\\n'.join(lines), media_type='text/csv')
    else:
        return {'error': 'Unknown format'}


# ============================================================================
# WEBSOCKET HANDLER
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        state.clients.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if websocket in state.clients:
            state.clients.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    state.initialize_simulations()

    try:
        while True:
            # Receive message
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
            except asyncio.TimeoutError:
                data = None

            if data:
                action = data.get('action')

                if action == 'start':
                    state.running = True
                    await manager.broadcast({'type': 'event', 'message': 'Simulation started', 'event_type': 'event'})
                elif action == 'stop':
                    state.running = False
                    await manager.broadcast({'type': 'event', 'message': 'Simulation paused', 'event_type': 'event'})
                elif action == 'reset':
                    state.running = False
                    state.initialize_simulations()
                    await manager.broadcast({'type': 'event', 'message': 'Simulation reset', 'event_type': 'event'})
                elif action == 'update_param':
                    state.update_params({data['param']: data['value']})
                elif action == 'stimulate':
                    state.apply_stimulation(
                        data['x'], data['y'], data['z'],
                        data['intensity'], data.get('radius', 3.0)
                    )
                elif action == 'export':
                    # Handle export request
                    pass

            # Step simulation
            if state.running:
                state.step()

                # Check for events
                if state.neural_network and state.neural_network.neurogenesis_events > 0:
                    if random.random() < 0.1:  # Don't spam
                        await manager.broadcast({
                            'type': 'event',
                            'message': 'Neurogenesis event detected!',
                            'event_type': 'emergence'
                        })

                if state.terrarium and len(state.terrarium.organisms) > 0:
                    quantum_count = sum(1 for o in state.terrarium.organisms
                                       if o.quantum_state != QuantumState.GROUND)
                    if quantum_count > 0 and random.random() < 0.05:
                        await manager.broadcast({
                            'type': 'event',
                            'message': f'Quantum state detected in {quantum_count} organisms',
                            'event_type': 'quantum'
                        })

            # Send state update
            neural_data = state.get_neural_data()
            terrarium_data = state.get_terrarium_data()
            quantum_data = state.get_quantum_data()
            metrics_data = state.get_metrics_data()

            await websocket.send_json({
                'type': 'state',
                'tick': state.tick,
                'neural': neural_data,
                'terrarium': terrarium_data,
                'quantum': quantum_data,
                'metrics': metrics_data,
                'neural_stats': neural_data.get('stats', {}),
                'terrarium_stats': terrarium_data.get('stats', {}),
                'neural_history': list(state.neural_history)[-10:]
            })

            # Control update rate
            await asyncio.sleep(state.params['simulation_speed'] / 1000)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================================================
# BACKGROUND SIMULATION TASK
# ============================================================================

async def simulation_loop():
    """Background task for running simulation."""
    while True:
        if state.running:
            state.step()

            # Broadcast to all clients
            neural_data = state.get_neural_data()
            terrarium_data = state.get_terrarium_data()
            quantum_data = state.get_quantum_data()

            await manager.broadcast({
                'type': 'state',
                'tick': state.tick,
                'neural': neural_data,
                'terrarium': terrarium_data,
                'quantum': quantum_data,
                'neural_stats': neural_data.get('stats', {}),
                'terrarium_stats': terrarium_data.get('stats', {})
            })

        await asyncio.sleep(state.params['simulation_speed'] / 1000)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NQPU Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind to")
    args = parser.parse_args()

    print(f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║               NQPU Dashboard - Quantum Visualization          ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Open your browser to: http://{args.host}:{args.port}{' ' * (23 - len(args.host) - len(str(args.port)))}║
    ║                                                               ║
    ║  Features:                                                    ║
    ║  - 3D Neural Tissue Visualization (Three.js)                  ║
    ║  - Digital Terrarium with ASCII Art                           ║
    ║  - Quantum State Visualization                                ║
    ║  - Real-time Metrics and Charts                               ║
    ║  - Control Panel for Parameters                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

    # Initialize simulation state
    state.initialize_simulations()

    uvicorn.run(app, host=args.host, port=args.port)
