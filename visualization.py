
"""
visualization
==============

This module defines utilities for generating interactive HTML reports
from the outputs of the connectivity analysis pipeline.  It makes use
of Plotly to produce interactive heatmaps, bar charts and time line
plots.  The report is assembled as a single HTML file which embeds
the Plotly figures and summarises key metrics in a human readable
format.

Usage
-----
>>> from brainnet.visualization import ReportConfig, ReportGenerator
>>> report_cfg = ReportConfig(output_dir='reports')
>>> generator = ReportGenerator(report_cfg)
>>> html_path = generator.generate(
...     subject_id='01',
...     conn_matrix=conn_matrix,
...     graph_metrics=graph_metrics,
...     dyn_model=dyn_model,
...     roi_labels=roi_labels,
...     qc_metrics={'tSNR': 50.0}
... )
>>> print(f'Report saved to {html_path}')

The resulting HTML file can be opened in any modern web browser and
provides interactive views of the connectivity matrix, state dynamics
and graph metrics.

Note
----
Plotly must be installed in the environment for this module to
function.  The code attempts to import Plotly and will raise an
exception if it is unavailable.
"""

from __future__ import annotations

import os
import html
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import plotly.graph_objs as go
    import plotly.io as pio
except Exception as exc:
    raise ImportError("Plotly is required for report generation") from exc

# Import data structures from the modular static and dynamic packages.  These
# classes are defined in the new ``brainnet.static`` and ``brainnet.dynamic``
# subpackages.  If static_analysis or dynamic_analysis are still used
# elsewhere for backwards compatibility, they will import the same
# definitions.
from .static.connectivity import ConnectivityMatrix
from .static.metrics import GraphMetrics
from .dynamic.model import DynamicStateModel


@dataclass
class ReportConfig:
    """Configuration for report generation.

    Parameters
    ----------
    output_dir : str
        Directory where HTML reports will be written.  The directory
        will be created if it does not exist.
    include_qc : bool
        Whether to include basic quality control metrics (e.g. tSNR)
        in the report.
    time_unit : str
        Label to display on the horizontal axis of the state timeline.
        Use ``'window'`` to label by window index (for sliding window
        analyses) or ``'time'`` to use raw time points (for CAP).  The
        values themselves are always indices; the string only affects
        axis labelling.
    """

    output_dir: str
    include_qc: bool = True
    time_unit: str = 'window'

    def ensure_output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)


class ReportGenerator:
    """Generate interactive HTML reports from analysis results."""

    def __init__(self, config: ReportConfig) -> None:
        self.config = config
        self.config.ensure_output_dir()

    # --------------------------------------------------------------
    def generate(
        self,
        subject_id: str,
        conn_matrix: ConnectivityMatrix,
        graph_metrics: GraphMetrics,
        dyn_model: DynamicStateModel,
        roi_labels: Sequence[str],
        qc_metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        """Create a report for a single subject and write it to disk.

        Parameters
        ----------
        subject_id : str
            Identifier for the subject, used in headings and file names.
        conn_matrix : ConnectivityMatrix
            Static connectivity matrix.
        graph_metrics : GraphMetrics
            Node and global graph metrics derived from the static
            connectivity.
        dyn_model : DynamicStateModel
            Dynamic state model containing state patterns and sequence.
        roi_labels : sequence of str
            Names of ROIs corresponding to the connectivity matrix.
        qc_metrics : dict, optional
            Quality control metrics (e.g. tSNR) to include in the report.

        Returns
        -------
        str
            Path to the generated HTML report file.
        """
        # prepare figures
        fig_conn = self._plot_connectivity_heatmap(conn_matrix)
        fig_nodes = self._plot_node_metrics(graph_metrics, roi_labels)
        fig_dyn = self._plot_dynamic_metric(dyn_model)
        fig_timeline = self._plot_state_timeline(dyn_model, self.config.time_unit)
        fig_trans = self._plot_transition_matrix(dyn_model)
        fig_occupancy = self._plot_state_occupancy(dyn_model)
        # assemble HTML parts
        sections = []
        sections.append(f"<h1>Subject {html.escape(subject_id)}</h1>")
        # QC section
        if self.config.include_qc and qc_metrics:
            qc_lines = [f"<li>{html.escape(name)}: {val:.2f}</li>" for name, val in qc_metrics.items()]
            qc_html = "<h2>Quality Metrics</h2><ul>" + "".join(qc_lines) + "</ul>"
            sections.append(qc_html)
        # connectivity
        sections.append("<h2>Static Connectivity Matrix</h2>" + self._figure_to_html(self._plot_connectivity_heatmap(conn_matrix)))
        sections.append("<h3>Node Metrics</h3>" + self._figure_to_html(self._plot_node_metrics(graph_metrics, roi_labels)))
        # dynamic
        sections.append("<h2>Dynamic Connectivity</h2>" + self._figure_to_html(self._plot_dynamic_metric(dyn_model)))
        sections.append("<h3>State Timeline</h3>" + self._figure_to_html(self._plot_state_timeline(dyn_model, self.config.time_unit)))
        sections.append("<h3>Transition Matrix</h3>" + self._figure_to_html(self._plot_transition_matrix(dyn_model)))
        sections.append("<h3>State Occupancy</h3>" + self._figure_to_html(self._plot_state_occupancy(dyn_model)))
        
        # Add dynamic brain network visualization section using external software
        sections.append("<h2>Dynamic Brain Network Visualization</h2>" + self._plot_dynamic_network_features(dyn_model))
        
        # Assemble final HTML
        html_content = "<html><head><meta charset='utf-8'><title>Report for Subject " + html.escape(subject_id) + "</title></head><body>" + "".join(sections) + "</body></html>"

        # Save the HTML report
        report_path = os.path.join(self.config.output_dir, f"report_{subject_id}.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return report_path

    # --------------------------------------------------------------
    def _figure_to_html(self, fig: go.Figure) -> str:
        """Convert a Plotly figure to an HTML div.

        The Plotly.js library itself should be included only once in
        the page.  This helper serialises the figure without
        including the script tag.  It also hides the mode bar for a
        cleaner presentation.
        """
        return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={'displayModeBar': False})

    # --------------------------------------------------------------
    def _plot_connectivity_heatmap(self, conn_matrix: ConnectivityMatrix) -> go.Figure:
        """Create heatmap of the connectivity matrix."""
        z = conn_matrix.matrix
        labels = conn_matrix.labels
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=labels,
                y=labels,
                colorscale='RdBu',
                zmin=-np.max(np.abs(z)),
                zmax=np.max(np.abs(z)),
                colorbar=dict(title='r')
            )
        )
        fig.update_layout(
            title='Connectivity matrix (Pearson correlation)',
            xaxis=dict(showgrid=False, tickfont=dict(size=10), automargin=True),
            yaxis=dict(showgrid=False, tickfont=dict(size=10), automargin=True),
            height=600,
        )
        return fig

    # --------------------------------------------------------------
    def _plot_node_metrics(self, graph_metrics: GraphMetrics, roi_labels: Sequence[str]) -> go.Figure:
        """Plot node‑wise metrics as bar charts."""
        node_metrics = graph_metrics.node_metrics
        # sort ROIs by degree descending
        degree = node_metrics.get('degree')
        indices = np.argsort(degree)[::-1]
        sorted_labels = [roi_labels[i] for i in indices]
        sorted_degree = degree[indices]
        clustering = node_metrics.get('clustering')[indices]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sorted_labels, y=sorted_degree, name='Degree'))
        fig.add_trace(go.Bar(x=sorted_labels, y=clustering, name='Clustering'))
        fig.update_layout(
            barmode='group',
            title='Node metrics (sorted by degree)',
            xaxis_title='ROI',
            yaxis_title='Metric value',
            height=400,
        )
        return fig

    # --------------------------------------------------------------
    def _plot_dynamic_metric(self, dyn_model: DynamicStateModel) -> go.Figure:
        """Plot dynamic connectivity metric across windows/time.

        For K‑means and HMM methods, the metric is the mean absolute
        connectivity per window.  For CAP, the global activation
        measure is used.  These values are stored in the ``extra``
        attribute of :class:`DynamicStateModel`.
        """
        if dyn_model.method in {'kmeans', 'hmm'}:
            metric = dyn_model.extra.get('window_metric')
            ytitle = 'Mean |r|' if dyn_model.method == 'kmeans' else 'HMM metric'
        elif dyn_model.method == 'cap':
            metric = dyn_model.extra.get('global_amp')
            ytitle = 'Global activation (z)'
        else:
            metric = None
        if metric is None:
            # return empty figure
            return go.Figure()
        x = np.arange(len(metric))
        fig = go.Figure(
            data=go.Scatter(x=x, y=metric, mode='lines'),
        )
        fig.update_layout(
            title='Dynamic connectivity metric',
            xaxis_title=f'Window index',
            yaxis_title=ytitle,
            height=300,
        )
        return fig

    # --------------------------------------------------------------
    def _plot_state_timeline(self, dyn_model: DynamicStateModel, time_unit: str) -> go.Figure:
        """Plot a timeline of discrete brain states.

        For K‑means and HMM analyses, the timeline shows the assigned
        state for each sliding window.  For CAP analysis, it shows
        state assignments for event time points only; non‑event
        time points are labelled as -1 and displayed as grey.
        """
        seq = dyn_model.state_sequence
        # map state labels to colours
        unique_states = np.unique(seq[seq >= 0])
        if unique_states.size == 0:
            return go.Figure()
        n_states = dyn_model.n_states
        # assign colours (cycle through Plotly palette)
        default_colors = [
            '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',
            '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
        ]
        colors = {s: default_colors[s % len(default_colors)] for s in range(n_states)}
        # build a heatmap with one row
        z = np.full((1, len(seq)), -1)
        z[0, :] = seq
        # create custom colorscale: map each state to its colour; -1 -> grey
        color_scale = []
        # grey for -1
        color_scale.append([0.0, '#DDDDDD'])
        color_scale.append([0.0, '#DDDDDD'])
        for s in range(n_states):
            frac = (s + 1) / (n_states + 1)
            color = colors.get(s, default_colors[s % len(default_colors)])
            color_scale.append([frac, color])
            color_scale.append([frac, color])
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                colorscale=color_scale,
                showscale=False,
            )
        )
        fig.update_layout(
            title='State timeline',
            xaxis=dict(title=f'{time_unit.capitalize()} index', showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=120,
        )
        return fig

    # --------------------------------------------------------------
    def _plot_transition_matrix(self, dyn_model: DynamicStateModel) -> go.Figure:
        """Plot the state transition probability matrix."""
        metrics = dyn_model.metrics
        mat = metrics.transition_matrix
        n = dyn_model.n_states
        labels = [f'State {i}' for i in range(n)]
        fig = go.Figure(
            data=go.Heatmap(
                z=mat,
                x=labels,
                y=labels,
                colorscale='Blues',
                zmin=0,
                zmax=mat.max() if mat.size > 0 else 1,
                colorbar=dict(title='P'),
                text=np.round(mat, 2),
                texttemplate='%{text}',
                textfont=dict(color='black')
            )
        )
        fig.update_layout(
            title='State transition probabilities',
            height=400,
        )
        return fig

    # --------------------------------------------------------------
    def _plot_state_occupancy(self, dyn_model: DynamicStateModel) -> go.Figure:
        """Plot occupancy and dwell time bar charts."""
        metrics = dyn_model.metrics
        n = dyn_model.n_states
        x = [f'State {i}' for i in range(n)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=metrics.occupancy, name='Occupancy'))
        fig.add_trace(go.Bar(x=x, y=metrics.mean_dwell_time, name='Mean dwell time'))
        fig.update_layout(
            barmode='group',
            title='State occupancy and dwell time',
            yaxis_title='Fraction / Windows',
            height=400,
        )
        return fig

    # --------------------------------------------------------------
    def _plot_dynamic_network_features(self, dyn_model: DynamicStateModel) -> str:
        """Generate dynamic brain network visualization using an external tool (e.g. BrainNetViewer).

        ``dyn_model.extra`` is expected to be a JSON‑serialisable dictionary whose
        structure is agreed upon by the calling code.  The contents are written to a
        temporary JSON file which is then consumed by an external BrainNetViewer
        process.  If ``dyn_model.extra`` is empty, a friendly message is returned
        instead of attempting the visualisation.
        """
        import tempfile
        import json
        import subprocess

        extra_data = dyn_model.extra
        if not extra_data:
            return "<p>No extra dynamic network metrics provided.</p>"

        # Write dynamic network extra metrics to a temporary JSON file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', dir=self.config.output_dir) as tmp:
            json.dump(extra_data, tmp, indent=2)
            tmp_path = tmp.name

        # Define output image file path
        output_img = tmp_path.replace('.json', '.png')

        try:
            # Call external BrainNetViewer command line tool
            # Assumes BrainNetViewer is installed and available in PATH
            subprocess.run(['brainnetviewer', '--input', tmp_path, '--output', output_img], check=True)
            html_fig = f"<img src='{output_img}' alt='Dynamic Network Visualization' style='max-width:100%;'/>"
        except Exception as e:
            html_fig = "<p>BrainNetViewer visualization is not available.</p>"
        return html_fig


__all__ = [
    'ReportConfig',
    'ReportGenerator',
]