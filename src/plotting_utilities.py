import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots_new', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels):
    if 'TransNAS_TSAD' in name:
        y_true = torch.roll(y_true, 1, 0)

    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/output.pdf')

    for dim in range(y_true.shape[1]):
        y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim}')
        ax1.plot(smooth(y_t), linewidth=0.2, label='True')
        ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
        ax3 = ax1.twinx()
        ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
        ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
        if dim == 0:
            ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        ax2.plot(smooth(a_s), linewidth=0.2, color='g')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')

        # Save the figure to PDF
        pdf.savefig(fig)

        # Display the figure in the notebook
        plt.show()

    pdf.close()

import plotly.graph_objs as go

def customize_pareto_front(study, target_names, width=700, height=700):
    fig = plot_pareto_front(study, target_names=target_names)
    fig.update_layout(
        title='Pareto Front ',
        title_font_size=20,
        font=dict(size=16),
        xaxis_title=target_names[0],
        yaxis_title=target_names[1],
        legend_title_text='Trial',  # Customize the legend title
        legend_orientation="h",     # Horizontal legend
        legend=dict(x=0, y=2.1),    # Legend position
        hovermode='closest',        # Show hover for closest point
        width=width,                # Set width from parameter
        height=height,              # Set height from parameter
        margin=dict(l=50, r=50, t=50, b=50)  # Set custom margins (optional)
    )
    fig.update_traces(
        marker=dict(size=10, line=dict(width=4)),  # Marker style
        selector=dict(mode='markers')  # Apply updates to marker traces
    )

    fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',  # Keep plot background transparent
    paper_bgcolor='white',  # White background for the entire figure
    )

    # Cool color scheme for the axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightblue')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightblue')

    # Construct the file path for saving the figure
    output_directory = "config.dataset/output"  # Assuming config.dataset is a directory path
    os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist
    file_path = os.path.join(output_directory, "pareto_front.html")  # Construct the file path

    # Save the figure as an HTML file
    fig.write_html(file_path)

    return fig

def customize_optimization_history(study, target, title, width=700, height=700):
    fig = plot_optimization_history(study, target=target)
    fig.update_layout(
        title=title,
        title_font_size=20,
        font=dict(size=16),
        xaxis_title='Trial',
        yaxis_title='Objective Value',
        width=width,                # Set width from parameter
        height=height,              # Set height from parameter
    )
    return fig

def customize_param_importances(study, target, title, width=800, height=600):
    fig = plot_param_importances(study, target=target)
    fig.update_layout(
        title=title,
        title_font_size=20,
        font=dict(size=16),
        width=width,
        height=height,
        margin=dict(l=50, r=50, t=50, b=50),  # Adjust margins to fit the layout
        plot_bgcolor='white',  # White background for readability
        paper_bgcolor='white',  # White paper for a clean look
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey')  # Gridlines for better readability
    )

    # Customize the bar color for better visibility
    fig.update_traces(marker_color='SkyBlue', marker_line_color='RoyalBlue', marker_line_width=1.5, opacity=0.8)

    # Add hover template for more information on hover
    fig.update_traces(hovertemplate='%{y}: %{x}')

    return fig

def customize_contour_plot(study, params, target, width=800, height=600):
    fig = plot_contour(study, params=params, target=target)
    fig.update_layout(
        title='Contour Plot',
        title_font_size=20,
        font=dict(size=16),
        width=width,
        height=height,
    )
    return fig

def customize_parallel_coordinate(study, target):
    fig = plot_parallel_coordinate(study, target=target)
    fig.update_layout(
        title='Parallel Coordinate Plot',
        title_font_size=20,
        font=dict(size=16)
    )
    return fig

def customize_edf_plot(study, target, width=800, height=600):
    fig = plot_edf(study, target=target)
    fig.update_layout(
        title='Empirical Distribution Function Plot',
        title_font_size=20,
        font=dict(size=16),
        width=width,
        height=height,
    )
    return fig
