"""UI controls for trajectory linking and visualization."""

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go
import plotly.express as px


def setup_trajectory_ui(results, stack):
    """Set up trajectory linking and visualization UI.
    
    Parameters
    ----------
    results : dict
        Results dictionary from detection_ui containing 'tracks' and 'track_params'
    stack : np.ndarray
        The image stack array (frames × height × width)
    
    Returns
    -------
    dict
        Updated results dictionary with 'linked_trajectories' and 'link_params'
    """
    
    if results.get('tracks') is None:
        print("❌ No tracking data available. Please run 'Track Frames' first.")
        return results
    
    # Parameters for trajectory linking
    search_range_input = widgets.FloatText(
        value=5.0,
        description='Search Range:',
        style={'description_width': '120px'},
        tooltip='Maximum displacement between frames (pixels)'
    )
    
    memory_input = widgets.IntText(
        value=3,
        description='Memory:',
        style={'description_width': '120px'},
        tooltip='Frames a particle can disappear and reappear'
    )
    
    # Visualization controls
    frame_slider = widgets.IntSlider(
        value=results['track_params']['start_frame'],
        min=results['track_params']['start_frame'],
        max=results['track_params']['end_frame'],
        step=1,
        description='Frame:',
        style={'description_width': '80px'}
    )
    
    trail_length = widgets.IntSlider(
        value=10,
        min=0,
        max=50,
        step=1,
        description='Trail Length:',
        style={'description_width': '100px'}
    )
    
    show_labels = widgets.Checkbox(
        value=True,
        description='Show Particle IDs',
        style={'description_width': '120px'}
    )
    
    # Buttons
    link_plot_button = widgets.Button(description='Link & Plot', button_style='success')
    done_button = widgets.Button(description='Done & Return Results', button_style='info')
    
    # Output areas
    plot_output = widgets.Output()
    
    # Storage for linked data
    linked_data = {'linked': None, 'n_trajectories': 0}
    
    def on_link_plot_clicked(b):
        """Link detections into trajectories and plot."""
        import trackpy as tp
        
        search_range = search_range_input.value
        memory = memory_input.value
        frame_idx = frame_slider.value
        trail = trail_length.value
        show_ids = show_labels.value
        
        with plot_output:
            clear_output(wait=True)
            
            try:
                # Link trajectories if not done yet or parameters changed
                print(f"Linking trajectories...")
                print(f"  Search range: {search_range} pixels")
                print(f"  Memory: {memory} frames")
                
                tracks_df = results['tracks'].copy()
                linked = tp.link(tracks_df, search_range=search_range, memory=memory)
                
                # Count unique trajectories
                n_trajectories = linked['particle'].nunique()
                
                # Store results
                linked_data['linked'] = linked
                linked_data['n_trajectories'] = n_trajectories
                
                results['linked_trajectories'] = linked
                results['link_params'] = {
                    'search_range': search_range,
                    'memory': memory
                }
                
                print(f"✓ Linked {len(linked)} detections into {n_trajectories} trajectories\n")
                
                # Create figure
                fig = go.Figure()
                
                # Get color palette
                colors = px.colors.qualitative.Plotly + px.colors.qualitative.Set1
                
                # Plot all particle trajectories as lines
                for particle_id in linked['particle'].unique():
                    color = colors[int(particle_id) % len(colors)]
                    
                    # Get full trajectory for this particle
                    particle_traj = linked[linked['particle'] == particle_id].sort_values('frame')
                    
                    # Plot entire trajectory as a line
                    fig.add_trace(go.Scatter(
                        x=particle_traj['x'],
                        y=particle_traj['y'],
                        mode='lines',
                        line=dict(color=color, width=1),
                        name=f'Particle {int(particle_id)}',
                        hovertemplate=f'Particle {int(particle_id)}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<extra></extra>'
                    ))
                
                print(f"✓ Displayed {n_trajectories} particle trajectories")
                
                # Update layout
                fig.update_layout(
                    title=f'Particle Trajectories ({n_trajectories} particles)',
                    xaxis=dict(title='X (pixels)', scaleanchor='y', scaleratio=1),
                    yaxis=dict(title='Y (pixels)', autorange='reversed'),
                    width=900,
                    height=800,
                    hovermode='closest'
                )
                
                fig.show()
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def on_done_clicked(b):
        """Hide UI and display summary of results."""
        if linked_data['linked'] is None:
            print("⚠ No trajectories linked yet. Click 'Link & Plot' first.")
            return
        
        # Hide UI controls
        link_params.layout.display = 'none'
        viz_params.layout.display = 'none'
        
        with plot_output:
            clear_output(wait=True)
            print('='*60)
            print('TRAJECTORY LINKING COMPLETED - Results stored and ready')
            print('='*60)
            
            print(f"\n✓ Trajectory Results:")
            print(f"  Total trajectories: {linked_data['n_trajectories']}")
            print(f"  Total detections: {len(linked_data['linked'])}")
            print(f"  Parameters: {results['link_params']}")
            
            print(f"\nAccess results with:")
            print(f"  results['linked_trajectories'] - DataFrame with 'particle' column")
            print(f"  results['link_params'] - Linking parameters used")
    
    link_plot_button.on_click(on_link_plot_clicked)
    done_button.on_click(on_done_clicked)
    
    # Create UI panels
    link_params = widgets.VBox([
        widgets.HTML('<b>Trajectory Linking Parameters:</b>'),
        search_range_input,
        memory_input
    ])
    
    viz_params = widgets.VBox([
        widgets.HTML('<b>Visualization Controls:</b>'),
        frame_slider,
        trail_length,
        show_labels,
        link_plot_button,
        widgets.HTML('<br><b>When finished:</b>'),
        done_button
    ])
    
    # Display UI
    display(widgets.HBox([link_params, viz_params]))
    display(widgets.HTML('<hr>'))
    display(widgets.HTML('<b>Trajectory Output:</b>'))
    display(plot_output)
    
    return results
