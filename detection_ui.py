"""UI controls for particle detection and tracking."""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

from detect import detect_frame
from detect_utils import track_frames


def setup_detection_ui(stack, stack_shape):
    """Set up all detection and tracking UI controls.
    
    Parameters
    ----------
    stack : np.ndarray or None
        The image stack array (frames × height × width)
    stack_shape : tuple or None
        Shape of the stack (n_frames, height, width)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'tracks': DataFrame with tracked detections (or None)
        - 'track_params': dict of parameters used for tracking (or None)
        - 'last_detection': DataFrame from last single-frame detection (or None)
        - 'detect_params': dict of parameters used for last detection (or None)
    """
    
    # Storage for results that will be returned
    results = {
        'tracks': None,
        'track_params': None,
        'last_detection': None,
        'detect_params': None
    }
    
    # Single-frame parameters as text inputs
    diameter_input = widgets.FloatText(
        value=5,
        description='Diameter:',
        style={'description_width': '100px'}
    )
    
    separation_input = widgets.FloatText(
        value=3,
        description='Separation:',
        style={'description_width': '100px'}
    )
    
    minmass_input = widgets.FloatText(
        value=0,
        description='Minmass (0=auto):',
        style={'description_width': '150px'}
    )
    
    # Frame tracking parameters
    start_frame_input = widgets.IntText(
        value=0,
        description='Start Frame:',
        style={'description_width': '100px'}
    )
    
    end_frame_input = widgets.IntText(
        value=50,
        description='End Frame:',
        style={'description_width': '100px'}
    )
    
    # Single-frame selection slider
    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=120,
        step=1,
        description='Frame:',
        style={'description_width': '80px'}
    )
    
    # Contrast sliders
    vmin_slider = widgets.FloatSlider(
        value=0,
        min=0,
        max=1000,
        step=10,
        description='Vmin:',
        style={'description_width': '80px'}
    )
    
    vmax_slider = widgets.FloatSlider(
        value=1000,
        min=0,
        max=10000,
        step=100,
        description='Vmax:',
        style={'description_width': '80px'}
    )
    
    # Buttons
    run_button = widgets.Button(description='Run Detection', button_style='success')
    track_button = widgets.Button(description='Track Frames', button_style='warning')
    done_button = widgets.Button(description='Done & Return Results', button_style='info')
    
    # Output areas
    detection_output = widgets.Output()
    track_output = widgets.Output()
    
    # Adjust slider ranges based on first frame
    if stack is not None:
        first_frame = stack[0]
        frame_max = float(first_frame.max())
        vmax_slider.max = frame_max * 3
        vmax_slider.value = min(frame_max, frame_max * 3)
        frame_slider.max = stack_shape[0] - 1
        end_frame_input.value = stack_shape[0] - 1
        vmax_slider.value = frame_max
        vmin_slider.max = frame_max
    
    def on_run_clicked(b):
        if stack is None:
            print('Please load a TIFF file first.')
            return
        
        frame_idx = frame_slider.value
        diameter = diameter_input.value
        separation = separation_input.value
        minmass = minmass_input.value if minmass_input.value > 0 else None
        vmin = vmin_slider.value
        vmax = vmax_slider.value
        
        with detection_output:
            clear_output(wait=True)
            
            try:
                frame = stack[frame_idx]
                pts = detect_frame(
                    frame,
                    method='trackpy',
                    diameter=diameter,
                    separation=separation,
                    minmass=minmass
                )
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 10), 
                                          gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 1]})
                
                # Image with detections (left, spanning both rows)
                ax_img = axes[0, 0]
                ax_img.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax)
                if len(pts) > 0:
                    ax_img.scatter(pts['x'], pts['y'], s=50, facecolors='none', edgecolors='lime', linewidth=2)
                ax_img.set_title(f'Frame {frame_idx} - {len(pts)} particles detected')
                ax_img.set_xlabel('X')
                ax_img.set_ylabel('Y')
                
                # X fractional distribution (top right)
                ax_x = axes[0, 1]
                if len(pts) > 0:
                    frac_x = np.mod(pts['x'].values, 1)
                    ax_x.hist(frac_x, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
                    ax_x.axvline(0.5, color='r', linestyle='--', label='Expected (0.5)')
                    ax_x.set_title(f'X fractional dist (mean={np.mean(frac_x):.3f})')
                    ax_x.set_xlabel('Fractional part')
                    ax_x.legend()
                
                # Y fractional distribution (bottom right)
                ax_y = axes[1, 1]
                if len(pts) > 0:
                    frac_y = np.mod(pts['y'].values, 1)
                    ax_y.hist(frac_y, bins=40, edgecolor='black', alpha=0.7, color='coral')
                    ax_y.axvline(0.5, color='r', linestyle='--', label='Expected (0.5)')
                    ax_y.set_title(f'Y fractional dist (mean={np.mean(frac_y):.3f})')
                    ax_y.set_xlabel('Fractional part')
                    ax_y.legend()
                
                # Hide bottom left subplot
                axes[1, 0].axis('off')
                
                plt.tight_layout()
                display(fig)
                plt.close(fig)
                plt.close('all')
                
                print(f'Detection Parameters:')
                print(f'  Diameter: {diameter}')
                print(f'  Separation: {separation}')
                print(f'  Minmass: {minmass if minmass is not None else "auto-estimated"}')
                print(f'\nDetections: {len(pts)} particles')
                if len(pts) > 0:
                    print(f'X range: [{pts["x"].min():.1f}, {pts["x"].max():.1f}]')
                    print(f'Y range: [{pts["y"].min():.1f}, {pts["y"].max():.1f}]')
                    print(f'Mass range: [{pts["mass"].min():.0f}, {pts["mass"].max():.0f}]')
                
                # Store results and parameters
                results['last_detection'] = pts
                results['detect_params'] = {
                    'diameter': diameter,
                    'separation': separation,
                    'minmass': minmass,
                    'frame': frame_idx
                }
            
            except Exception as e:
                print(f'Error running detection: {str(e)}')
                import traceback
                traceback.print_exc()
    
    def on_track_clicked(b):
        if stack is None:
            print('Please load a TIFF file first.')
            return
        
        start_frame = int(start_frame_input.value)
        end_frame = int(end_frame_input.value)
        diameter = diameter_input.value
        separation = separation_input.value
        minmass = minmass_input.value if minmass_input.value > 0 else None
        
        with track_output:
            clear_output(wait=True)
            
            try:
                print(f'Tracking frames {start_frame} to {end_frame}...')
                print(f'Parameters: diameter={diameter}, separation={separation}, minmass={minmass}')
                tracks = track_frames(
                    stack,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    diameter=diameter,
                    separation=separation,
                    minmass=minmass,
                    use_batch=True,
                    processes=None
                )
                
                print(f'✓ Tracked {len(tracks)} detections across {end_frame - start_frame + 1} frames')
                print(f'  X range: [{tracks["x"].min():.1f}, {tracks["x"].max():.1f}]')
                print(f'  Y range: [{tracks["y"].min():.1f}, {tracks["y"].max():.1f}]')
                print(f'  Mass range: [{tracks["mass"].min():.0f}, {tracks["mass"].max():.0f}]')
                
                # Calculate and display pixel bias stats
                if len(tracks) > 0:
                    frac_x = np.mod(tracks['x'].values, 1)
                    frac_y = np.mod(tracks['y'].values, 1)
                    
                    print(f'\nPixel Bias Statistics:')
                    print(f'  X fractional: mean={np.mean(frac_x):.3f}, std={np.std(frac_x):.3f}')
                    print(f'  Y fractional: mean={np.mean(frac_y):.3f}, std={np.std(frac_y):.3f}')

                    # Histograms for tracked fractional positions
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    axes[0].hist(frac_x, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
                    axes[0].axvline(0.5, color='r', linestyle='--', label='Expected (0.5)')
                    axes[0].set_title(f'Tracked X fractional (mean={np.mean(frac_x):.3f})')
                    axes[0].set_xlabel('Fractional part')
                    axes[0].legend()

                    axes[1].hist(frac_y, bins=40, edgecolor='black', alpha=0.7, color='coral')
                    axes[1].axvline(0.5, color='r', linestyle='--', label='Expected (0.5)')
                    axes[1].set_title(f'Tracked Y fractional (mean={np.mean(frac_y):.3f})')
                    axes[1].set_xlabel('Fractional part')
                    axes[1].legend()

                    plt.tight_layout()
                    display(fig)
                    plt.close(fig)
                    plt.close('all')
                
                # Store results and parameters
                results['tracks'] = tracks
                results['track_params'] = {
                    'diameter': diameter,
                    'separation': separation,
                    'minmass': minmass,
                    'start_frame': start_frame,
                    'end_frame': end_frame
                }
            
            except Exception as e:
                print(f'Error tracking frames: {str(e)}')
                import traceback
                traceback.print_exc()
    
    def on_done_clicked(b):
        """Hide UI and signal that results are ready."""
        # Hide all UI elements
        detection_params.layout.display = 'none'
        tracking_params.layout.display = 'none'
        
        # Display summary in track_output
        with track_output:
            clear_output(wait=True)
            print('='*60)
            print('UI COMPLETED - Results stored and ready to use')
            print('='*60)
            
            if results['tracks'] is not None:
                print(f"\n✓ Tracked Results:")
                print(f"  Frames: {len(results['tracks'])} detections")
                print(f"  Parameters: {results['track_params']}")
            else:
                print(f"\n  No tracking results (Track Frames not run)")
            
            if results['last_detection'] is not None:
                print(f"\n✓ Last Detection:")
                print(f"  Frame {results['detect_params']['frame']}: {len(results['last_detection'])} particles")
                print(f"  Parameters: {results['detect_params']}")
            else:
                print(f"\n  No detection results (Run Detection not run)")
            
            print(f"\nAccess results in notebook with: results['tracks'], results['track_params']")
    
    run_button.on_click(on_run_clicked)
    track_button.on_click(on_track_clicked)
    done_button.on_click(on_done_clicked)
    
    # Create parameter control panels side by side
    detection_params = widgets.VBox([
        widgets.HTML('<b>Single-Frame Detection:</b>'),
        diameter_input,
        separation_input,
        minmass_input,
        widgets.HTML('<b>Frame Selection:</b>'),
        frame_slider,
        widgets.HTML('<b>Image Contrast:</b>'),
        vmin_slider,
        vmax_slider,
        run_button
    ])
    
    tracking_params = widgets.VBox([
        widgets.HTML('<b>Multi-Frame Tracking:</b>'),
        start_frame_input,
        end_frame_input,
        track_button,
        widgets.HTML('<br><b>When finished:</b>'),
        done_button
    ])
    
    display(widgets.HBox([detection_params, tracking_params]))
    display(widgets.HTML('<hr>'))
    display(widgets.HTML('<b>Detection Output:</b>'))
    display(detection_output)
    display(widgets.HTML('<b>Tracking Output:</b>'))
    display(track_output)
    
    return results
