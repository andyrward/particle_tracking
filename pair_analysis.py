import numpy as np
import pandas as pd


def analyze_particle_pairs(results, n_pairs, pixel_size=110.0):
    """
    Analyze positional fluctuations between random particle pairs.
    
    Parameters:
    -----------
    results : dict
        Results dictionary containing 'linked_trajectories'
    n_pairs : int
        Number of random particle pairs to analyze
    pixel_size : float
        Size of one pixel in nm (default: 110.0)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: particle_1, particle_2, sigma_dx_nm, sigma_dy_nm, n_frames
    """
    if 'linked_trajectories' not in results:
        print("Run trajectory linking first!")
        return None
    
    linked = results['linked_trajectories']
    
    # Remove any duplicate (particle, frame) entries - keep first occurrence
    linked = linked.drop_duplicates(subset=['particle', 'frame'], keep='first')
    
    # Find particles present in all frames
    total_frames = linked['frame'].nunique()
    frame_counts = linked.groupby('particle')['frame'].nunique()
    particles_all_frames = frame_counts[frame_counts == total_frames].index.values
    
    if len(particles_all_frames) < 2:
        print(f"Need at least 2 particles present in all frames, found {len(particles_all_frames)}")
        return None
    
    print(f"Found {len(particles_all_frames)} particles present in all {total_frames} frames")
    
    # Randomly select pairs from particles present in all frames
    np.random.seed(42)  # For reproducibility
    pair_stats = []
    
    # Filter to only keep particles present in all frames
    linked_filtered = linked[linked['particle'].isin(particles_all_frames)].copy()
    
    for _ in range(n_pairs):
        # Select two different particles from those present in all frames
        p1, p2 = np.random.choice(particles_all_frames, size=2, replace=False)
        
        # Get trajectories - already guaranteed to have all frames
        traj1 = linked_filtered[linked_filtered['particle'] == p1].set_index('frame')[['x', 'y']]
        traj2 = linked_filtered[linked_filtered['particle'] == p2].set_index('frame')[['x', 'y']]
        
        # Verify no duplicate indices after set_index
        if traj1.index.duplicated().any() or traj2.index.duplicated().any():
            print(f"WARNING: Duplicates found for particles {p1} or {p2} after duplicate removal!")
            continue
        
        # Join on frame index - should have exactly total_frames rows
        common = traj1.join(traj2, how='inner', lsuffix='_1', rsuffix='_2')
        
        # Verify we have all frames
        if len(common) != total_frames:
            print(f"WARNING: Pair ({p1}, {p2}) has {len(common)} frames, expected {total_frames}")
            continue
        
        # Calculate deltas
        delta_x = common['x_2'] - common['x_1']
        delta_y = common['y_2'] - common['y_1']
        
        # Calculate standard deviations and convert to nm
        sigma_dx_nm = delta_x.std() * pixel_size
        sigma_dy_nm = delta_y.std() * pixel_size
        
        pair_stats.append({
            'particle_1': int(p1),
            'particle_2': int(p2),
            'sigma_dx_nm': sigma_dx_nm,
            'sigma_dy_nm': sigma_dy_nm,
            'n_frames': len(common)
        })
    
    return pd.DataFrame(pair_stats)
