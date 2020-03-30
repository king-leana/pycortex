#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leana King (rotation)
Pycortex Documentation Notes & Functions 

General notes: 
 - way to change scale of values on pycortex map in cortex.quickflat.make_figure
    (input for vmin and vmax, AFTER creating volume)
 - added documentation to geodesic and mapper functions 
 - Volume flatmaps require data in 8-bit format, whereas vertex flatmaps require float data
   (also, on the tutorial pg, is says 8-bit for the vertex map, which is incorrect)

For onboarding:
    - more documented set-up for cottoncandy 
    i.e. 1.) boto3 installation, THEN cotton candy 
    + where to find config files / useful linux commands 
    

Created on Mon Mar 30 08:48:48 2020
@author: leana
"""
import numpy as np
import cortex 
import cortex.polyutils
np.random.seed(1234)



# Function 1 - Re-scaling pycotrex graph
    # for specific ROI and single dimension variable 
    # TODO: make more generalizable to whole brain and multidimensional variables
def get_scaled_ROI_variables (volume_data, variable_data, subject, xfm, mask=None):
    # TODO : create check for variable dimensions to match masked or unmasked data
     """
    Re-scales pycortex graph to +/- absolute max value in ROI
    ----------
    Parameters
    volume_data : 
        output from cortex.Volume of ROI selected from cortex.utils.get_roi_masks
    variable_data : numpy array
        data to be plotted on pycortex graph in in either masked or unmasked format
    subject : str
        The subject name
    xfm : str
        The transform
    mask : 
        optional 
        volume mask for subject 
        from 3D voxels to 1D
    """
    volume_array = volume_data.data # dtype for this array is int
    volume_array = volume_array.astype(float) # dtype trying to load in is float 
    # If mask is present, then convert data to mask format
    if mask is None:
        ind = np.where((volume_array==1) | (volume_array==-1)) # get both L & R ROI
        # TODO: Array dimension check 
        for i in list(range(len(ind[0]))):
            x = ind[0][i]
            y = ind[1][i]
            z = ind[2][i]
            volume_array[x,y,z] = variable_data[x,y,z]
        data = volume_array
    else:     
        mask.shape  ## should be (30,100,100)
        mask.sum()  ## should be around 80k, varies by subject
        masked_volume = volume_array[mask]
        masked_volume.shape  ## should be (985, 80k-ish)
        ind = np.where((masked_volume==1) | (masked_volume==-1)) # get both L & R ROI
        # will have to adjust the following if data is not 3D
        for i in list(range(len(ind[0]))):
            x = ind[0][i]
            masked_volume[x] = variable_data[x]
        data = masked_volume
    # Get max and min        
    vmax = np.amax(data)
    print('Max value of variable in ROI = '+str(vmax))
    vmin = np.amin(data)    
    print('Min value of variable in ROI = '+str(vmin))
    # Set the limit the scaled graph
    if np.abs(vmax) > np.abs(vmin):
        lim = vmax
    else:
        lim = vmin 
    print('Scale of graph will be +/- '+str(lim))
    # Re-scale volume data w/ max(abs(roi variable data)) 
    roi_data_scaled = cortex.Volume(data, subject, xfm,
                             vmin=-lim, 
                             vmax=lim, 
                             cmap="RdBu_r") 

    return roi_data_scaled 




# Subset Function 
    # Get verticies of interest for a specified ROI along with extended radius outside ROI
def get_roi_verts (subject, xfm, roi, radius):
    # First we need to import the surfaces for this subject
    surfs = [cortex.polyutils.Surface(*d)
             for d in cortex.db.get_surf(subject, "fiducial")]
    all_roi = cortex.utils.get_roi_verts(subject, roi)[roi] #1D list of all verticies IN ROI
    # We have to then separate these vertices by hemisphere
    numl = surfs[0].pts.shape[0]
    roi_verts = [all_roi[all_roi < numl], all_roi[all_roi >= numl] - numl]
    # 2.)Calculate the distance of every vertex outside of the ROI by selecting the closest point on the edge and calculating the geodesic distance to that point (done seperatley for each hemisphere)
    dists = [s.geodesic_distance(verts) for s, verts in zip(surfs, roi_verts)]
    all_dists = np.hstack((dists[0], dists[1])) # ~150,000 vertices total for each hemisphere => ~300,000 total 
    # Visualize these distances onto the cortical surface
    #dist_map = cortex.Vertex(all_dists, subject, cmap="hot") #all_dists<radi_edge
    #cortex.quickshow(dist_map)
    #plt.show()
    # 3.) Get extended radius outside ROI
    radi_edge = (max(all_dists))*radius
    # Select for vertices within that edge 
    radi_ind = np.where(all_dists<radi_edge) # ~ 12,000 vertices 
    # Verticies of interest (roi + extended raius from edges)
    vois = radi_ind[0]
    return vois





# Function 2 - Graph PCA's of a single ROI (w/ or w/out extended radius)
# Get PCA values for ONLY ROI and radius around ROI selected 
    # data must be in form of 3PCs x nvoxels 
    # Useful for graphing PCAs of a single ROI
def get_roi_pca_verts (subject, xfm, roi, radius, variable):
    # Verticies of interest (roi + extended raius from edges)
    vois = get_roi_verts(subject, xfm, roi, radius)
    
    # 4.) Convert volume data into vertex data
    voxel_data = variable
    voxel_vol = cortex.Volume(voxel_data, subject, xfm) #mask=mask)
    # Then we have to get a mapper from voxels to vertices for this transform
        # does the mapper not work unless data is in [x,y,z] format?? 
    mapper = cortex.get_mapper(subject, xfm, 'line_nearest', recache=True)
    # Just pass the voxel data through the mapper to get vertex data
    vertex_map = mapper(voxel_vol)
    vertex_data = vertex_map.data # array shape = 985 x 304380
    # outputs variable values for each vertice (~300,000 total)
    vertex_plot_data = vertex_data
    # 5.) Zero-out non-Vertices of Interest
    # loop through all features 
    for f in list(range(len(vertex_plot_data))):
        for i in list(range(len(vertex_plot_data[0]))):
            # VOIS are the verticies of interest we want to plot
            if i not in vois:
                vertex_plot_data[f,i] = 0
    # convert dtype to float64
    vertex_plot_data = vertex_plot_data.astype(np.float64)

    return vertex_plot_data



# Function 3 - Get feature space for ONLY ROI selected (w/ or w/out extended radius)
    # output is an array with nan in places of non-ROI vertices
def get_roi_feature_verts (subject, xfm, roi, radius, variable):
    # Verticies of interest (roi + extended raius from edges)
    vois = get_roi_verts(subject, xfm, roi, radius)
    
    # 4.) Convert volume data into vertex data
    variable = masked_wts #[985, nvox]
    voxel_data = variable
    voxel_vol = cortex.Volume(voxel_data, subject, xfm) #mask=mask)
    # Then we have to get a mapper from voxels to vertices for this transform
        # does the mapper not work unless data is in [x,y,z] format?? 
    mapper = cortex.get_mapper(subject, xfm, 'line_nearest', recache=True)
    # Just pass the voxel data through the mapper to get vertex data
    vertex_map = mapper(voxel_vol)
    vertex_data = vertex_map.data # array shape = 985 x 304380
    # outputs variable values for each vertice (~300,000 total)
    vertex_plot_data = vertex_data
    # 5.) Zero-out non-Vertices of Interest
    # loop through all features 
    for f in list(range(len(vertex_plot_data))):
        for i in list(range(len(vertex_plot_data[0]))):
            # VOIS are the verticies of interest we want to plot
            if i not in vois:
                vertex_plot_data[f,i] = np.nan
    
    return vertex_plot_data



