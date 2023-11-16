import numpy as np
import matplotlib.pyplot as plt
import napari

def make_animation(stack, spots):
    # Create a function to update the viewer for each frame
    def update_frame(event):
        frame = event.value[0]
        viewer.dims.set_current_step(0,frame)
        current_spots = spots[spots['frame'] == frame]
        rfit = current_spots[['x_mle', 'y_mle']].values
        rraw = current_spots[['x', 'y']].values
        raw_layer = viewer.layers['Raw']
        fit_layer = viewer.layers['Fit']
        raw_layer.data = []; fit_layer.data = []
        raw_layer.data = rraw  # Update the points layer data
        fit_layer.data = rfit  # Update the points layer data  
        viewer.layers.selection.active = raw_layer
        viewer.layers.selection.active = fit_layer #fixes display bug

    current_spots = spots[spots['frame'] == 0]
    rfit = current_spots[['x_mle', 'y_mle']].values
    rraw = current_spots[['x', 'y']].values

    # Create a Napari viewer
    viewer = napari.Viewer()

    # Add the TIFF stack as an image layer
    viewer.add_image(stack, colormap='gray', name='Stack')

    # Set the initial step to 0 to start from frame 0
    viewer.dims.set_current_step(0,0)
    viewer.add_points(rraw, name='Raw', size=3, face_color='red', symbol='x')
    viewer.add_points(rfit, name='Fit', size=3, face_color='blue', symbol='x')

    # Create the animation by connecting the update function to the keyframes
    viewer.dims.events.current_step.connect(update_frame)

    # Show the Napari viewer
    napari.run()
