# Capturing Calibration Images

The calibration pattern must be a non-square UV LED grid, where the LED markers act
like the internal corners of a checkerboard grid. The grid should be non-square and
composed of LEDs emitting in the 395nm range.

An example of such a grid:

![UV LED grid pattern, unlit](../.fig/led_grid_off.jpeg)
![UV LED grid pattern, illuminated](../.fig/led_grid_on.jpeg)
![UV LED grid pattern, backside/mounting](../.fig/led_grid_backside.jpeg)

If the camera is constructed correctly, the output image of the pattern should look
like this:

![Example camera view of the UV LED grid](../.fig/camera_view.jpg)

## Capturing Good Calibration Images

For good calibration, the UV LED grid must appear in many different parts of the
image.

Capture images where:

- the full UV LED grid is visible
- all LEDs are detected clearly
- LED blobs are small and not saturated
- the grid appears near the image center
- the grid appears near the left side of the image
- the grid appears near the right side of the image
- the grid appears near the top of the image
- the grid appears near the bottom of the image
- the grid appears near the image corners
- the grid appears at different distances from the camera
- the grid is tilted in different directions

Hold the grid still for a moment in each position so a sharp frame is captured.
Diversity matters more than the number of samples; the GUI tells you when the set
covers enough (at least 20 accepted samples spanning all four ranges).
