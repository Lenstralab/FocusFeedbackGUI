![cyllensgui icon](Icon.ico)

# Focus Feedback GUI
Gui to live track single particles. Written for Zeiss' Zen Black software.

# Installation
- Install Python 3.8, 3.9 or 3.10: https://www.python.org/
- Install Rust: https://rustup.rs/
- Install using pip:


    pip install git+https://github.com/Lenstralab/FocusFeedbackGUI.git

- Configure:
Edit C:\Program Files\Python3x\Lib\site-packages\focusfeedbackgui\conf.yml

# Usage
- Start Zen and configure it to acquire images and acquire at least one image, for example by pressing 'Continuous', if 
you don't do this, Focus Feedback GUI will wait until you do it.
- Run this command (you can make a shortcut to it):


    python FocusFeedbackGUI/focusfeedbackgui

The program should start and show a screen with 3 tabs.

## Program layout
The 'Main' tab shows graphs to graph the ellipticity, intensity,
width, R-squared, piezo position and particle position during the experiment. Above the graphs are controls for the
feedback, which are grayed out at the moment.
- Stay primed: keep the GUI ready and waiting for a new time-lapse experiment to start, as soon as an experiment is
started in ZEN, the GUI will start tracking.
- Center on click: When this is enabled, it is possible to click anywhere in the image in ZEN, and the GUI will instruct
ZEN to center the sample on that position.
- Prime for experiment: When this is clicked, the GUI will be ready and waiting for a new time lapse experiment to
start, in contrast to the 'Stay primed option', this will not make the GUI to be ready and waiting after one experiment.
- Stop: stop running feedback.
- Zhuang / PID: Use Zhuang or PID feedback mode. Zhuang mode is more accurate and faster, but requires calibration.

The 'Configuration' tab has important configuration options which need to be set before tracking can be started.

- Feedback channel: Current channels defined in ZEN can be enabled for feedback. Next to them the emission wavelength of
the fluorophore in that channel should be set. Multiple channels can be enabled simultaneously.
- Feedback mode: for use with feedback on multiple channels, either 'Average' where the focus is set to the average
position of all found particles, or 'Alternate' where the focus is rotated between all enabled channels.
- Cylindrical lens back/front: Which cylindrical lens is set to the back (TV1) and front (TV2) cameras on the duolink.
In case there's only one camera, only 'Cylindrical lens back' should be used. This data is used to get the correct
calibration from the configuration file.
- Duolink filterset: which filterset is currently installed in the duolink. This data will be saved as metadata with a
tracking experiment.
- Duolink filter: which of the two positions is currently set to be the active set. Enabling a disabled option will
make ZEN switch the duolink filter.
- Max stepsize: the maximum distance the GUI can move the focus each frame.
- Calibrate with beads: opens a file selector to select a file to use for calibrating the parameters used for Zhuang
mode feedback. Before this, make a z-stack (100 nm interval, 101 planes) centered on beads (for example Tetraspec, 100
nm)

The 'Map' tab will display a map of the sample. The user can see here which positions have been visited, and at which
positions an experiment has been done.

## Tracking
- Set the configuration in the 'Configuration' tab.
  - Set the cylindrical lens(es) to the correct position, both
  physically (slider on the camera) and in the 'Cylindrical lens back/front' settings. Note that each magnification has
  a different cylindrical lens.
  - Enable the desired Feedback channel(s) and set the correct emission wavelength for the enabled channel(s). The
  wavelength will be used to filter bad fits.
  - Make sure the correct duolink filter is set in the 'Duolink filterset' and 'Duolink filter' options.
- Calibrate using beads.
  - Make sure the settings in the 'Configuration' tab are correct. 
  - Put a slide with beads on the microscope.
  - Make one or more full frame z-stacks (100 nm interval, 101 planes). This stack can later also be used to correct for
  distortions caused by the cylindrical lens (daily or after each time the duolink filter is changed)
  - Calibrate by pressing the button 'Calibrate with beads' in the 'Configuration' tab.
  - When the calibration is done, check the calibration in the pdf file which is saved next to the file with the bead
  stack. 
  - Optionally save the calibration by using File>Save (Ctrl+S), it will be save in conf.yml.
- Track.
  - Configure ZEN for a time-lapse experiment.
  - Enable 'auto-save' in ZEN.
  - Enable either 'Stay primed' or click the 'Prime for experiment button'.
  - Enable 'Center on click'
  - Find an interesting particle in the sample, focus on it and center it in the white square that the GUI drew on ZEN
  by clicking on the particle, due to inaccuracies in the stage, you might have to click the particle again to center
  better.
  - Start the experiment in ZEN.
  - As soon as the GUI has found a particle it will draw an ellipse around it reflecting the fitted parameters.
  - The GUI will start to display live graphs about the particle. Dashed lines reflect limits on the fit parameters, the
  parameters need to be within limits in at least 3 of the last 5 frames to be considered reliable and be used for
  tracking.

## PZL file
With every tracking experiment a file is saved which will have the same name as the .czi file, but with a .pzl
extension. This file contains metadata about the experiment and the parameters fitted during the tracking.
- Columns: The names of the columns in 'p' down below.
- CylLens: For both channels the names of the cylindrical lenses inserted.
- DLFilterChannel: Which of the two filtersets in the duolink was active
- DLFilterSet: The name of the filterset which was inserted in the duolink
- FeedbackChannels: Which channels were used for feedback
- ROIPos: The set (see Configuration) position of the square during the experiment.
- ROISize: The size of the square.
- maxStep: The max step size during the experiment.
- mode: The mode used for fitting, 'zhuang' or 'pid'.
- q: For each enabled channel the parameters used for tracking in Zhuang mode.
- theta: The rotation of the cylindrical lens (and particle) in radians.
- p: The parameters fit during the tracking, see 'Columns' for what is written in each column.

## Configuration
The file 'conf.yml' contains configuration and calibration for the GUI. See the configuration file for more details.
Some values can be changed by the gui and then saved by using the File>Save (As) menu.
