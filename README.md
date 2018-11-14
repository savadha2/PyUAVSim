# PyUAVSim
Light weight Python simulator for small UAVs


Completed Feature List:
1. Autopilot for fixed wing
2. Sensors (GPS, IMU) with error models
3. Path following and Path manager implemented
4. Changed interfaces so that Users can write apps with the base classes provided

The simulator is near complete with all the necessary bare minimum functional implementation. It should be straightforward to extend its functionality. 

Extensions:

1. State estimation using kalman filter/variants
2. Pathplanner class with RRT 
3. Wind models is not there due to their slightly complex nature - this will be added at the very end
4. support for quadrotos


Reference: Small Unmanned Arcraft - Theory and Practice by Randal W. Beard and Timothy W. McLain
