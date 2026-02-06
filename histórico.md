Registro de Experimentos:

#Exp. 1: T2.yaml
Mudar: 
commands:
- lin_vel_x: [0.5, 3.0] 
- gait_frequency: [2.5, 4.0] 
rewards:
- scales:
- - tracking_lin_vel_x: 2.0
- - feet_swing: 5.0
- - action_rate: -0.5
- base_height_target: 0.65
