
NAMES = {
    'case_id_[-]': 'Case ID',
    'ws_[m/s]': "Wind Speed",
    'ti_[-]': "Turbulence Intensity",
    'alpha_[-]': "Shear Exponent",
    'yaw_[deg]': "Yaw Angle",
    'parked_[bool]': "Parked Flag",
    'power_[kW]': "Average Power",
    'rpm_[rpm]': "Average RPM",
    'pitch_[deg]': "Average blade pitch",
    'm10_del_blew_1_[kN-m]': "Blade Edgewise DEL for blade 1, with Wohler exponent 10",
    'm10_del_blew_2_[kN-m]': "Blade Edgewise DEL for blade 2, with Wohler exponent 10",
    'm10_del_blew_3_[kN-m]': "Blade Edgewise DEL for blade 3, with Wohler exponent 10",
    'm10_del_blew_avg_[kN-m]': "Average Blade Edgewise DEL, with Wohler exponent 10",
    'm10_del_blfw_1_[kN-m]': "Blade Flapwise DEL for blade 1, with Wohler exponent 10",
    'm10_del_blfw_2_[kN-m]': "Blade Flapwise DEL for blade 2, with Wohler exponent 10",
    'm10_del_blfw_3_[kN-m]': "Blade Flapwise DEL for blade 3, with Wohler exponent 10",
    'm10_del_blfw_avg_[kN-m]': "Average Blade Flapwise DEL, with Wohler exponent 10",
    'm10_del_ttyaw_[kN-m]': "Tower Top Yaw DEL, with Wohler exponent 10",
    'm10_del_tbss_[kN-m]': "Tower Base Side-to-Side DEL, with Wohler exponent 10",
    'm10_del_tbfa_[kN-m]': "Tower Base Fore-Aft DEL, with Wohler exponent 10",
    'm9_del_blew_1_[kN-m]': "Blade Edgewise DEL for blade 1, with Wohler exponent 9",
    'm9_del_blew_2_[kN-m]': "Blade Edgewise DEL for blade 2, with Wohler exponent 9",
    'm9_del_blew_3_[kN-m]': "Blade Edgewise DEL for blade 3, with Wohler exponent 9",
    'm9_del_blew_avg[kN-m]': "Average Blade Edgewise DEL, with Wohler exponent 9",
    'm9_del_blfw_1_[kN-m]': "Blade Flapwise DEL for blade 1, with Wohler exponent 9",
    'm9_del_blfw_2_[kN-m]': "Blade Flapwise DEL for blade 2, with Wohler exponent 9",
    'm9_del_blfw_3_[kN-m]': "Blade Flapwise DEL for blade 3, with Wohler exponent 9",
    'm9_del_blfw_avg_[kN-m]': "Average Blade Flapwise DEL, with Wohler exponent 9",
    'm9_del_ttyaw_[kN-m]': "Tower Top Yaw DEL, with Wohler exponent 9",
    'm9_del_tbss_[kN-m]': "Tower Base Side-to-Side DEL, with Wohler exponent 9",
    'm9_del_tbfa_[kN-m]': "Tower Base Fore-Aft DEL, with Wohler exponent 9",
    'm7_del_blew_1_[kN-m]': "Blade Edgewise DEL for blade 1, with Wohler exponent 7",
    'm7_del_blew_2_[kN-m]': "Blade Edgewise DEL for blade 2, with Wohler exponent 7",
    'm7_del_blew_3_[kN-m]': "Blade Edgewise DEL for blade 3, with Wohler exponent 7",
    'm7_del_blew_avg[kN-m]': "Average Blade Edgewise DEL, with Wohler exponent 7",
    'm7_del_blfw_1_[kN-m]': "Blade Flapwise DEL for blade 1, with Wohler exponent 7",
    'm7_del_blfw_2_[kN-m]': "Blade Flapwise DEL for blade 2, with Wohler exponent 7",
    'm7_del_blfw_3_[kN-m]': "Blade Flapwise DEL for blade 3, with Wohler exponent 7",
    'm7_del_blfw_avg_[kN-m]': "Average Blade Flapwise DEL, with Wohler exponent 7",
    'm7_del_ttyaw_[kN-m]': "Tower Top Yaw DEL, with Wohler exponent 7",
    'm7_del_tbss_[kN-m]': "Tower Base Side-to-Side DEL, with Wohler exponent 7",
    'm7_del_tbfa_[kN-m]': "Tower Base Fore-Aft DEL, with Wohler exponent 7",
    'm4_del_blew_1_[kN-m]': "Blade Edgewise DEL for blade 1, with Wohler exponent 4",
    'm4_del_blew_2_[kN-m]': "Blade Edgewise DEL for blade 2, with Wohler exponent 4",
    'm4_del_blew_3_[kN-m]': "Blade Edgewise DEL for blade 3, with Wohler exponent 4",
    'm4_del_blew_avg[kN-m]': "Average Blade Edgewise DEL, with Wohler exponent 4",
    'm4_del_blfw_1_[kN-m]': "Blade Flapwise DEL for blade 1, with Wohler exponent 4",
    'm4_del_blfw_2_[kN-m]': "Blade Flapwise DEL for blade 2, with Wohler exponent 4",
    'm4_del_blfw_3_[kN-m]': "Blade Flapwise DEL for blade 3, with Wohler exponent 4",
    'm4_del_blfw_avg_[kN-m]': "Average Blade Flapwise DEL, with Wohler exponent 4",
    'm4_del_ttyaw_[kN-m]': "Tower Top Yaw DEL, with Wohler exponent 4",
    'm4_del_tbss_[kN-m]': "Tower Base Side-to-Side DEL, with Wohler exponent 4",
    'm4_del_tbfa_[kN-m]': "Tower Base Fore-Aft DEL, with Wohler exponent 4",
    'm3_del_blew_1_[kN-m]': "Blade Edgewise DEL for blade 1, with Wohler exponent 3",
    'm3_del_blew_2_[kN-m]': "Blade Edgewise DEL for blade 2, with Wohler exponent 3",
    'm3_del_blew_3_[kN-m]': "Blade Edgewise DEL for blade 3, with Wohler exponent 3",
    'm3_del_blew_avg[kN-m]': "Average Blade Edgewise DEL, with Wohler exponent 3",
    'm3_del_blfw_1_[kN-m]': "Blade Flapwise DEL for blade 1, with Wohler exponent 3",
    'm3_del_blfw_2_[kN-m]': "Blade Flapwise DEL for blade 2, with Wohler exponent 3",
    'm3_del_blfw_3_[kN-m]': "Blade Flapwise DEL for blade 3, with Wohler exponent 3",
    'm3_del_blfw_avg_[kN-m]': "Average Blade Flapwise DEL, with Wohler exponent 3",
    'm3_del_ttyaw_[kN-m]': "Tower Top Yaw DEL, with Wohler exponent 3",
    'm3_del_tbss_[kN-m]': "Tower Base Side-to-Side DEL, with Wohler exponent 3",
    'm3_del_tbfa_[kN-m]': "Tower Base Fore-Aft DEL, with Wohler exponent 3",
    'measured_ws_[m/s]': "Measured Wind Speed",
    'rotor_thrust_[kN]': "Average Rotor Thrust",
    'rotor_torque_[kN-m]': "Average Rotor Torque",
    'cp_[-]': "Average Power Coefficient",
    'cq_[-]': "Average Torque Coefficient",
    'ct_[-]': "Average Thrust Coefficient",
    'rotor_area_[m^2]': "Rotor Swept Area",
    'tsr_[-]': "Tip Speed Ratio",
    'aero_power_[W]': "Average Aerodynamic Power",
    'rotor_avg_u_[m/s]': "Average Rotor Axial Wind Speed",
    'rotor_avg_v_[m/s]': "Average Rotor Tangential Wind Speed",
    'rotor_avg_w_[m/s]': "Average Rotor Vertical Wind Speed",
    'hss_shaft_pwr_[kW]': "Average High-Speed Shaft Power",
    'lss_tip_mys_[kN-m]': "Low-Speed Shaft Tip Moment Y",
    'lss_tip_mzs_[kN-m]': "Low-Speed Shaft Tip Moment Z",
    'lss_shft_fys_[kN]': "Low-Speed Shaft Shear Force (non-rotating) Y",
    'lss_shft_fzs_[kN]': "Low-Speed Shaft Shear Force (non-rotating) Z",
    'lss_shft_fxa_[kN]': "Low-Speed Shaft Thrust Force",
    'lss_shft_fya_[kN]': "Low-Speed Shaft Shear Force",
    'root_fxb1_[kN]': "Flapwise shear force at blade root blade 1",
    'root_fxb2_[kN]': "Flapwise shear force at blade root blade 2",
    'root_fxb3_[kN]': "Flapwise shear force at blade root blade 3",
    'root_fxc1_[kN]': "Axial force at blade root blade 1",
    'root_fxc2_[kN]': "Axial force at blade root blade 2",
    'root_fxc3_[kN]': "Axial force at blade root blade 3",
    'root_fyb1_[kN]': "Edgewise shear force at blade root blade 1",
    'root_fyb2_[kN]': "Edgewise shear force at blade root blade 2",
    'root_fyb3_[kN]': "Edgewise shear force at blade root blade 3",
    'root_fyc1_[kN]': "In-plane shear force at blade root blade 1",
    'root_fyc2_[kN]': "In-plane shear force at blade root blade 2",
    'root_fyc3_[kN]': "In-plane shear force at blade root blade 3",
    'root_fzb1_[kN]': "Axial force at blade root blade 1",
    'root_fzb2_[kN]': "Axial force at blade root blade 2",
    'root_fzb3_[kN]': "Axial force at blade root blade 3",
    'root_fzc1_[kN]': "Axial force at blade root blade 1",
    'root_fzc2_[kN]': "Axial force at blade root blade 2",
    'root_fzc3_[kN]': "Axial force at blade root blade 3",
    'yaw_br_fzn_[kN]': "Yaw Bearing Axial Force",
    'm10_del_pitchwise_1_[kN-m]': "Pitchwise DEL for blade 1, with Wohler exponent 10",
    'm10_del_pitchwise_2_[kN-m]': "Pitchwise DEL for blade 2, with Wohler exponent 10",
    'm10_del_pitchwise_3_[kN-m]': "Pitchwise DEL for blade 3, with Wohler exponent 10",
    'm10_del_pitchwise_avg_[kN-m]': "Average Pitchwise DEL, with Wohler exponent 10",
    'm7_del_pitchwise_1_[kN-m]': "Pitchwise DEL for blade 1, with Wohler exponent 7",
    'm7_del_pitchwise_2_[kN-m]': "Pitchwise DEL for blade 2, with Wohler exponent 7",
    'm7_del_pitchwise_3_[kN-m]': "Pitchwise DEL for blade 3, with Wohler exponent 7",
    'm7_del_pitchwise_avg_[kN-m]': "Average Pitchwise DEL, with Wohler exponent 7",
    'm4_del_pitchwise_1_[kN-m]': "Pitchwise DEL for blade 1, with Wohler exponent 4",
    'm4_del_pitchwise_2_[kN-m]': "Pitchwise DEL for blade 2, with Wohler exponent 4",
    'm4_del_pitchwise_3_[kN-m]': "Pitchwise DEL for blade 3, with Wohler exponent 4",
    'm4_del_pitchwise_avg_[kN-m]': "Average Pitchwise DEL, with Wohler exponent 4",
    'm7_del_yaw_br_mxp_[kN-m]': "Yaw Bearing Roll Moment DEL, with Wohler exponent 7",
    'm7_del_yaw_br_myp_[kN-m]': "Yaw Bearing Pitch Moment DEL, with Wohler exponent 7",
    'm7_del_twr_bs_mzt_[kN-m]': "Tower Base Yaw Moment DEL, with Wohler exponent 7",
    'm4_delyaw_br_mxp_[kN-m]': "Yaw Bearing Roll Moment DEL, with Wohler exponent 4",
    'm4_del_yaw_br_myp_[kN-m]': "Yaw Bearing Pitch Moment DEL, with Wohler exponent 4",
    'm4_del_twr_bs_mzt_[kN-m]': "Tower Base Yaw Moment DEL, with Wohler exponent 4",
}

SYMBOLS = {
    'yaw_[deg]': r'$\gamma [^\circ]$',
    'ws_[m/s]': r'$V_w [m/s]$',
    'power_[kW]': r'$P [kW]$',
    'ct_[-]': r'$C_T [-]$',
    'm10_del_blew_avg_[kN-m]': r'$DEL_{br,ew} [kN-m]$',
    'm10_del_blfw_avg_[kN-m]': r'$DEL_{br,fw} [kN-m]$',
    'm7_del_ttyaw_[kN-m]': r'$DEL_{tt,yaw} [kN-m]$',
    'm4_del_tbss_[kN-m]': r'$DEL_{tb,ss} [kN-m]$',
    'm4_del_tbfa_[kN-m]': r'$DEL_{tb,fa} [kN-m]$'
}