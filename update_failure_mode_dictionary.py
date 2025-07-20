import pandas as pd

# New mapping: (Description, Code, Keyword)
new_modes = [
    ("Leakage", "Leakage", "leak,leakage,oil leak,gas leak,water leak,coolant leak,air leak,leaking,leaked,seepage,dripping,spillage"),
    ("Vibration", "Vibration", "vibration,vibrating,excessive vibration,abnormal vibration,oscillation,shaking,rattling"),
    ("Clearance/Alignment Failure", "Clearance/Alignment Failure", "misalignment,misaligned,alignment issue,alignment error,out of alignment,excessive clearance,improper clearance"),
    ("Deformation", "Deformation", "bent,deformed,warped,shrunk,dented,twisted,distorted,buckled"),
    ("Looseness", "Looseness", "loose,loosened,detached,disconnected,unsecured,slack,not tight"),
    ("Sticking", "Sticking", "stuck,jammed,seized,sticking,not moving,frozen,immobile"),
    ("Cavitation", "Cavitation", "cavitation,bubbles,implosion,vapor bubbles,pitting (cavitation context)"),
    ("Corrosion", "Corrosion", "corrosion,rust,oxidized,corroded,rusted,tarnished,pitted (corrosion context),scale,scaling"),
    ("Erosion", "Erosion", "erosion,eroded,abrasive wear,material loss (erosion context),surface loss (erosion context)"),
    ("Wear", "Wear", "wear,worn,abraded,thinning,scuffed,surface loss (wear context),pitting (wear context),scoring"),
    ("Breakage", "Breakage", "broken,breakage,fracture,snapped,shattered,split,ruptured"),
    ("Fatigue", "Fatigue", "fatigue,stress fracture,fatigue crack,cyclic failure,fatigue failure"),
    ("Overheating", "Overheating", "overheating,overheated,excessive heat,high temperature,thermal failure,heat damage"),
    ("Burst", "Burst", "burst,exploded,blown,ruptured (burst context),high pressure burst"),
    ("Control Failure", "Control Failure", "control failure,regulation failure,plc failure,logic failure,control system error,automation failure"),
    ("Faulty Signal/Indication/Alarm", "Faulty Signal/Indication/Alarm", "faulty signal,false alarm,alarm failure,signal error,indication error,incorrect reading,false positive,false negative"),
    ("Calibration", "Calibration", "calibration,calibrate,out of calibration,miscalibrated,recalibration needed"),
    ("Software/PLC Program Error", "Software/PLC Program Error", "software error,plc error,program error,logic error,code error,software fault,plc fault,program failure"),
    ("Power Supply", "Power Supply", "power supply failure,ups failure,voltage drop,electrical power loss,power outage,power interruption"),
    ("Short Circuiting", "Short Circuiting", "short circuit,shorted,electrical short,shorting"),
    ("Open Circuit", "Open Circuit", "open circuit,circuit open,broken circuit,open connection"),
    ("Earth/Isolation Fault", "Earth/Isolation Fault", "earth fault,ground fault,isolation fault,insulation failure,ground leakage"),
    ("Blockage/Plugged", "Blockage/Plugged", "blockage,blocked,plugged,clog,clogged,obstruction,jam (blockage context),restricted flow,blocked valve,blocked filter"),
    ("Contamination", "Contamination", "contamination,contaminated,foreign material,dirt ingress,debris,ingress,fouling"),
    ("Bearing", "Bearing", "bearing failure,bearing noise,seized bearing,worn bearing,bearing overheat,bearing play,bearing damage,bearing vibration"),
    ("Seal", "Seal", "seal failure,leaking seal,worn seal,damaged seal,broken seal,sealant failure,seal leak"),
    ("Overload", "Overload", "overload,overcurrent,excessive load,overburdened,overdraw,overamp,overcapacity,load exceeded"),
    ("Lubrication Failure", "Lubrication Failure", "lubrication failure,lubricant loss,oiling failure,greasing failure,dry running,lack of oil,lack of grease,insufficient lubrication,lube failure"),
    ("Sensor Failure", "Sensor Failure", "sensor failure,faulty sensor,sensor error,sensor malfunction,sensor misreading,transducer failure,probe failure,detector failure"),
    ("Actuator Failure", "Actuator Failure", "actuator failure,solenoid failure,servo failure,actuator stuck,actuator error,actuator malfunction"),
    ("Electrical Arcing", "Electrical Arcing", "arcing,arc,spark,sparking,flashover,electrical arc,arc fault"),
    ("Belt/Chain Failure", "Belt/Chain Failure", "belt failure,chain failure,snapped belt,broken chain,belt slip,chain slip,worn belt,worn chain,belt misalignment,chain misalignment"),
    ("Pump Failure", "Pump Failure", "pump failure,pump not working,pump stopped,pump jammed,pump leak,pump noise,pump cavitation,pump overheating"),
    ("Valve Failure", "Valve Failure", "valve failure,stuck valve,valve leak,valve jammed,valve not opening,valve not closing,valve blockage,valve seat damage"),
    ("Motor Failure", "Motor Failure", "motor failure,motor not running,motor stopped,burnt motor,motor noise,motor overload,motor jammed,motor overheating"),
    ("Gearbox Failure", "Gearbox Failure", "gearbox failure,gear box failure,gear failure,gear noise,stripped gear,worn gear,gearbox leak,gearbox overheating"),
    ("Hydraulic Failure", "Hydraulic Failure", "hydraulic failure,hydraulic leak,hydraulic pressure loss,hydraulic hose failure,hydraulic pump failure,hydraulic actuator failure"),
    ("Pneumatic Failure", "Pneumatic Failure", "pneumatic failure,air leak,pneumatic pressure loss,air hose failure,pneumatic actuator failure"),
    ("Fuse/Breaker Failure", "Fuse/Breaker Failure", "fuse failure,blown fuse,breaker failure,tripped breaker,circuit breaker failure,fuse blown"),
    ("Relay/Contactor Failure", "Relay/Contactor Failure", "relay failure,contactor failure,stuck relay,stuck contactor,relay malfunction,contactor malfunction"),
]

# Create DataFrame
new_df = pd.DataFrame(new_modes, columns=["Description", "Code", "Keyword"])

# Overwrite the Excel file
new_df.to_excel("failure_mode_dictionary_2.xlsx", index=False)

print("failure_mode_dictionary_2.xlsx has been updated with the new mapping.") 