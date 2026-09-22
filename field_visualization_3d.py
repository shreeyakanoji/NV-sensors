import numpy as np
import plotly.graph_objects as go

import axon_field_model as afm
from phantom_experiment import traveling_pulse_current



AXON_HALF_LENGTH_M = 3e-3
N_TIME_FRAMES = 40
TIME_RANGE_S = (-2e-3, 2e-3)
N_FIELD_RINGS = 5
RING_RADII_M = np.array([100e-6, 200e-6, 300e-6, 500e-6, 800e-6])
N_SENSORS = 5
SENSOR_STANDOFF_M = 500e-6


def circle_points(center_y, radius, axis_range=64):

  
    theta = np.linspace(0, 2 * np.pi, axis_range)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    y = np.full_like(theta, center_y)
    return x, y, z


def build_visualization(I_peak_A, label, out_path, velocity_mult=1.0):
    time_points = np.linspace(*TIME_RANGE_S, N_TIME_FRAMES)
    sensor_y = np.linspace(-AXON_HALF_LENGTH_M, AXON_HALF_LENGTH_M, N_SENSORS)

    # precompute axon geometry (static across frames)
    axon_y = np.linspace(-AXON_HALF_LENGTH_M, AXON_HALF_LENGTH_M, 100)

    frames = []
    slider_steps = []

    for frame_i, t in enumerate(time_points):
        
        I_along_axon = traveling_pulse_current(I_peak_A, axon_y, t, velocity_mult=velocity_mult)
        pulse_center_y = 2.0 * velocity_mult * t  

        frame_data = []


        current_nA = np.abs(I_along_axon) * 1e9
        frame_data.append(go.Scatter3d(
            x=np.zeros_like(axon_y), y=axon_y * 1000, z=np.zeros_like(axon_y),
            mode="lines+markers",
            line=dict(color="rgba(120,120,120,0.4)", width=4),
            marker=dict(size=4, color=current_nA, colorscale="Hot",
                        colorbar=dict(title="Local current (nA)", x=1.0) if frame_i == 0 else None,
                        cmin=0, cmax=max(current_nA.max(), 1e-6)),
            name="Axon (current-carrying)",
            hovertemplate="Current: %{marker.color:.3f} nA<extra></extra>",
        ))

       
        I_at_pulse = traveling_pulse_current(I_peak_A, np.array([pulse_center_y]), t, velocity_mult=velocity_mult)[0]
        for radius in RING_RADII_M:
            B_pT = afm.axon_field_pT(abs(I_at_pulse), radius)
            cx, cy, cz = circle_points(pulse_center_y * 1000, radius * 1e6)
            frame_data.append(go.Scatter3d(
                x=cx, y=cy, z=cz, mode="lines",
                line=dict(color=f"rgba(28,114,147,{min(1.0, B_pT/50+0.1):.2f})", width=3),
                name=f"Field ring r={radius*1e6:.0f}\u00b5m",
                hovertemplate=f"B \u2248 {B_pT:.2f} pT at r={radius*1e6:.0f}\u00b5m<extra></extra>",
                showlegend=(frame_i == 0),
            ))

       
        sensor_readings_pT = []
        for sy in sensor_y:
            I_local = traveling_pulse_current(I_peak_A, np.array([sy]), t, velocity_mult=velocity_mult)[0]
            B = afm.axon_field_pT(abs(I_local), SENSOR_STANDOFF_M)
            sensor_readings_pT.append(B)
        sensor_readings_pT = np.array(sensor_readings_pT)

        frame_data.append(go.Scatter3d(
            x=np.full(N_SENSORS, SENSOR_STANDOFF_M * 1e6), y=sensor_y * 1000, z=np.zeros(N_SENSORS),
            mode="markers",
            marker=dict(size=8 + sensor_readings_pT / (sensor_readings_pT.max() + 1e-9) * 10,
                        color=sensor_readings_pT, colorscale="Viridis",
                        cmin=0, cmax=max(sensor_readings_pT.max(), 1e-6),
                        colorbar=dict(title="Sensor reading (pT)", x=1.15) if frame_i == 0 else None),
            name="Sensor array",
            text=[f"{v:.3f} pT" for v in sensor_readings_pT],
            hovertemplate="Sensor reading: %{text}<extra></extra>",
        ))

        frames.append(go.Frame(data=frame_data, name=str(frame_i)))
        slider_steps.append(dict(
            method="animate",
            args=[[str(frame_i)], dict(mode="immediate", frame=dict(duration=80, redraw=True), transition=dict(duration=0))],
            label=f"{t*1000:.2f}ms",
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=f"3D field simulation \u2014 {label}",
            scene=dict(
                xaxis_title="Standoff distance (\u00b5m)",
                yaxis_title="Position along axon (mm)",
                zaxis_title="(\u00b5m)",
                aspectmode="manual",
                aspectratio=dict(x=1, y=2, z=1),
            ),
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
                ],
            )],
            sliders=[dict(steps=slider_steps, currentvalue=dict(prefix="Time: "))],
            width=1000, height=750,
        ),
    )

    fig.write_html(out_path)
    print(f"saved {out_path}")
    return sensor_readings_pT


if __name__ == "__main__":
    print("Building 3D visualization: mammalian axon (realistic, hard case)...")
    build_visualization(afm.I_MAMMALIAN_A, "Mammalian axon", "outputs_3d_field_mammalian.html")

    print("\nBuilding 3D visualization: giant axon (proven regime)...")
    build_visualization(afm.I_GIANT_A, "Giant axon (squid/worm)", "outputs_3d_field_giant.html")
