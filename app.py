import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from shiny import reactive
from shiny.express import input, output, render, ui, output_args

# Thelen muscle model with full calculations
def thelen_muscle(onoff, freq, excursion, L0, F0, Vx, af, tau_a, tau_d):
    try:
        onset = onoff[0] / 100
        offset = onoff[1] / 100
        excursion = excursion / 1000  # Convert mm to meters
        tau_a = tau_a / 1000  # Convert to seconds
        tau_d = tau_d / 1000  # Convert to seconds

        act_pct = 1
        onset_time = onset / freq
        offset_time = offset / freq
        dt = 0.001 / freq  # Time step
        V0 = Vx * L0
        penn0 = 0.087
        k_shape = 0.5
        w = L0 * np.sin(penn0)

        # Time and cycle percentage (125% of cycle)
        t = np.arange(0, 1.25 * (1 / freq) + dt, dt)
        cycle_pct = t * freq * 100

        # Position
        position = excursion * np.sin(2 * np.pi * freq * t)
        position_mm = position * 1000

        # Excitation and Activation Dynamics
        excitation = np.where((t >= onset_time) & (t <= offset_time), 1 * act_pct, 0)
        activation = np.zeros(len(t))

        for i in range(1, len(t)):
            u = excitation[i - 1]
            a = activation[i - 1]
            tau = tau_a if u >= a else tau_d
            # Exact solution to first-order ODE: prevents overshoot regardless of dt/tau ratio
            activation[i] = u + (a - u) * np.exp(-dt / tau)

        # Pennation and muscle length
        penn = np.arcsin(w / (position + L0))
        muscle_length = position / np.cos(penn0) + L0
        muscle_length_norm = muscle_length / L0

        # Velocity
        v = (-2 * np.pi * freq * excursion * np.cos(2 * np.pi * freq * t)) / np.cos(penn)
        v_norm = v / (V0 * ((Vx / 2) + (Vx / 2) * activation) / Vx)

        # Force-Length relationship
        fl_norm = np.exp(-((muscle_length_norm - 1) ** 2) / k_shape)

        # Force-Velocity relationship
        fv_norm = np.where(v > (V0 * ((Vx / 2) + (Vx / 2))), 0,
                           np.where(v_norm > 0, (1 - v_norm) / (1 + v_norm / af),
                                    (1.8 - (0.8 * (1 + v / V0)) / (1 - 7.56 * 0.21 * v / V0))))

        # Contractile element force
        force_ce = activation * F0 * fl_norm * fv_norm

        # Total force
        force_total = force_ce

        # Work and Power calculations
        work = force_total * v * dt
        power = force_total * v
        work_actual = np.sum(work)
        work_positive = np.sum(work[work > 0])
        work_negative = np.sum(work[work < 0])
        # Cycle-averaged mean power = net work per cycle / cycle period
        power_actual = work_actual * freq
        power_positive = work_positive * freq
        power_negative = work_negative * freq

        # Data to return
        sim_data = pd.DataFrame({
            't': t,
            'cycle_pct': cycle_pct,
            'position': position,
            'position_mm': position_mm,
            'velocity': v,
            'force_total': force_total,
            'work': work,
            'power': power,
            'excitation': excitation,
            'activation': activation
        })

        return {
            'sim_data': sim_data,
            'work_actual': work_actual,
            'work_positive': work_positive,
            'work_negative': work_negative,
            'power_actual': power_actual,
            'power_positive': power_positive,
            'power_negative': power_negative,
        }
    except Exception as e:
        print(f"Error in thelen_muscle: {e}")
        return None

# Optimization function: coordinate descent (alternates onset/offset until convergence)
# Uses work_actual (net work over full cycle) as objective to avoid the variable-window
# bias in power_actual, which unfairly favours later onset values.
def thelen_muscle_opt(freq, excursion, L0, F0, Vx, af, tau_a, tau_d):

    def score(r):
        return r['work_actual'] if r else -np.inf

    def best_offset_given_onset(onset, current_best=None):
        """Coarse+fine sweep of offset with onset fixed."""
        boff = None
        best = -np.inf
        for offset in range(onset + 5, 100, 5):
            r = thelen_muscle([onset, offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
            if score(r) > best:
                boff, best = offset, score(r)
        if boff is None:
            return current_best
        best = -np.inf
        fine = boff
        for offset in range(max(onset + 1, boff - 5), min(100, boff + 6)):
            r = thelen_muscle([onset, offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
            if score(r) > best:
                fine, best = offset, score(r)
        return fine

    def best_onset_given_offset(offset, current_best=None):
        """Coarse+fine sweep of onset with offset fixed."""
        bon = None
        best = -np.inf
        for onset in range(0, min(75, offset), 5):
            r = thelen_muscle([onset, offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
            if score(r) > best:
                bon, best = onset, score(r)
        if bon is None:
            return current_best
        best = -np.inf
        fine = bon
        for onset in range(max(0, bon - 5), min(offset, bon + 6)):
            r = thelen_muscle([onset, offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
            if score(r) > best:
                fine, best = onset, score(r)
        return fine

    # Initialise
    cur_onset = 25
    cur_offset = best_offset_given_onset(cur_onset)
    if cur_offset is None:
        return None, None, None, None

    # Coordinate descent – up to 5 rounds or until no change
    for _ in range(5):
        new_onset  = best_onset_given_offset(cur_offset,  cur_onset)
        new_offset = best_offset_given_onset(new_onset,   cur_offset)
        if new_onset == cur_onset and new_offset == cur_offset:
            break
        cur_onset, cur_offset = new_onset, new_offset

    opt = thelen_muscle([cur_onset, cur_offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
    if opt is not None:
        opt['best_onset']  = cur_onset
        opt['best_offset'] = cur_offset
    r_final = thelen_muscle([cur_onset, cur_offset], freq, excursion, L0, F0, Vx, af, tau_a, tau_d)
    max_power = r_final['power_actual'] if r_final else -np.inf
    return opt, cur_onset, cur_offset, max_power

# Define the run_simulation function with theoretical results
@reactive.calc
def run_simulation():
    # Simulated muscle parameters
    muscle_params = {
        "onoff": [input.onset(), input.offset()],
        "freq": input.cycle_freq(),
        "excursion": input.excursion(),
        "L0": input.length_optimal(),
        "F0": input.max_isometric_force(),
        "Vx": input.max_velocity(),
        "af": input.force_velocity_curvature(),
        "tau_a": input.activation_time(),
        "tau_d": input.deactivation_time(),
    }
    
    # Always compute slider-based simulation
    sim_results = thelen_muscle(**muscle_params)

    # Compute optimized results only when checkbox is checked
    opt_results = None
    if input.optimize():
        opt_results, _, _, _ = thelen_muscle_opt(
            input.cycle_freq(), input.excursion(), input.length_optimal(),
            input.max_isometric_force(), input.max_velocity(),
            input.force_velocity_curvature(), input.activation_time(),
            input.deactivation_time()
        )

    # Theoretical muscle parameters (instantaneous activation/deactivation, optimal timing)
    theoretical_params = muscle_params.copy()
    theoretical_params['onoff'] = [25, 75]  # Active only during shortening phase
    # Use near-zero time constants so exp(-dt/tau) → 0: effectively a step function
    theoretical_params['tau_a'] = 0.001  # ~0.001 ms → instantaneous activation
    theoretical_params['tau_d'] = 0.001  # ~0.001 ms → instantaneous deactivation
    
    # Calculate theoretical results with zero onset/offset and instantaneous activation/deactivation
    theoretical_results = thelen_muscle(**theoretical_params)
    
    return sim_results, theoretical_results, opt_results

# Persistent click position for Graphs2 scrubbing
_g2_xmax = reactive.Value(None)

# Info modal handler
@reactive.effect
@reactive.event(input.info_btn)
def show_info_modal():
    m = ui.modal(
        ui.p("Developed by Jim Martin, Jenna Link, Marc Klimstra"),
        title="About Virtual Muscle Lab",
        easy_close=True,
        footer=ui.modal_button("Close")
    )
    ui.modal_show(m)

# Define the UI layout
ui.tags.style("""
    body, .bslib-page-fill { padding-top: 52px !important; }
    .app-banner {
        position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
        background-color: #d32f2f; color: white;
        padding: 10px 20px; display: flex; align-items: center;
    }
    /* Plots scale to container width */
    .shiny-plot-output img { max-width: 100%; height: auto !important; }
    /* Scrollable tables */
    .tbl-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; width: 100%; }
    /* Mobile adjustments */
    @media (max-width: 767px) {
        .app-banner { padding: 8px 12px; }
        .app-banner img.banner-logo { height: 22px !important; margin-left: 8px !important; margin-right: 8px !important; }
        .app-banner span { font-size: 1.1em !important; }
        .shiny-input-container { width: 100% !important; }
        .irs--shiny .irs-line, .irs--shiny .irs-bar { width: 100% !important; }
        .nav-item a { padding: 6px 10px !important; font-size: 0.9em; }
    }
""")
ui.tags.script("""
    (function() {
        function _sendWidth() {
            if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                Shiny.setInputValue('window_width', window.innerWidth, {priority: 'event'});
            } else {
                setTimeout(_sendWidth, 150);
            }
        }
        _sendWidth();
        window.addEventListener('resize', function() {
            if (typeof Shiny !== 'undefined' && Shiny.setInputValue) {
                Shiny.setInputValue('window_width', window.innerWidth, {priority: 'event'});
            }
        });
    })();
""")
# Hidden pre-declared input so window_width always exists with a desktop default
ui.tags.div(
    ui.input_numeric("window_width", "", value=1200, min=100, max=5000),
    style="display:none; position:absolute;"
)
ui.div(
    ui.img(src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAcIAAABRCAYAAABBnaYcAAAgAElEQVR4Xu2dB3gURR/G3xBCCQlJSEBApIMJ0iH0SA+ICFJDQIMYIOKnNAFpgnSUqiBFmtKLQADpRVrAiHSI9B5CS2gBQUC+Z5bMuNlsu7vdu8tlfs/jQ/5zezOzmzjvzsw7M24vX758CQ6Hw+FwMihuXAg5HA6Hk5HhQsjhcDicDA0XQg6Hw+FkaLgQcjgcDidDw4WQw+FwOBkaLoQcDofDydBwIeRwOBxOhoYLIYfD4XAyNFwILWRXXDzM2oIgs3sm1ArMx2KzuPYoAfGPElhsNKVzBSJHZk8Wuyq3H/yNpOQnSEp+CjcAXtk84J09C7yzeyCXVzZXvW0Ox+XQLYT7Tieg5uCVLDaKQS0rY2R4NRbbSp2hq7Ar7jqLjSCTG/Bi+WfCz6Ej1mDrsaspnxjP2e8/QPF8viw2g8idvbD3RiyLjcQ3iw/2vr8OHpkyo8HwaGw/fo19ZgQdQkpiYfdQFtuDswn3EHMqAccuJyLuWhLOJNzDxVsPNIsuntcH1UvmRTXyX4m8qFg0N/uMw+E4D1wIdSAWwpW/n0frCRtTPjGeIa2DMSysKouNJvFJEt5e0xzPX75gaUYSUbItBlXsKfycXoXw8dPn2HDoEqIPXMDmI1dw5+GTlLuzDV/PLAirWQIfvh2Imnbo+XM4HH1wIdSBWAifPX+BfF3nIdGgxlFKsbw+ODflQxYbzZJzq/H1n+NYbDRrGy/Am77FhJ/TmxCSXt/UTcewNOZsyt2YR9E8OdGxTiA+e6csH0blcBwMF0IdiIWQMGDRfoyNPshiozn0bRgqFDFnGO2D7Z/iwO0jLDaS4NzlsbD+NBanFyFc9+dFDFy8HyeuJrG624tsHu6IalgafZpVQAF/L3sVy+FwRHAh1IFUCC/ffoDCn85nsdF82bwixn5Qg8VGceXhNTRc35bFRjOm6iC0LPIui51dCG/d/xvdZv2GVbEXWJ0dSe+m5TGhYy1HVoHDyZBwIdSBVAgJZppmSM/g6oyPWGwUM+PmY+KxGSw2EmKS+a3ZKnhmzs7SnFkIZ28/iT7zY3D/8T+svs5AleJ5sOKLd1AwwNsZqsPhZAi4EOpATgh/2X8ObSZuYrHR7B/VWnAbGknzTRE4de8ci41EbJKhOKMQkl5gpx+2YcPhy6yezgYx1RBn7LuVCjtb1exOv3ELMG7OWvSNbIZv+6rPnatd6xbYRvg3tGY5bJ4zOCVVHj35vDy1Qvh32YYYtOs9GYFF8uOvjd8JaWpEDZmJH5dvQ9vG1bFscm8hjeapRtWyxfH78jEsrtZ2AGKPyf+/TK5tXr8KosIaIpev/HD7lr1HMeeX7diw+zCSH7/yO3h5ZsNbxQvg7eBSwneLFfyv/aHlbZ49GKG1yglpeuothtSLQPKRe7Ziku4lo1C9bkLd/lgxBsFlXn3XLLgQ6kBOCM02zfRoUg6TO4Ww2FYuPLiMdzaEs9hoxCYZirMJ4eGLt9Fo5Fph/V96YE63evi4Xqn0UFXTUBMlKWrXihvtpRN7IqxJTRZL0ZMPFUKCd8UPhQb73JYpqcRDjvwhXZBw+16qxp3mSYVCjnKBhTFzeBSLqTARAfbxTr1mlwpkvty+2LNoRJo6zViyBd2GzRJ+Fn///sPHOHXx1dKz6UO74JPw/5YpyQkhSZNyJeGOcH+k7IL5Alg6gdxDq9BqaNR5pCC6l3dMVxRq+jsQvzCYCRdCHcgJIcFM00wen+y4MetjuLmRpdq2M+XEbEw9MZfFRiI1yVCcSQjXHLiAsEmb8fSZOctGzGLSR7XQ893yLM5oqImSFLVridiQxpcIlt5GWCkfglgIw3pOxPJN+2WvF0N6YUQEiEhc3/NKiAhyeWohJ0wU0ptq0nWU8Lm0J3n+yg0UD/1ceAYrv+8j+92xs1ajf5cWqZ6PWnli1J4dRet50d4g4Uj0uDRCbgZcCHWgJIRmm2Z2DmuB2qVeZ7G1kD0TiEnmanI8SzMSqUmG4ixCSFyhzb5Zz+qV3hjZrhoGtaqc3qptCHoaVoratURsiCj4eOfAlpijqkOkWvkQxKJ14Pg5VGkzII3ASVESALk8tdASJiIm/tU6CT8n/j6PiRq9t65tG6TqYWqhVR5F7dlRxGIs90ISlTJ8rJaH0XAh1IGSEBIaDo/GNoN3T6F80rA0pnetw2JrOZJ4AmFbu7LYSORMMhRnEMJjl++g6sAVePJP+uoJSlnUPRTtQ0qyOKOgp2GlqF1LxIYOFZZ/v6/QM1yqMESqlQ9BKlpyQ55SlIZQlfJUQ48wBb3TQxjqFF+jdm9q6CmPoDd/pZcCLZE0Cy6EOlATwhX7z6GtSaYZ3xxZcWdOJNzdM7E0axh1aDLmn1nOYiORM8lQHC2E15MeoWK/pbh5P33MCWqxd0SrDLcjjd6GlaB2rVhsxsxcjYGTFis2tnrzEROV0otR6mktUzHVKOWphh5houIsvobOD0qHTLXQUx5B7dmJEZthxC8GVCBH92qPAVEtUq42Hy6EOlATQrNNM5sGNUOj8gVZbClkWLRGdFMkPb3L0oxEziRDcbQQ1hq8EjGnzdtc3N4QN+nxie0z1MJ7vQ0rQe1aqdjQhl1uiNSSfCjinszDQwtYOqVR5EhhSFaugVfKUw0tYVKqj1iAiBguGt9D1xycVnkUtWcnhV5LDTG0zlpDzGbAhVAHakJI6L9wH75Zc4jFRtKpbhDmflqfxZYSe+sQInYo190WlEwyFEcKYb8FMRi39jCri6sQWu4NbB7c3FVuRxNrGla5a6ViQxpdpSFSS/IRQ4cipfkpzddRaJ5qrtEpX0WmGnJVEyZyb02jxgh1kbsHMqdZr+Mw4d4J5GWgRYMqqVyiUtTKE6P27KRIe4WfDpstvCxIHav2gAuhDrSE0EzTDDnaJ2leZ3hkdmdplvBl7AhEXzRnk3AlkwzFUUJI9gyt9ZXxJ6U4Cyt6N0br6sqNpithScOqdq2cgNEhUtIDObFuUhpDid58KDQ/qeWfDkfK9T4JNE81pAJEhUlt+YTSMC2BiBBxh05fsoUJIuk9dgsPTeMYJZghhAT6zMh9EOGWGzq2B1wIdaAlhAQzTTNr+r2LZsFFWKyXZ/8+R5VVjfD4ufFzZGomGYojhPDJP8/xZo9FuHLnIauHq/GaryfOff8BvLJncbVbS4MlDavatUoCRht4sWhYkw+B9vykc4+0DKWejlqeStA8lZAactRYtiEGc1f+JvTGCESMYpaMSiWGZgkhgc5lErTyNwsuhDrQI4RmmmbCa5bA4p6NWKyXbdd24397+7PYSNRMMhRHCGGf+XsxYZ05m4o7A9mzuOOr1lUwoEUlZ6iO6VjSsKpdqyQ24iFS2gjTHpwl+VDoXCAVPaW5OjFaecqhJEw0fbTMXKQWZMg0ot8UoWcm7dUqlSdF7XegBH3ejuoNErgQ6kCPEJppmsmWxR1Jc7sge9bMLE0PvfZ9hQ1XtrPYSNRMMhR7C+Hd5CfIHzUv3S+VUKJ9rZIYH1ET+fxysDRXx5KGVe1aNbGRDpH+eeK8sPDd0nwI0mFQpeFSMVp5yqEkTHRNo7RXqpctKYv+CeL6KJUnRe13oAQt01Inq5FwIdSBHiEkfLlwH741yTSzvHdjtLFgXujRs8eoteY9U4ZFtUwyFHsL4dBlsRj+ywFWvqtQqkAuzP20HqqW0DfU5UpQYVETEgq13ssNQWqJDW3oyRBpq5RtwOQac618CHS9IDHG1AwflGYtnxQ9eUpRE6aolKUcSnOSWsjVR608MVwIrWRQy8oYGV6NxbZSZ+gq7Ip7tV+eUegVQjNNMy2rFsXKPk1YrMWqi+sxIHYUi41EyyRDsbcQ+neahaTkp6x8e1A4t7fQQ0u4+wiXbhs7L+mXIyuGh1UVDu/NqIiHFtV6OOIhTjVnppLYiL9PBFCpV6OVD0EsyETEtZYD6MlTipowid2Ycp+rQZ+3tM5q5YnhQmglriSEBDMaf0Jm90yCe9Rbp0Eicmcv7L0Ry2Kj0GOSoZjxLDqElBROZpBi5hytlA9C3kRE7UA0LPcGS6Ocvn4XW45exdjVB3H97iOWbimd65fC2A414O+djaVlVKJSejhkDunXmQPSmEBI4622XICgR2zoMCZFLi89+SwTLZ4ndVJzbxL05ClFS5jEw71iQSN1q1y6WJpnSFB7jlrlUbgQWomrCeHyfWeFzZ3NYP5nDfBh7UAWK5H4JAlvr2mO5y+N31ZMj0mGYk8hbDpmHdYfMvdopaDX/bCge0NUKpqHpSlBtnSbtP6IcPK9JVQt/hpmRNVF+cKpd+7P6NBeFoEIjI/MiQlqgqNXbGiDT5CKAUFvPmInpJaDk+apto5Q6fQJNWGi6xrF90G/RwRSejoEvW+5YWg95RG4EFqJqwmhmaaZJhUKYf3A91isxJJzq/H1n+NYbCTRjX5CkJ++PS/tJYTkWQd8PJuVawYNyhTA2v5NkT2LZYalP8/fwjuj1uKOxt9DXl9PoQfYsY72i05GhZgqJsxbh32HT7O1b6RBD6kUhMjW9VUbaL0CJh0itVYIo0S9WK11cTRPNaRGEj3CJDa+0D1QSY+QLJW4cv02e4EgkOdYpmQhfNHpPdn89JRH4EJoJa4mhASzTDOZMrkJe4/6eakPl32w/VMcuG38EoKy/qWwoqF+wbGXEJp9SDIRwa1D3mexpRAxDO6vvNdrn/cq4Ou2VZAjmwdL43A49oO7RnWg1yxDMdM0M+uTuuhc/y0WS7ny8Jpw5JIZ6DXJUOwlhJ/N3oUfNh9n5RoJmaM7NbkDAnJqz4mq8dNvf6HTtNRLWd4ulR+zP6mHEvl8WRqHw7E/XAh1YKkQEuoPi8aOE8YaRQj1yxTANpXeycy4+Zh4bAaLjcISkwzFXkJYpvdinLiaxMo1EiOPP2o1fgNWxV5AoQBvfPdxCJoHF2WfcTgcx8GFUAfWCKGZphlycj3ZZkuO5psicOqe8tZL1mKJSYZiLyF0azOVlWkkr/lkx43ZkSy2lfikR5i17aQwDMrhcJwHLoQ6sEYIzTTNTI2sjf81LsNiyoUHl/HOhnAWG4klJhmKPYTwVPxdBPVcxMo0ki+bV8TYD2qwmMPhuCZcCHVgjRASzDoKqFZgPuwZ0YrFlCknZmPqibksNgpLTTIUewhh9B8X0GLcBlamkWwf8j7qlSnAYg6H45pwIdSBtUJopmlGOjxKDuAlJpmryfEszSgsNclQ7CGEP249iagff2NlGsnLFZb/zjkcTvqDC6EOrBVCglmmGbL58hfvVWDxkcQTCNvalcVGYY1JhmIPIZy68Rg+n7ublWkU+f1yIP7HV4epcjgc14YLoQ5sEcJlMWfRbrLxppnKRfPgwDf/LZMYdWgy5p9RXqtmLdaYZCj2EMKJ647gi/l7WZlGQXaRiZvcgcUcDsd14UKoA1uE0EzTzJXpHfFGgLcwLFojuimSnt5lnxmFNSYZij2EcNyaQ+i3cB8r0yjIFmeHx7VjMYfDcV24EOrAFiEkmGWaGRVeDQNbVkbsrUOI2GF9/ZSw1iRDsYcQfr/hKHrM28PKNIqCAV64PP0jFnM4HNdFtxDuP3MDNQb9wmKjSA9brBFsMU6cu3EfJT6XP53aFsoW9MfRCeH4MnYEoi9uZOlGYa1JhmIPIZy7Iw6R03ewMo0iR9bMSF74CYs5HI7rolsI/zh3E1UHqG80aw1kn8VxETVZbCs1B/2CfWdusNgIjGgUzTLNHJsYhojYdoYfwGuLSYZiDyE0c5/RxHmdkUtjX1cOh5P+0S2Ex68kouwXS1hsFB/XDcKcT+uz2FaCeizCqevGzpXlyZkdN+fYtsOIWTvNdGqcGzE51He3twZbTDIUewjhzpPxqPv1alamkfzavynerVSYxRwOxzXRLYRmDe+9H1wUq/vpP3ldizyRs3H7gbHGFLI35KXpHVlsLWacoP5T89sYnXkLOcyFpRmBLSYZij2E8MqdhyjU7WdWppH0a14R3/CdZTgcl0e3ECbcfYT8Xeex2CgK5/bGxWm2iwzhzoMnyB1pvblDiVIF/HByku1WeqNNM76eHoh/bywmeAdj4V3jhl1tNclQ7CGEhKzh0/DP839TSjWOgJzZED+jE7J4uLM0W3jx70vU+3o1WlYtJmyRl9k9E/uMw+E4Dt1C+PzFv/BoN43FRhI/sxPy58rBYmtZFXsercYbbxqpV7oAtg9VPvFBL2cT7qFk94UstpUuVT0wudBIxPtXQ9Nbxu0oY6tJhmIvISzdazFOXjPn9InpXergk9DSLLYF8ZrHonlyYkLHWni/Cj+BgsNxNLqFkJC/y1wk3HvMYqOYEFELvd8rz2JriZiyFQt2n2axUUTWK4XZ3eqx2BaMNM3sbHUYwe5bgMyeaPA8HxJf/MM+s5bs7tmwr8V6m0wyFHsJYefpOzBnRxwr10iIWebM9x3g723b8zh/4z6Cei3CM0nPtcabeTGza12ULujP0jgcjn2xSAirDViB2HM3WWwUBfy9hMXhbm7Wz3OdI72tHguh/270MzysKr5qHcxiW1gacxbhBuw0UzjAEyfrDWVxtH9tDLt1gcXWElbsfQwP7sdiW7CXEC7cfRofTtnKyjWa8JolsLhnIxZbyt3kp8LSIzUTV0TtQIztUB35/GwfGeFwOJZhkRC2n7wZS2LOsthIlI4W0kvtIauw+6/rLDaSZb0aoW2NEiy2lYBOs5GYbJuhZ2jdF+iXezyLH/mWRq3E+yy2FiNMMhR7CeH1pEd4Pcr4+WsxnzUugymRtVmsFzK33nTMrzh08TZLUyKbhzv6NKsgbJKQPUtmls55xfkrN/Dt7DU4euoSYo/9d+ZmYJH8KPtmIYzu3R7FCuZl6VKqtR2Q6ntKvDwlv0xs2YYYtOs9WSjvr43aTu2oITPx4/JtaNu4OpZN7i2kuQW2Ef5VKoOyZe9RzPllO/Yc/AsJt++x9Kpli6NcYGHUq1Yafjm90KjzSPaZHvpGNsO3fT9kMSHpXjJmLtuKNdv/wMlz15D8+FXblC+3L8qULIQvOr2H0FrlUq5Oi9pzJfVtXr8KosIaIpevF0un0O9unj1YtQxCWM+JWL5pP6YP7YJPwkNZuhFYJITfRB9E/0X7WWwkpBE4NiEcJfL5sjS9dPtxJ2ZsPcFiozk/NQJFX8vJYlsxwjQT13oNCmU6xWJCS/cgXPwnmcWWYpRJhmIvISRU/nIZDl7QFhtbaFa5CCZ2rIVieX1YmhrkEN5BS/Zb7GLO5+uJUe2ro1PdIJaW0ZmxZAu6DZslPAYvz2x4q/h/x2NdSbjDxKJr2waYOTwq5ZPU0EaXCJmPt/zB1oTfl49hP0vxrvihIBTntkxRFV1C/pAuQr3+WDEGwWWKC2laQkjEvkOf75iwSOtK04m49olshs9HzEn55D+omEm/S/ioRd1UInLg+DnU6ziMiR8RLsr9h49x6uKrzgVJ3/DjIFUxkyuP1peI6p5FI9I8M71CSMTav9qrTfD1vohYgkVCuO3YVTQcsYbFRuOdzQNTO9cWhon0kJT8BJHTdiD6gO1Dgkrk8sqKxHldWGwEti5FCSmWA5sqDWExZVdAHfS8eZ7FlmKUSYZiTyE06xQKOTqElBT+RkPLFWRplDPX7yHmdAJGrTyA8zcfsHRrKFcoAN99HILapV5naRmRMTNXY+CkxYIADoxqiQFRLdI8BtKD+mjAVEF4xD0wMXobXTVor0SuZyWG1If01ogAXN/zSsAJakJIGvtC9boJokSEZ9H4HmmEg1yzfOM+VCpdlImrFL33SUSwSpsBws/kBWJM7w5phE4szESAYpaMSnONWnmkvk26jhI+J/ckfclQ+64Y+jdAnqf05cIILBJCIjz+nYzrMSgREpgPozvUEA6glfLoyTNhCHTNgQv4eecpPHn2gn1mBo3LF8TGQc1YbBTERv/bSeucntMb3UKET9qhwGdeRVHl/jMWW4KRJhmKPYXQXn+bUsjyHzKvR4ZAL91+yNKNhKy1Jcdu6e2JuhKkIS4e+rlwS1qNpVhIlk7sibAmqXes0tvoqkHFQypwUpQEU00Iaf1Ca5bD5jmDWbql6L1P2mOV1lEOmqdcj1urPHFvLvH3eamEVOu7FFpX8nslw9NKLzvWYpEQEuwxBEXJmd0D5QvnBvHQPH/xEpduP0B80iP2uT2wde5SCVuOZ0poPQM5M8nPB/b3rIzNDy03NBlpkqHYUwgJH03dhp93pR4udhU83DPh00ZlMLRNFfh5ZXWV29KECoreho8OocoNn+ltdLWQG/KUojSEqiSEtAdJer2Xd0xP0+uyBD33SZ+TlqBTxC8k1ohZ0Ds9hGFW6TV6vkufDX1BIM+WYOtzEmOxEI5fexh9F8Sw2NWRngRvFNYez9SmvBd+KvkVi6WcCaiFsJuXWawXI00yFHsL4en4uwjsuYiV74r45ciKIW2C8b9GZeGR2fUX5FNBURMdKVRspCKkp9HVQ1SKCUaud0RQM9UoCSEVfKU8LUHPfTaKHIktMUcxuld72aFmOWi+UrOKnvLoy4P0Gj3fpc+G9vLNMM1YLIRX7ySjYLefWOzK1C6VHzuHtTTtFvvOj8H4dZaZZqKbn0fDrMqngLzM6ocaj73xBPp/rUabZCj2FkJC82/WY+2fF1Nq4LoMaR2MYWFVXfcGJb0QqXCoQRtX6fConkZXD7RepPf28FDauX41kVESQiWhsAY992lNef3GLcC4OWvTiLVWeWrPS+u7dFhV3HOlw9NyLxrWYrEQEpqN/RXrDl5isauyvHdjtKmu7y3UGizdaSbAOysuNB4Ddzf1ecCffEPwXaL+34/RJhmKI4Qw7moSyvdbmmbhuitBFvlfnhYBr+xZXOm20kCHxCxt8GjjKp370mp0LYEO9UnFVm0+jKAkhDTdkp6vEnruk5YnV0clqBBKTS9q5RERbBo1RnhW0t8HQe27BGqSkYovff7SXr+1WCWEO45fQ/3h0Sx2RQrl9saFqRHIRE7lNRFycgI5QUEP3Wu6Y8zro1msRJJfJdS/c4vFaphhkqE4QggJY1YfxMDF5izzcQYWdQ9F+xBjh7GdESqE0oZXCy0hVEP6HSVoAy2du6Rzb0qGFypASkIoTbcGLXEhWFOelhCSFxYfheUTUiGjaNWV9lylgqckkNZilRASKvRdiiOX7rDY1TByj0k1luw9g/bfkdMjtNnfYifKesSyWBG3zGiZqYSuNYVmmGQojhJCQpX+y3HgvL6XgfRE/TIFsG2I7fvepgfM6hHKNdgU6To7JWjPT2puoWUozV8pCRBNlzb41qAlLgRanpE9QiXU7kmtrmq/f/Hzlw63WoPVQhh79gaqDVSeq0rPlC6YC0e+bQd3O5wOQEwzAZFz8OCx+j6hgXk9cfDt/7ZU02Jzrjrof1t7TaEZJhmKI4UwPjEZb/VejPsazzU9QUYpDn8bBr8Mcliws84RUuhcIBU9tbkwChUgqRBaM2enhJq4UKwpz9I5Qpo+WmaulKL0XQI1xWih9NJhCVYLIaHdpM1Yts+cLdccyYGxbVG5WB67VeGLn/di4q9HVMsbXS8ZPQJ+YLEWerZcM8skQ3GkEBI2Hb6Md0avS6lN+sY7uwcOfhNm1c5L6RlndI1SpMOgSsOlYpSEkIqqVGSsQU1cKGqGHiVovlLhUSqPmlqkvWYxSt8V9/jEuwiJuZ+y8420h2oNNglh4sO/UarXYty6/zdLS+844jBWPaaZS20WILebZXupRmYti0OPlTd6NsskQ3G0EBKmbDyG7nbaccZMNg9uJruTjavjjOsIxVChJkOMNcMHya6VE6MkhOJ1fSfWTZIVDb0oiYsYe60jjEpZaqI0Z6pUVz0vFQSl9ZqWYpMQEmJOJSBkyEpTTn2wNyFB+bHz6xamG2TkqDN0FXbFyQtdw8CciC47iMV6ORxQGx/flN9+zkyTDMUZhJAw4pcDGLJMx9yqE0LMWqv7NEGz4CJOWDvzEQ+PShtLKaQXQXeWkbtWqdG1BfGaNj0L1JWEkECHK7Uafy303ictTzqXKgfNU+5atfKs/Z3Qumm5aKnQ2tqTtlkICTO3nsAnP+5kcXok6HU/7B7eSjiV3BGomWZ+evcm2uSwfO3mv575UekhWXSd1vlqpkmG4ixCSBi18k8MXvp7Ss3SD0affJIeob0XMkzm6L1GpYgXz5PeoFaDrCaE4r0/SQ9q2tDOsr0cIi4EpV6jkrhIEW+4Teo9xoS9Rgm0dyf3kiD3XTWTjBQ987J6MEQICel5xxmyh+P+ka2R28e83pEWSqaZHFkz48r73yPbS+v2sZyQszoW3k17ELCZJhmKMwkhQe1lwxlZ2L0hOoS86YxVsztUDAnSeSNLT59QarCthfZeCFpDdGpCSCAi0Kr7eEGcCFKHK71XtXuw5D7FYkgg820UOgdHIOlap0+olUfX/Ul7lHLftXTnGJqH3uvlMEwICXO2x+Hzubvw9z/mboRtJI3KFcT8zxsijwNFkNJn/l5MWJfaNNOxcjZMKzqMxZYS718NTW+lXqcY6FscaxrPZ7FZOJsQEmLP3kTbiRtx5Y720hJHUSBXDvzSpwmqlnjNUVVwSsjb/7cK5xG+HVwK/To3VxUhuUbXCKJShuf09GC0hJBAenwzZc4HJBBBKpQ/t+rZi5bep1J5pAcXUikIka3rq+ajpzzayyOIhzul3xWbZPT28OhLki2mGUOFkECOGOrw3Wb8cc6513CRI5/GR9RC14Zvsbo7GnKEz5s9UptmNjU/gpCs1m3OLZDZEw2e50Pii/96ml9X7ovw4vqcYrbgjEJISH7yDL1+2oPZ2+NSauo8NKlYCIu7h8InR8bZWJvDcTSGCyGB5PjDpmPC7h4Pn6hvB+YIWlUtJpx7mNeEzbRtpfaQVcIxU4T8vtlwpuEwkNM3bCHavzaG3XplmrGHSYbirEJIIcsrPp62HQn3HrM0R/GGvxdGhlfTfRYnh/n3LTwAAAIcSURBVMMxDlOEkEKOTBqz6k/8sPk4S3Mk5HxDcoxNg7JvOLIaqojnsfq/DXyV95uUT6xHvKbQHiYZirMLIYHMyfZbGIOZW0+m1Nq+5PfLga/bVkGXBs4zMsHhZDRMFULKzXuPMXn9Ucz7LQ43HbDmMLxmCXRvUg7VSsqPqTsb/p1mISn5KY602YESbgcMqV5L9yBhyzV7mGQo6UEIKdeTHuH7DUcxfctxPPjb/FGM0HJvILxmSXxUN4jVgcPhOAa7CKGY7cevYXXseeHfU9eVF3vbwms+2VG71OtoXKGQcHqEVzYP9ll6gOw0s+fEeeysknYBqrXsCqiDGU/d7GKSoaQnIRSz5sAFLNx9GusPXTLM+EXcv9VL5kW7miXRsmqxDHW4Lofj7NhdCMWQHWl+P3MDcdeS8Ff8XVy8eR+3HvwtHFZ7R+PAWl/PLPD3zobcObPjjQBvBOb3Q+DrfihfJAClCuRi16VHiGlm/foZiPLVv6WaFs+8imJDiZ52MclQ0qsQitkddx274uKx99R1HL+cqDmf6OOZBa/5eKKAvxfKFvIXtuqrUCR3uv+b5HBcGYcKIUeZf+8eBZ6rN7qWkikgWDiZwl4cvXQHdx89NbQ4IjJBBfxYbG/InCIZyXj89Dkr2jNrZqFeZFNsDoeT/uBCyOFwOJwMDRdCDofD4WRouBByOBwOJ0PDhZDD4XA4GRouhBwOh8PJ0Pwfzju4WPmKa5cAAAAASUVORK5CYII=", alt="University of Victoria", class_="banner-logo", style="height:34px; width:auto; margin-right:16px; flex-shrink:0;"),
    ui.span("Virtual Muscle Lab", style="font-size:1.5em; font-weight:bold; color:white;"),
    ui.input_action_button(
        "info_btn", "\u24d8",
        style="background:none; border:1px solid rgba(255,255,255,0.6); border-radius:50%; color:white; font-size:1em; width:30px; height:30px; line-height:1; cursor:pointer; margin-left:12px; padding:0;"
    ),
    ui.img(src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOEAAABLCAYAAACGNODyAAAYJElEQVR4Xu2dA5RdSRPHK2tlkTWztm0ja9u2bdv2bta2mbVt2/Zmme/8el7d1PRc9Zs78+7s179zcjLvDZ7uv6u61L1EZJBEIpGW0SuKMBJpLb32EBnUr8Ln8JCIHJTcao7dRGSZ5FbneVhEDkxuRSL1IoowEmkxUYQtYqaZZpKJJ55Yfvzxx9xn8PLLL8vff/8ts8wyi3z33Xfy5ZdfyjTTTOPu//bbb93PTDrppDLRRBM1fqONBx54wP2/0EILSa9e7Do68vjjj8sff/yR3I60hijCFnH44YfLEkssIb/99pt7BpNMMokT0muvvSZff/21u2/66aeXddddV+666y6ZddZZZcCAATLyyCPLhhtuKBdffHG7Z77rrrvKcccdJ99//73MOeec8s4777j7xxprLLn11ltljjnmkDfeeMOJGBA134sibD1RhC1i7bXXliuuuCJ59MMOO0z23XdfWX311eXaa69194033ngy+eSTy0MPsdMWefrpp2X22WeXMcYYI7GCyjjjjCOff/65PPbYYzLffPMl9wOC32effWT99deXSy+91N236qqrynXXXdf4iUgriSKsCSqUNdZYQ6655prUZ/Xkk086KzfKKKPITz/9lNwPKsIHH3xQFl544eR+UIFvsMEGcskllyT3R+pBFGFNKCNCtYR5IsRqsg+06N/eaKON5KKLLkruj9SDKMKaUEaETzzxhMw111xy9NFHy8CBA5P7YcQRR5TddttNHn30UZl//vmT++Hggw+WAw44QI455hi544473H2TTTaZnH/++Y2fiLSSKMKaECJCxPT7778n98MII4wgu+++u4t4zjvvvMn9oCK84YYb5KWXXpLhhx9eZphhBllmmSqzsZFmiSKsCWmBGR8VYZ8+fVwU1EKwhqhqWmDmoIMOkgMPPFDWW289ueyyy9x97C2feuqpxk9EWkkUYU048sgjZa+99sq1hIiGVEOaCMcee2z54osvUkWoVnbNNdeUq6++Ork/Ug+iCGsCOT5yfWuttZZcddVVqc/qhRdecEn+tBTFBBNMIB9//LGzlvPMM09yP+C+4qqSlrj++uuT+0cddVT54YcfktuR1tBrLZFBc5nHnlREeie3ivlVRNrSwm28KCL9k1vNsaaIzJ3cEplEREZObhXDbumt5JbISyJyYXKrnpx22mmy7bbb5kYwSbZPNdVULqmP4CzkE99++20nVBLxllNPPVW222472X777d3jwHDDDefyhueee27jpyKtokMXxTEiMntyq5hXRWT75FbXcBR7mORWMQhwq+RW/UE05513nquKIXhywgknyCOPPJI88SGGGEIWWGABF9kkqLL//vvLjTfeKK+88or7/phjjimbbbaZHHHEEa4CB5f22Wefde7pzDPP7KKg/G3K3wjcvPjii86tJcrK40VaSxRhDaBk7bPPPnMlZFgo0gevvsryNhjqRbGEgwYNktFGG82Vr3344Yfue6Qnevfu7UQHWEWCNNSlUp9K/lDL04YaaiiZcMIJZdxxx3X7x0jriSKMRFpMp0WIQ7RD4+sTG/9Xxaki8p6IHCkidt9ahHVHq35OZ4jI28mtSKTzdFqEuiccTkRuT+6thp0aQZVm94TDiMidyb3VQMPxc8mtSKTzRBEGEkUYqZoowkDKinDxxReXZZddtl264KOPPpLbbrutQx5wyy23lAUXXFDGH398d5sAzE033ZTUee6wQ5vDf8opp7j/syDPSApjyimndLeJnvbv31+eeeYZGXrooV1FDpUylLgBuUa6Lvi79Cgutthi7ZqD6UkkuX/33Xcn9ymkPIjYUiTw77//uppV/g5pFvsYadx+++0udwlbbLGFK7MjgGThb5KGIZpLysYv0/NZZZVVXAqG10/QiQZoGp+32WYb16NJQGq//fZzBfAjjTSS+x1ypPRonnzyye72888/74JZ/C7lfXnwWU0xxRSuCGK55ZZrVyr4888/yy677OJSRhYK788++2wZcsgh3W2CZbSWRREGUlaECqLTGk2ilu+++27jO+0ZffTR5ZNPPpF//vnHfVj8r1ByRulZWjOvDxc/F9IHH3zgLkguZsuee+4pRx2Fgy8dKmiGGWYYufzyy11SHzbZZBO58MLsDOuxxx7risZJj9hi8KWXXtoJDc4444ykAoi0yGqrreZExf0K0d1PP/3UvW4Wn5NOOsk1NG+99dYy7bTTyjfffOOKGO67777kdyz6/iAI8p68ZnKgpHpIzZCSUXi+PG+gp/PKK690X/NeIUhSNtpUDQhsttlmc8Li7ykUOiBwvs8itemmm7o0E9ByNvfcNtM9GBZURM/v8D7/8ssvUYShhIqQD27n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", alt="The University of Utah", class_="banner-logo", style="height:34px; width:auto; margin-left:auto; flex-shrink:0;"),
    class_="app-banner"
)

with ui.sidebar(open="desktop"):
            ui.input_slider("onset", "Onset (% of cycle)", min=0, max=74, value=22),
            ui.input_slider("offset", "Offset (% of cycle)", min=1, max=99, value=66),
            ui.input_slider("excursion", "Excursion amplitude (mm)", min=1, max=50, value=20),
            ui.input_slider("cycle_freq", "Cycle frequency (Hz)", min=0.5, max=5.0, value=2.0, step=0.5),
            ui.input_numeric("length_optimal", "Length optimal (m)", value=0.084),
            ui.input_numeric("max_isometric_force", "Max isometric force (N)", value=1871),
            ui.input_numeric("max_velocity", "Max velocity (fiber lengths/s)", value=10),
            ui.input_numeric("force_velocity_curvature", "Force-velocity curvature", value=0.30),
            ui.input_numeric("activation_time", "Activation time (ms)", value=10),
            ui.input_numeric("deactivation_time", "Deactivation time (ms)", value=40),
            ui.input_checkbox("optimize", "Optimize Onset/Offset", value=False)


with ui.card():
    with ui.navset_bar(title=""):
        with ui.nav_panel(title="Force,Velocity,Power"):
            ui.input_switch("show_workloop", "Show Workloop", value=False)
            @render.plot
            def combined_graphs():
                results = run_simulation()
                sim_results = results[0]
                theoretical_results = results[1]
                opt_results = results[2]
                if sim_results is None or theoretical_results is None:
                    return

                is_mobile = input.window_width() < 768

                cycle_pct_sim = sim_results['sim_data']['cycle_pct']
                cycle_pct_theoretical = theoretical_results['sim_data']['cycle_pct']

                if is_mobile:
                    fig, axes = plt.subplots(4, 1, figsize=(5, 13))
                    ax_f, ax_v, ax_p, ax_pw = axes
                    legend_kw = dict(fontsize=7, loc='upper right')
                    tick_kw = dict(labelsize=7)
                    xlabel_kw = dict(fontsize=7)
                    title_kw = dict(fontsize=8)
                else:
                    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
                    ax_f, ax_v, ax_p, ax_pw = axs[0,0], axs[0,1], axs[1,0], axs[1,1]
                    legend_kw = dict()
                    tick_kw = dict()
                    xlabel_kw = dict()
                    title_kw = dict()

                # Force
                ax_f.plot(cycle_pct_sim, sim_results['sim_data']['force_total'], label='Simulated')
                ax_f.plot(cycle_pct_theoretical, theoretical_results['sim_data']['force_total'], label='Theoretical', linestyle='--')
                if opt_results is not None:
                    ax_f.plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['force_total'], label='Optimized', linestyle=':', color='purple')
                ax_f.set_title("Force vs. % of Cycle", **title_kw)
                ax_f.set_xlabel("% of Cycle", **xlabel_kw)
                if not is_mobile:
                    ax_f.set_ylabel("Force (N)")
                ax_f.tick_params(**tick_kw)
                ax_f.legend(**legend_kw)

                # Velocity
                ax_v.plot(cycle_pct_sim, sim_results['sim_data']['velocity'], color="green", label='Simulated')
                ax_v.plot(cycle_pct_theoretical, theoretical_results['sim_data']['velocity'], color="lightgreen", linestyle='--', label='Theoretical')
                if opt_results is not None:
                    ax_v.plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['velocity'], label='Optimized', linestyle=':', color='purple')
                ax_v.set_title("Velocity vs. % of Cycle", **title_kw)
                ax_v.set_xlabel("% of Cycle", **xlabel_kw)
                if not is_mobile:
                    ax_v.set_ylabel("Velocity (m/s)")
                ax_v.tick_params(**tick_kw)
                ax_v.legend(**legend_kw)

                # Position
                ax_p.plot(cycle_pct_sim, sim_results['sim_data']['position_mm'], color="orange", label='Simulated')
                ax_p.plot(cycle_pct_theoretical, theoretical_results['sim_data']['position_mm'], color="darkorange", linestyle='--', label='Theoretical')
                if opt_results is not None:
                    ax_p.plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['position_mm'], label='Optimized', linestyle=':', color='purple')
                ax_p.set_title("Position vs. % of Cycle", **title_kw)
                ax_p.set_xlabel("% of Cycle", **xlabel_kw)
                if not is_mobile:
                    ax_p.set_ylabel("Position (mm)")
                ax_p.tick_params(**tick_kw)
                ax_p.legend(**legend_kw)

                # Power
                ax_pw.plot(cycle_pct_sim, sim_results['sim_data']['power'], color="red", label='Simulated')
                ax_pw.plot(cycle_pct_theoretical, theoretical_results['sim_data']['power'], color="lightcoral", linestyle='--', label='Theoretical')
                if opt_results is not None:
                    ax_pw.plot(opt_results['sim_data']['cycle_pct'], opt_results['sim_data']['power'], label='Optimized', linestyle=':', color='purple')
                ax_pw.set_title("Power vs. % of Cycle", **title_kw)
                ax_pw.set_xlabel("% of Cycle", **xlabel_kw)
                if not is_mobile:
                    ax_pw.set_ylabel("Power (W)")
                ax_pw.tick_params(**tick_kw)
                ax_pw.legend(**legend_kw)

                for ax in [ax_f, ax_v, ax_p, ax_pw]:
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))

                fig.tight_layout()
                return fig

            with ui.panel_conditional("input.show_workloop"):

                @render.plot
                def work_loop1():
                    results = run_simulation()
                    sim_results = results[0]
                    theoretical_results = results[1]
                    opt_results = results[2]
                    if sim_results is None or theoretical_results is None:
                        print("Simulation failed: one or both result sets are None")
                        return

                    fig, ax = plt.subplots()

                    # Extract force and position data for the work-loop graph
                    force_total_sim = sim_results['sim_data']['force_total']
                    position_sim = sim_results['sim_data']['position_mm']

                    force_total_theoretical = theoretical_results['sim_data']['force_total']
                    position_theoretical = theoretical_results['sim_data']['position_mm']

                    # Plot force vs. position (excursion)
                    ax.plot(position_sim, force_total_sim, label="Simulated Work Loop")
                    ax.plot(position_theoretical, force_total_theoretical, label="Theoretical Work Loop", linestyle='--')
                    if opt_results is not None:
                        ax.plot(opt_results['sim_data']['position_mm'], opt_results['sim_data']['force_total'], label="Optimized Work Loop", linestyle=':', color='purple')

                    ax.set_title("Work Loop (Force vs. Excursion)")
                    ax.set_xlabel("Excursion (mm)")
                    ax.set_ylabel("Force (N)")
                    ax.legend()

                    return fig

            @render.ui
            def workloop_metrics():
                r = run_simulation()
                sim = r[0]
                theo = r[1]
                opt = r[2]
                if sim is None or theo is None:
                    return ui.p("No results available")
                opt_col = [
                    round(opt['work_actual'], 4),
                    round(opt['work_positive'], 4),
                    round(opt['work_negative'], 4),
                    round(opt['power_actual'], 4),
                    round(opt['power_positive'], 4),
                    round(opt['power_negative'], 4),
                ] if opt is not None else ["\u2014", "\u2014", "\u2014", "\u2014", "\u2014", "\u2014"]
                headers = ["Metric", "Simulated", "Theoretical", "Optimized"]
                rows = [
                    ["Total Work (J)",     round(sim['work_actual'], 4),    round(theo['work_actual'], 4),    opt_col[0]],
                    ["Positive Work (J)",  round(sim['work_positive'], 4),  round(theo['work_positive'], 4),  opt_col[1]],
                    ["Negative Work (J)",  round(sim['work_negative'], 4),  round(theo['work_negative'], 4),  opt_col[2]],
                    ["Mean Power (W)",     round(sim['power_actual'], 4),   round(theo['power_actual'], 4),   opt_col[3]],
                    ["Positive Power (W)", round(sim['power_positive'], 4), round(theo['power_positive'], 4), opt_col[4]],
                    ["Negative Power (W)", round(sim['power_negative'], 4), round(theo['power_negative'], 4), opt_col[5]],
                ]
                th = "style='padding:8px 14px; border:1px solid #ccc; background:#f0f0f0; font-weight:bold; text-align:center; white-space:nowrap;'"
                td = "style='padding:8px 14px; border:1px solid #ccc; text-align:center;'"
                td_left = "style='padding:8px 14px; border:1px solid #ccc; text-align:left; white-space:nowrap;'"
                html = "<table style='border-collapse:collapse; width:auto; margin-top:1rem;'><thead><tr>"
                for h in headers:
                    html += f"<th {th}>{h}</th>"
                html += "</tr></thead><tbody>"
                for row in rows:
                    html += "<tr>"
                    html += f"<td {td_left}>{row[0]}</td>"
                    for cell in row[1:]:
                        html += f"<td {td}>{cell}</td>"
                    html += "</tr>"
                html += "</tbody></table>"
                return ui.div(ui.HTML(html), class_="tbl-scroll")

        with ui.nav_panel(title="Interactive Workloop"):

            @reactive.effect
            @reactive.event(input.g2_force_click)
            def _store_g2_click():
                try:
                    c = input.g2_force_click()
                    if c is not None:
                        _g2_xmax.set(c['x'])
                except Exception:
                    pass

            with ui.div(style="margin: 0 0 6px 0;"):
                @output_args(click=True, height="180px")
                @render.plot
                def g2_force():
                    results = run_simulation()
                    sim_results = results[0]
                    theoretical_results = results[1]
                    opt_results = results[2]
                    if sim_results is None or theoretical_results is None:
                        return
                    xmax = _g2_xmax()

                    sim_data = sim_results['sim_data']
                    theo_data = theoretical_results['sim_data']
                    full_xmax = float(sim_data['cycle_pct'].max())

                    if xmax is not None:
                        sim_mask = sim_data['cycle_pct'].values <= xmax
                        theo_mask = theo_data['cycle_pct'].values <= xmax
                    else:
                        sim_mask = np.ones(len(sim_data), dtype=bool)
                        theo_mask = np.ones(len(theo_data), dtype=bool)

                    fig, ax = plt.subplots(figsize=(8, 0.75))
                    ax.plot(sim_data['cycle_pct'][sim_mask], sim_data['force_total'][sim_mask], label='Simulated', color='blue')
                    ax.plot(theo_data['cycle_pct'][theo_mask], theo_data['force_total'][theo_mask], label='Theoretical', color='orange', linestyle='--')
                    if opt_results is not None:
                        opt_data = opt_results['sim_data']
                        opt_mask = opt_data['cycle_pct'].values <= xmax if xmax is not None else np.ones(len(opt_data), dtype=bool)
                        ax.plot(opt_data['cycle_pct'][opt_mask], opt_data['force_total'][opt_mask], label='Optimized', linestyle=':', color='purple')
                    ax.set_xlim(left=0, right=full_xmax)
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
                    ax.set_ylabel("Force (N)")
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:10.3g}"))
                    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.24, top=0.98)
                    return fig

            with ui.div(style="margin: 0 0 6px 0;"):
                @output_args(height="180px")
                @render.plot
                def g2_position():
                    results = run_simulation()
                    sim_results = results[0]
                    theoretical_results = results[1]
                    if sim_results is None or theoretical_results is None:
                        return
                    xmax = _g2_xmax()

                    sim_data = sim_results['sim_data']
                    theo_data = theoretical_results['sim_data']
                    full_xmax = float(sim_data['cycle_pct'].max())

                    if xmax is not None:
                        sim_mask = sim_data['cycle_pct'].values <= xmax
                        theo_mask = theo_data['cycle_pct'].values <= xmax
                    else:
                        sim_mask = np.ones(len(sim_data), dtype=bool)
                        theo_mask = np.ones(len(theo_data), dtype=bool)

                    fig, ax = plt.subplots(figsize=(8, 0.7))
                    ax.plot(sim_data['cycle_pct'][sim_mask], sim_data['position_mm'][sim_mask], label='Simulated', color='orange')
                    ax.plot(theo_data['cycle_pct'][theo_mask], theo_data['position_mm'][theo_mask], label='Theoretical', color='darkorange', linestyle='--')
                    ax.set_xlim(left=0, right=full_xmax)
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
                    ax.set_ylabel("Position (mm)")
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:10.3g}"))
                    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.24, top=0.98)
                    return fig

            with ui.div(style="margin: 0;"):
                @output_args(height="360px")
                @render.plot
                def g2_workloop():
                    results = run_simulation()
                    sim_results = results[0]
                    theoretical_results = results[1]
                    opt_results = results[2]
                    if sim_results is None or theoretical_results is None:
                        return
                    xmax = _g2_xmax()

                    sim_data = sim_results['sim_data']
                    theo_data = theoretical_results['sim_data']
                    if xmax is not None:
                        sim_mask = sim_data['cycle_pct'].values <= xmax
                        theo_mask = theo_data['cycle_pct'].values <= xmax
                    else:
                        sim_mask = np.ones(len(sim_data), dtype=bool)
                        theo_mask = np.ones(len(theo_data), dtype=bool)

                    # Fix axes to full data range so they don't rescale
                    full_pos = pd.concat([sim_data['position_mm'], theo_data['position_mm']])
                    full_force = pd.concat([sim_data['force_total'], theo_data['force_total']])
                    pos_margin = (full_pos.max() - full_pos.min()) * 0.05
                    force_margin = (full_force.max() - full_force.min()) * 0.05

                    fig, ax = plt.subplots(figsize=(8, 3.5))
                    ax.plot(sim_data['position_mm'][sim_mask], sim_data['force_total'][sim_mask], label='Simulated', color='blue')
                    ax.plot(theo_data['position_mm'][theo_mask], theo_data['force_total'][theo_mask], label='Theoretical', color='orange', linestyle='--')
                    if opt_results is not None:
                        opt_data = opt_results['sim_data']
                        opt_mask = opt_data['cycle_pct'].values <= xmax if xmax is not None else np.ones(len(opt_data), dtype=bool)
                        ax.plot(opt_data['position_mm'][opt_mask], opt_data['force_total'][opt_mask], label='Optimized', linestyle=':', color='purple')

                    # Add directional arrows along the visible trajectory; more appear as scrub progress increases.
                    def _add_path_arrows(x_vals, y_vals, color):
                        x = np.asarray(x_vals)
                        y = np.asarray(y_vals)
                        n = len(x)
                        if n < 2:
                            return

                        stride = 12
                        for i1 in range(stride, n, stride):
                            i0 = i1 - 1
                            ax.annotate(
                                "",
                                xy=(x[i1], y[i1]),
                                xytext=(x[i0], y[i0]),
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=10, shrinkA=0, shrinkB=0),
                                zorder=6,
                            )

                        # Always show an arrow at the tip of the currently visible trajectory.
                        ax.annotate(
                            "",
                            xy=(x[-1], y[-1]),
                            xytext=(x[-2], y[-2]),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=12, shrinkA=0, shrinkB=0),
                            zorder=7,
                        )

                    _add_path_arrows(sim_data['position_mm'][sim_mask], sim_data['force_total'][sim_mask], 'blue')
                    _add_path_arrows(theo_data['position_mm'][theo_mask], theo_data['force_total'][theo_mask], 'orange')
                    if opt_results is not None:
                        _add_path_arrows(opt_data['position_mm'][opt_mask], opt_data['force_total'][opt_mask], 'purple')
                    ax.set_xlim(full_pos.min() - pos_margin, full_pos.max() + pos_margin)
                    ax.set_ylim(full_force.min() - force_margin, full_force.max() + force_margin)
                    ax.set_xlabel("Excursion (mm)")
                    ax.set_ylabel("Force (N)")
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:10.3g}"))
                    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.16, top=0.98)
                    return fig

            ui.p(
                "Click on the Force graph to reveal data up to that point on all plots.",
                style="color:#666; font-size:0.85em; margin-top:4px; margin-bottom:0;",
            )

   # New function to render the optimized onset/offset table
@render.ui
def optimized_onset_offset():
    if not input.optimize():
        return ui.div()
    results = run_simulation()
    opt_results = results[2]
    if opt_results is None:
        return ui.div()
    th = "style='padding:8px 14px; border:1px solid #ccc; background:#f0f0f0; font-weight:bold; text-align:center; white-space:nowrap;'"
    td_left = "style='padding:8px 14px; border:1px solid #ccc; text-align:left; white-space:nowrap;'"
    td = "style='padding:8px 14px; border:1px solid #ccc; text-align:center;'"
    rows = [
        ["Optimized Onset (%)", opt_results['best_onset']],
        ["Optimized Offset (%)", opt_results['best_offset']],
    ]
    html = "<table style='border-collapse:collapse; width:auto; margin-top:1rem;'><thead><tr>"
    for h in ["Parameter", "Value"]:
        html += f"<th {th}>{h}</th>"
    html += "</tr></thead><tbody>"
    for row in rows:
        html += f"<tr><td {td_left}>{row[0]}</td><td {td}>{row[1]}</td></tr>"
    html += "</tbody></table>"
    return ui.div(ui.HTML(html), class_="tbl-