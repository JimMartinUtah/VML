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
        background-color: #2c5f8a; color: white;
        padding: 10px 20px; display: flex; align-items: center;
    }
    /* Plots scale to container width */
    .shiny-plot-output img { max-width: 100%; height: auto !important; }
    /* Scrollable tables */
    .tbl-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; width: 100%; }
    /* Mobile adjustments */
    @media (max-width: 767px) {
        .app-banner { padding: 8px 12px; }
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
    ui.span("Virtual Muscle Lab", style="font-size:1.5em; font-weight:bold; color:white;"),
    ui.input_action_button(
        "info_btn", "\u24d8",
        style="background:none; border:1px solid rgba(255,255,255,0.6); border-radius:50%; color:white; font-size:1em; width:30px; height:30px; line-height:1; cursor:pointer; margin-left:12px; padding:0;"
    ),
    class_="app-banner"
)

with ui.sidebar(open="desktop"):
            ui.input_slider("onset", "Onset (% of cycle)", min=0, max=74, value=22),
            ui.input_slider("offset", "Offset (% of cycle)", min=1, max=99, value=66),
            ui.input_slider("excursion", "Excursion amplitude (mm)", min=1, max=50, value=20),
            ui.input_slider("cycle_freq", "Cycle frequency (Hz)", min=0.5, max=4.5, value=2.0),
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
    return ui.div(ui.HTML(html), class_="tbl-scroll")
