# src.integrator.progress_plot

A plotter that can be used for combined solves

### Functions

| [`plot_it`](#src.integrator.progress_plot.plot_it)(OUTPUT_ROOT[, h2_price_records, ...])           | cheap plotter of iterative progress                 |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| [`plot_price_distro`](#src.integrator.progress_plot.plot_price_distro)(OUTPUT_ROOT, price_records) | cheap/quick analyisis and plot of the price records |

### Classes

| `Path`(\*args, \*\*kwargs)                         | PurePath subclass that can make system calls.   |
|----------------------------------------------------|-------------------------------------------------|
| `datetime`(year, month, day[, hour[, minute[, ...) | The year, month and day arguments are required. |

### src.integrator.progress_plot.plot_it(OUTPUT_ROOT, h2_price_records=[], elec_price_records=[], h2_obj_records=[], elec_obj_records=[], h2_demand_records=[], elec_demand_records=[], load_records=[], elec_price_to_res_records=[])

cheap plotter of iterative progress

### src.integrator.progress_plot.plot_price_distro(OUTPUT_ROOT, price_records: list[float])

cheap/quick analyisis and plot of the price records
