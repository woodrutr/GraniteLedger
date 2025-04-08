# src.integrator.runner

A gathering of functions for running models solo

### Functions

| `getLogger`([name])                                                               | Return a logger with the specified name, creating it if necessary.              |
|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| `plot_price_distro`(OUTPUT_ROOT, price_records)                                   | cheap/quick analyisis and plot of the price records                             |
| `run_elec_model`(settings[, solve])                                               | build electricity model (and solve if solve=True) after passing in settings     |
| [`run_elec_solo`](#src.integrator.runner.run_elec_solo)([settings])               | Runs electricity model by itself as defined in settings                         |
| [`run_h2_solo`](#src.integrator.runner.run_h2_solo)([settings])                   | Runs hydrogen model by itself as defined in settings                            |
| `run_hydrogen_model`(settings)                                                    | run hydrogen model in standalone                                                |
| `run_residential`(settings)                                                       | This runs the residential model in stand-alone mode.                            |
| [`run_residential_solo`](#src.integrator.runner.run_residential_solo)([settings]) | Runs residential model by itself as defined in settings                         |
| [`run_standalone`](#src.integrator.runner.run_standalone)(settings)               | Runs standalone methods based on settings selections; running 1 or more modules |
| `value`(obj[, exception])                                                         | A utility function that returns the value of a Pyomo object or expression.      |

### Classes

| `Config_settings`(config_path[, args, test])   | Generates the model settings that are used to solve.   |
|------------------------------------------------|--------------------------------------------------------|
| `Path`(\*args, \*\*kwargs)                     | PurePath subclass that can make system calls.          |

### src.integrator.runner.run_elec_solo(settings: [Config_settings](src.common.config_setup.md#src.common.config_setup.Config_settings) | None = None)

Runs electricity model by itself as defined in settings

* **Parameters:**
  **settings** ([*Config_settings*](src.common.config_setup.md#src.common.config_setup.Config_settings)) – Contains configuration settings for which regions, years, and switches to run

### src.integrator.runner.run_h2_solo(settings: [Config_settings](src.common.config_setup.md#src.common.config_setup.Config_settings) | None = None)

Runs hydrogen model by itself as defined in settings

* **Parameters:**
  **settings** ([*Config_settings*](src.common.config_setup.md#src.common.config_setup.Config_settings)) – Contains configuration settings for which regions and years to run

### src.integrator.runner.run_residential_solo(settings: [Config_settings](src.common.config_setup.md#src.common.config_setup.Config_settings) | None = None)

Runs residential model by itself as defined in settings

* **Parameters:**
  **settings** ([*Config_settings*](src.common.config_setup.md#src.common.config_setup.Config_settings)) – Contains configuration settings for which regions and years to run

### src.integrator.runner.run_standalone(settings: [Config_settings](src.common.config_setup.md#src.common.config_setup.Config_settings))

Runs standalone methods based on settings selections; running 1 or more modules

* **Parameters:**
  **settings** ([*Config_settings*](src.common.config_setup.md#src.common.config_setup.Config_settings)) – Instance of config_settings containing run options, mode and settings
