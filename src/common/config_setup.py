"""This file contains Config_settings class. It establishes the main settings used when running
the model. It takes these settings from the run_config.toml file. It has universal configurations
(e.g., configs that cut across modules and/or solve options) and module specific configs."""

###################################################################################################
# Setup

# Import packages
import pandas as pd
import numpy as np
import tomllib
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
import types
import argparse

# Constants
SHORT_TON_TO_METRIC_TON = 0.90718474

# Import python modules
from definitions import PROJECT_ROOT
from src.integrator.utilities import create_temporal_mapping
from src.common.utilities import make_dir


###################################################################################################
# Configuration Class


class Config_settings:
    """Generates the model settings that are used to solve. Settings include:  \n
    - Iterative Solve Config Settings \n
    - Spatial Config Settings \n
    - Temporal Config Settings \n
    - Electricity Config Settings \n
    - Other
    """

    def __init__(self, config_path: Path, args: argparse.Namespace | None = None, test=False):
        """Creates configuration object upon instantiation

        Parameters
        ----------
        config_path : Path
            Path to run_config.toml
        args : Namespace
            Parsed arguments fed to main.py or other parsed object
        test : bool, optional
            Used only for unit testing in unit_tests directory, by default False

        Raises
        ------
        ValueError
            No modules turned on; check run_config.toml
        ValueError
            sw_expansion: Must turn RM switch off if no expansion
        ValueError
            sw_expansion: Must turn learning switch off if no expansion

        """
        # __INIT__: Grab arguments namespace and set paths
        self.missing_checks = set()
        self.test = test
        self.args = args
        if not args:
            self.args = types.SimpleNamespace()
            self.args.op_mode = None
            self.args.debug = False
        self.PROJECT_ROOT = PROJECT_ROOT

        # __INIT__: Dump toml, sse args to set mode
        with open(config_path, 'rb') as src:
            config = tomllib.load(src)

        ############################################################################################
        # Universal Configs

        # __INIT__: Solve Mode
        self.default_mode = config['default_mode']
        if not self.args.op_mode:
            print('No mode arg passed, therefore...')
            self.selected_mode = self.default_mode
            print(f'using mode {self.default_mode} specified in run_config file')
        else:
            self.selected_mode = self.args.op_mode

        # __INIT__: Setting output paths
        # Setup the output directory and write out its name for other scripts to grab
        if test:
            OUTPUT_ROOT = Path(PROJECT_ROOT, 'unit_tests', 'test_logs')
        else:
            output_folder_name = f"{self.selected_mode}_{datetime.now().strftime('%Y_%m_%d_%H%Mh')}"
            OUTPUT_ROOT = Path(PROJECT_ROOT / 'output' / output_folder_name)
        self.OUTPUT_ROOT = OUTPUT_ROOT
        make_dir(self.OUTPUT_ROOT)

        #####
        ### __INIT__: Methods and Modules Configuration
        #####

        # __INIT__: Set modules from config
        self.electricity = config['electricity']
        self.hydrogen = config['hydrogen']
        self.residential = config['residential']

        # __INIT__: Redirects and raises based on conditions
        if (not any((self.electricity, self.hydrogen, self.residential))) and (
            self.selected_mode in ('unified-combo', 'gs-combo', 'standalone')
        ):
            raise ValueError('No modules turned on; check run_config.toml')

        # __INIT__: Single module case
        if [self.electricity, self.hydrogen, self.residential].count(True) == 1:
            print('Only one module is turned on; running standalone mode')
            self.run_method = 'run_standalone'

        # __INIT__: Combinations of Modules and Mode --> run guidance
        match self.selected_mode:
            case 'unified-combo':
                # No elec case
                if self.hydrogen and self.residential and not self.electricity:
                    print('not an available option, running default version')
                    print(
                        'running unified-combo with electricity, hydrogen, and residential modules'
                    )
                    self.electricity = True

                # else, assign method as gs and set options
                self.run_method = 'run_unified'
                self.method_options = {}
            case 'gs-combo':
                # No elec case
                if self.hydrogen and self.residential and not self.electricity:
                    print('not an available option, running default version')
                    print('running gs-combo with electricity, hydrogen, and residential modules')
                    self.electricity = True

                # else, assign method as gs and set options
                self.run_method = 'run_gs'
                self.method_options = {
                    'update_h2_price': self.hydrogen,
                    'update_elec_price': True,
                    'update_h2_demand': self.hydrogen,
                    'update_load': self.residential,
                    'update_elec_demand': False,
                }
            case 'standalone':
                self.run_method = 'run_standalone'
            case 'elec':
                self.run_method = 'run_elec_solo'
            case 'h2':
                self.run_method = 'run_h2_solo'
            case 'residential':
                self.run_method = 'run_residential_solo'
            case _:
                raise ValueError('Mode: Nonexistant mode specified')

        # __INIT__: Iterative Solve Configs
        self.tol = config['tol']
        self.force_10 = config['force_10']
        self.max_iter = config['max_iter']

        # __INIT__: Spatial Configs
        self.regions = config['regions']

        # __INIT__:  Temporal Configs
        self.sw_temporal = config.get('sw_temporal', 'default')
        self.cw_temporal = create_temporal_mapping(self.sw_temporal)

        # __INIT__:  Temporal Configs -
        self.sw_agg_years = int(config.get('sw_agg_years', 0))
        self.years = config['years']
        if self.sw_agg_years == 1:
            self.start_year = config['start_year']
        else:
            self.start_year = self.years[0]

        ############################################################################################
        # __INIT__: Carbon Policy Configs
        self._configure_carbon_policy(config)

        ############################################################################################
        # __INIT__:  Electricity Configs
        self.sw_trade = int(config.get('sw_trade', 0))
        self.sw_rm = int(config.get('sw_rm', 0))
        self.sw_ramp = int(config.get('sw_ramp', 0))
        self.sw_reserves = int(config.get('sw_reserves', 0))
        self.sw_learning = int(config.get('sw_learning', 0))
        self.sw_expansion = int(config.get('sw_expansion', 0))

        ############################################################################################
        # __INIT__: Residential Configs
        self.scale_load = config['scale_load']

        if not test:
            self.view_regions = config['view_regions']
            self.view_years = config['view_years']
            self.sensitivity = config['sensitivity']
            self.change_var = config['change_var']
            self.percent_change = config['percent_change']
            self.complex = config['complex']

        ############################################################################################
        # __INIT__: Hydrogen Configs
        self.h2_data_folder = self.PROJECT_ROOT / config['h2_data_folder']

    def _configure_carbon_policy(self, config: dict):
        """Parse carbon policy configuration data into normalized attributes."""

        # Establish safe defaults so downstream code can rely on attributes existing.
        self.carbon_cap_groups = OrderedDict()
        self.default_cap_group = None
        self.carbon_cap = None
        self.carbon_allowance_procurement = {}
        self.carbon_allowance_procurement_overrides = {}
        self.carbon_allowance_start_bank = 0.0
        self.carbon_allowance_bank_enabled = True
        self.carbon_allowance_allow_borrowing = False

        carbon_policy_section = config.get('carbon_policy')
        if not isinstance(carbon_policy_section, dict):
            carbon_policy_section = {}

        def _get_policy_value(*keys, default=None):
            for key in keys:
                if key in carbon_policy_section:
                    value = carbon_policy_section[key]
                    if value not in (None, ''):
                        return value
            for key in keys:
                if key in config:
                    value = config[key]
                    if value not in (None, ''):
                        return value
            return default

        legacy_cap_value = self._normalize_carbon_cap(
            _get_policy_value('carbon_cap')
        )
        legacy_allowances = (
            self._normalize_year_value_schedule(
                _get_policy_value('carbon_allowance_procurement')
            )
            or {}
        )
        legacy_prices = (
            self._normalize_year_value_schedule(
                _get_policy_value('carbon_price')
            )
            or {}
        )
        legacy_start_bank = self._normalize_float(
            _get_policy_value('carbon_allowance_start_bank', 'start_bank'),
            default=0.0,
        )
        legacy_bank_enabled = self._normalize_bool(
            _get_policy_value('carbon_allowance_bank_enabled', 'bank_enabled'),
            default=True,
        )
        legacy_allow_borrowing = self._normalize_bool(
            _get_policy_value(
                'carbon_allowance_allow_borrowing', 'allow_borrowing'
            ),
            default=False,
        )
        legacy_overrides = (
            self._normalize_year_value_schedule(
                _get_policy_value('carbon_allowance_procurement_overrides')
            )
            or {}
        )

        default_regions = tuple(int(region) for region in self.regions)

        known_dict_keys = {
            'allowances',
            'allowance_procurement',
            'carbon_allowance_procurement',
            'allowance_procurement_overrides',
            'prices',
            'price',
            'carbon_price',
            'cap_schedule',
            'caps',
            'carbon_caps',
        }

        def _parse_group_entries(groups):
            """Return ordered (name, config) pairs from raw TOML structures."""

            entries: list[tuple[str, dict]] = []
            if isinstance(groups, dict):
                for group_name, group_config in groups.items():
                    entries.append((str(group_name), group_config))
            elif isinstance(groups, list):
                for idx, group_entry in enumerate(groups, start=1):
                    if isinstance(group_entry, dict):
                        entry = dict(group_entry)
                        nested_groups: list[tuple[str, dict]] = []
                        for key, value in list(entry.items()):
                            if (
                                isinstance(value, dict)
                                and key not in known_dict_keys
                            ):
                                nested_groups.append((str(key), value))
                                entry.pop(key)
                        group_name = (
                            entry.get('name')
                            or entry.get('group')
                            or entry.get('id')
                            or f'group_{idx}'
                        )
                        entry.pop('name', None)
                        entry.pop('group', None)
                        entry.pop('id', None)
                        entries.append((str(group_name), entry))
                        entries.extend(nested_groups)
                    elif group_entry is not None:
                        entries.append((str(group_entry), {}))
            elif groups is not None:
                # Handle simple string/int names.
                entries.append((str(groups), {}))
            return entries

        combined_group_configs = OrderedDict()
        for source in (
            config.get('carbon_cap_groups'),
            carbon_policy_section.get('carbon_cap_groups'),
        ):
            for group_name, group_config in _parse_group_entries(source):
                existing = combined_group_configs.get(group_name, {})
                merged = dict(existing)
                if isinstance(group_config, dict):
                    merged.update(group_config)
                combined_group_configs[group_name] = merged

        if not combined_group_configs:
            combined_group_configs['default'] = {'regions': list(self.regions)}

        built_groups = OrderedDict()
        for group_name, raw_config in combined_group_configs.items():
            normalized = dict(self._normalize_single_cap_group(raw_config))
            normalized['name'] = group_name

            cap_value = normalized.get('cap', normalized.get('carbon_cap'))
            cap_float = None
            if isinstance(cap_value, str):
                lowered = cap_value.strip().lower()
                if lowered not in {'none', 'null', ''}:
                    try:
                        cap_float = float(cap_value)
                    except (TypeError, ValueError):
                        cap_float = None
            elif cap_value is not None:
                try:
                    cap_float = float(cap_value)
                except (TypeError, ValueError):
                    cap_float = None
            if cap_float is None:
                cap_float = legacy_cap_value
            normalized['cap'] = cap_float

            regions = normalized.get('regions')
            if regions:
                regions = tuple(self._normalize_regions(regions))
            else:
                regions = default_regions
            normalized['regions'] = regions

            allowance_schedule = (
                normalized.get('allowances')
                or normalized.get('allowance_procurement')
                or normalized.get('carbon_allowance_procurement')
            )
            allowances = (
                self._normalize_year_value_schedule(allowance_schedule)
                if isinstance(allowance_schedule, (dict, list))
                else allowance_schedule
            )
            if not allowances:
                allowances = dict(legacy_allowances)
            normalized['allowances'] = dict(allowances)
            normalized['allowance_procurement'] = dict(allowances)
            normalized['carbon_allowance_procurement'] = dict(allowances)

            price_schedule = (
                normalized.get('prices')
                or normalized.get('price')
                or normalized.get('carbon_price')
            )
            prices = (
                self._normalize_year_value_schedule(price_schedule)
                if isinstance(price_schedule, (dict, list))
                else price_schedule
            )
            if not prices:
                prices = dict(legacy_prices)
            normalized['prices'] = dict(prices)
            normalized['price'] = dict(prices)
            normalized['carbon_price'] = dict(prices)

            overrides_schedule = normalized.get('allowance_procurement_overrides')
            overrides = (
                self._normalize_year_value_schedule(overrides_schedule)
                if isinstance(overrides_schedule, (dict, list))
                else overrides_schedule
            )
            if not overrides:
                overrides = dict(legacy_overrides)
            normalized['allowance_procurement_overrides'] = dict(overrides)
            normalized['carbon_allowance_procurement_overrides'] = dict(overrides)

            normalized['start_bank'] = self._normalize_float(
                normalized.get('start_bank', normalized.get('carbon_allowance_start_bank')),
                default=legacy_start_bank,
            )
            normalized['bank_enabled'] = self._normalize_bool(
                normalized.get('bank_enabled', normalized.get('carbon_allowance_bank_enabled')),
                default=legacy_bank_enabled,
            )
            normalized['allow_borrowing'] = self._normalize_bool(
                normalized.get(
                    'allow_borrowing',
                    normalized.get('carbon_allowance_allow_borrowing'),
                ),
                default=legacy_allow_borrowing,
            )

            built_groups[group_name] = normalized

        self.carbon_cap_groups = OrderedDict(built_groups.items())

        if self.carbon_cap_groups:
            default_name, default_config = next(iter(self.carbon_cap_groups.items()))
            allowance_map = dict(
                default_config.get('allowance_procurement', {})
            )
            self.carbon_cap = default_config.get('cap')
            self.carbon_allowance_procurement = allowance_map
            self.carbon_allowance_procurement_overrides = dict(
                default_config.get('allowance_procurement_overrides', {})
            )
            self.carbon_allowance_start_bank = float(
                default_config.get('start_bank', 0.0)
            )
            self.carbon_allowance_bank_enabled = bool(
                default_config.get('bank_enabled', True)
            )
            self.carbon_allowance_allow_borrowing = bool(
                default_config.get('allow_borrowing', False)
            )
            self.default_cap_group = types.SimpleNamespace(
                name=default_name,
                cap=self.carbon_cap,
                regions=tuple(default_config.get('regions', default_regions)),
                allowance_procurement=self.carbon_allowance_procurement,
                prices=dict(default_config.get('prices', {})),
                start_bank=self.carbon_allowance_start_bank,
                bank_enabled=self.carbon_allowance_bank_enabled,
                allow_borrowing=self.carbon_allowance_allow_borrowing,
                allowance_procurement_overrides=self.carbon_allowance_procurement_overrides,
            )
        else:
            self.default_cap_group = None



    ################################################################################################
    # Set Attributes Update

    # Runs configuration checks when you set attributes
    def __setattr__(self, name, value):
        """Update to generic setattr function that includes checks for appropriate attribute values

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value
        """
        super().__setattr__(name, value)

        # __SETATTR__: dictionary of check methods and setting attributes to pass thru those methods
        check_dict = {
            '_check_regions': {'region'},
            '_check_res_settings': {'view_regions', 'view_years', 'regions', 'years'},
            '_check_int': {'max_iter', 'start_year', 'sw_learning', 'percent_change'},
            '_check_elec_expansion_settings': {'sw_expansion', 'sw_rm', 'sw_learning'},
            '_additional_year_settings': {'sw_agg_years', 'years', 'start_year'},
            '_check_true_false': {
                'electricity',
                'hydrogen',
                'residential',
                'force_10',
                'sensitivity',
                'complex',
                'carbon_allowance_bank_enabled',
                'carbon_allowance_allow_borrowing',
            },
            '_check_zero_one': {
                'sw_trade',
                'sw_expansion',
                'sw_rm',
                'sw_ramp',
                'sw_reserves',
                'sw_agg_years',
            },
        }

        # __SETATTR__: Create empty all_check_sets
        all_check_sets = set()

        # __SETATTR__: For each check method pass the setting attributes through
        for check_method in check_dict.keys():
            # __SETATTR__: Add checks to all_check_sets
            all_check_sets.union(check_dict[check_method])

            # __SETATTR__: If set value in check dictionary, run check method
            if name in check_dict[check_method]:
                method = getattr(self, check_method)
                method(name, value)

        # __SETATTR__: Create a list of all the items not being checked
        if name not in all_check_sets:
            self.missing_checks.add(name)

    ################################################################################################
    # Check Methods

    def _has_all_attributes(self, attrs: set):
        """Determines if all attributes within the set exist or not

        Parameters
        ----------
        attrs : set
            set of setting attributes

        Returns
        -------
        bool
            True or False
        """
        return all(hasattr(self, attr) for attr in attrs)

    def _additional_year_settings(self, name, value):
        """Checks year related settings to see if values are within expected ranges and updates
        other settings linked to years if years is changed.

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        # Years settings
        if hasattr(self, 'years'):
            if not isinstance(self.years, list):
                raise TypeError('years: must be a list')
            if not all(isinstance(year, int) for year in self.years):
                raise TypeError('years: only include years (integers) in year list')
            if min(self.years) < 1900:
                raise TypeError('years: should probably only include years after 1900')
            if max(self.years) > 2200:
                raise TypeError('years: should probably only include years before 2200')

        # Years related settings
        if self._has_all_attributes({'sw_agg_years', 'years', 'start_year'}):
            if self.sw_agg_years == 1:
                all_years_list = list(range(self.start_year, max(self.years) + 1))
            else:
                all_years_list = self.years
            solve_array = np.array(self.years)
            mapped_list = [solve_array[solve_array >= year].min() for year in all_years_list]
            self.year_map = pd.DataFrame({'year': all_years_list, 'Map_year': mapped_list})
            self.WeightYear = (
                self.year_map.groupby(['Map_year'])
                .agg('count')
                .reset_index()
                .rename(columns={'Map_year': 'year', 'year': 'WeightYear'})
            )

    def _normalize_carbon_cap_groups(self, groups_config):
        """Standardize carbon cap group settings into dictionaries."""

        normalized = {}
        if isinstance(groups_config, dict):
            items = groups_config.items()
        elif isinstance(groups_config, list):
            items = [(group_name, {}) for group_name in groups_config]
        else:
            items = []

        for group_name, group_config in items:
            normalized[str(group_name)] = self._normalize_single_cap_group(
                group_config
            )
        return normalized

    def _normalize_single_cap_group(self, group_config):
        if group_config is None:
            return {}
        if not isinstance(group_config, dict):
            return {'value': group_config}

        normalized = {}

        allowance_data = self._coalesce_group_value(
            group_config,
            'allowances',
            'allowance_procurement',
            'carbon_allowance_procurement',
        )
        if allowance_data is not None:
            allowances = self._normalize_year_value_schedule(allowance_data)
            if allowances:
                normalized['allowances'] = allowances

        price_data = self._coalesce_group_value(
            group_config,
            'prices',
            'price',
            'carbon_price',
        )
        if price_data is not None:
            prices = self._normalize_year_value_schedule(price_data)
            if prices:
                normalized['prices'] = prices

        if 'regions' in group_config:
            regions = self._normalize_regions(group_config['regions'])
            if regions:
                normalized['regions'] = regions

        for key, value in group_config.items():
            if key in {
                'allowances',
                'allowance_procurement',
                'carbon_allowance_procurement',
                'prices',
                'price',
                'carbon_price',
                'regions',
            }:
                continue
            normalized[key] = value

        return normalized

    def _normalize_caps_by_group(self, caps_config):
        """Normalize nested carbon cap schedules keyed by group name."""

        normalized = {}
        if isinstance(caps_config, dict):
            for group_name, schedule in caps_config.items():
                normalized_schedule = self._normalize_year_value_schedule(schedule)
                if normalized_schedule:
                    normalized[str(group_name)] = normalized_schedule
        return normalized

    def _normalize_year_value_schedule(self, schedule):
        """Convert schedules keyed by year into int/float dictionaries."""

        normalized = {}
        if isinstance(schedule, dict):
            iterator = schedule.items()
        elif isinstance(schedule, list):
            iterator = []
            for item in schedule:
                if isinstance(item, dict):
                    year = item.get('year')
                    value = item.get('value')
                    if value is None:
                        value = item.get('amount')
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    year, value = item
                else:
                    continue
                iterator.append((year, value))
        else:
            iterator = []

        for year, value in iterator:
            try:
                year_key = int(year)
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            normalized[year_key] = value_float
        return normalized

    def _normalize_carbon_cap(self, value):
        """Normalize carbon cap entries, handling disabled markers and units."""

        if value in (None, ''):
            return None
        if isinstance(value, str) and value.strip().lower() in {'none', 'null'}:
            return None
        try:
            return float(value) * SHORT_TON_TO_METRIC_TON
        except (TypeError, ValueError):
            return None

    def _normalize_float(self, value, default=0.0):
        """Coerce a value to float with a fallback default."""

        if value in (None, ''):
            return default
        if isinstance(value, str) and value.strip().lower() in {'none', 'null'}:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _normalize_bool(self, value, default=False):
        """Coerce an input to a boolean value."""

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {'true', 't', 'yes', 'y', '1', 'on'}:
                return True
            if lowered in {'false', 'f', 'no', 'n', '0', 'off'}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return default

    def _normalize_regions(self, regions):
        """Normalize region lists to integer identifiers."""

        if regions in (None, ''):
            return []
        normalized = []
        if isinstance(regions, str):
            candidates = [part.strip() for part in regions.split(',') if part.strip()]
        elif isinstance(regions, (list, tuple, set)):
            candidates = regions
        else:
            candidates = [regions]

        for region in candidates:
            try:
                normalized.append(int(region))
            except (TypeError, ValueError):
                continue
        return normalized

    def _coalesce_group_value(self, group_config, *keys):
        """Return the first value present in a group configuration for any key."""

        for key in keys:
            if key in group_config:
                return group_config[key]
        return None

    # TODO: no hard coded values! regions should be flexible, come up with a better check
    def _check_regions(self, name, value):
        """Checks to see if region is between the current default values of 1 and 25.

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        if not isinstance(value, list):
            raise TypeError('regions: must be a list')
        if min(value) < 1:
            raise ValueError('regions: Nonexistant region specified')
        if max(value) > 25:
            raise ValueError('regions: Nonexistant region specified')

    def _check_elec_expansion_settings(self, name, value):
        """Checks that switches for reserve margin and learning are on only if expansion is on.

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        # You cannot run with a reserve margin without expansion
        if self._has_all_attributes({'sw_expansion', 'sw_rm'}):
            if self.sw_expansion == 0:  # if expansion is off
                if self.sw_rm == 1:
                    raise ValueError('sw_expansion: Must turn RM switch off if no expansion')

        # You cannot run with learning without expansion
        if self._has_all_attributes({'sw_expansion', 'sw_learning'}):
            if self.sw_expansion == 0:  # if expansion is off
                if self.sw_learning == 1:
                    raise ValueError('sw_expansion: Must turn learning switch off if no expansion')

    def _check_true_false(self, name, value):
        """Checks if attribute is either true or false

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        if value is True:
            pass
        elif value is False:
            pass
        else:
            raise ValueError(f'{name}: Must be either true or false')

    def _check_int(self, name, value):
        """Checks if attribute is an integer

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        if not isinstance(value, int):
            raise ValueError(f'{name}: Must be an integer')

    def _check_zero_one(self, name, value):
        """Checks if attribute is either zero or one

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        if (value == 0) or (value == 1):
            pass
        else:
            raise ValueError(f'{name}: Must be either 0 or 1')

    def _check_res_settings(self, name, value):
        """Checks if view year or region settings are subsets of year or region

        Parameters
        ----------
        name : str
            attribute name
        value : _type_
            attribute value

        Raises
        ------
        TypeError
            Error
        """
        if self.residential:
            if hasattr(self, 'view_regions'):
                if not isinstance(self.view_regions, list):
                    raise ValueError(f'{name}: Must be a list')
                if not set(self.view_regions).issubset(self.regions):
                    raise ValueError('view_regions: Must be a subset of regions')
            if hasattr(self, 'view_years'):
                if not isinstance(self.view_years, list):
                    raise ValueError(f'{name}: Must be a list')
                if not set(self.view_years).issubset(self.years):
                    raise ValueError('view_years: Must be a subset of years')
