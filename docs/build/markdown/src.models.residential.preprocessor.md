# src.models.residential.preprocessor package

## Submodules

## src.models.residential.preprocessor.enduse_db module

## src.models.residential.preprocessor.enduse_demand module

## src.models.residential.preprocessor.generate_inputs module

This file contains the options to re-create the input files. It creates:
: - Load.csv: electricity demand for all model years (used in residential and electricity)
  - BaseElecPrice.csv: electricity prices for initial model year (used in residential only)

Uncomment out the functions at the end of this file in the “if \_\_name_\_ == ‘_\_main_\_’” statement
in order to generate new load or base electricity prices.

<!-- !! processed by numpydoc !! -->

### src.models.residential.preprocessor.generate_inputs.base_price()

Runs the electricity model with base price configuration settings and then
merges the electricity prices and temporal crosswalk data produced from the run
to generate base year electricity prices.

* **Returns:**
  pandas.core.frame.DataFrame
  : dataframe that contains base year electricity prices for all regions/hours

<!-- !! processed by numpydoc !! -->

### src.models.residential.preprocessor.generate_inputs.compare_load_method_results()

runs the two methods for developing future load estimates and then creates to review files.
review1 sums the hourly data up by region and year. review2 writes out the hourly data for the
final model year for all regions. The data is written out to csvs for user inspection.

<!-- !! processed by numpydoc !! -->

### src.models.residential.preprocessor.generate_inputs.scale_load()

Reads in BaseLoad.csv (load for all regions/hours for first year)
and LoadScalar.csv (a multiplier for all model years). Merges the
data and multiplies the load by the scalar to generate new load
estimates for all model years.

* **Returns:**
  pandas.core.frame.DataFrame
  : dataframe that contains load for all regions/years/hours

<!-- !! processed by numpydoc !! -->

### src.models.residential.preprocessor.generate_inputs.scale_load_with_enduses()

Reads in BaseLoad.csv (load for all regions/hours for first year), EnduseBaseShares.csv
(the shares of demand for each enduse in the base year) and EnduseScalar.csv (a multiplier
for all model years by enduse category). Merges the data and multiplies the load by the
adjusted enduse scalar and then sums up to new load estimates for all model years.

* **Returns:**
  pandas.core.frame.DataFrame
  : dataframe that contains load for all regions/years/hours

<!-- !! processed by numpydoc !! -->

## Module contents

<!-- !! processed by numpydoc !! -->
