Visualization Tools
===================

The project contains PySide6-based viewers for inspecting generated or measured runs.

Single-run viewer
-----------------

The single-run viewer loads one run folder and displays multiple variables from ``measured.csv``.

Features:

* one screen with multiple synchronized time plots,
* synchronized x-axis,
* independent y-axis for each variable,
* line-width setting,
* double-click zoom reset.

Independent y-axes are useful here because the displayed variables have different physical units and ranges.

Variable-comparison viewer
--------------------------

The variable-comparison viewer displays one selected variable across all regimes.

Features:

* one selected variable,
* one subplot per behavior type,
* synchronized x-axis,
* synchronized y-axis,
* overlay of multiple trials if available,
* line-width setting,
* double-click zoom reset.

Synchronized y-axes are useful here because each subplot shows the same physical quantity.

Practical usage
---------------

Use the single-run viewer for detailed inspection of one run.

Use the variable-comparison viewer for quick comparison of how one variable differs across regimes.
