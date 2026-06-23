Windows Sphinx build
====================

Use build_html.bat inside each documentation directory, or run:

python -m sphinx -E -b html -d _build\doctrees . _build\html

Do not use the older command without -d on this Windows/Sphinx 9.1 setup, because it may try to write doctree files into _build\html\.doctrees before that hidden directory exists.

Documentation directories:
- ipmai_pid_simulator_sessions_constant_regimes 05\ipmai_sphinx_docs_windows_detailed
- Harmonic_Drive_Cycle_Monitor\harmonic_drive_cycle_monitor_sphinx_docs_windows_detailed
- Robotic_Cycle_Monitor\robotic_cycle_monitor_sphinx_docs_windows_detailed
