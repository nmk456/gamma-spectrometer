for %%x in (back_panel front_panel driver_v1) do start "" /D "%%x" /WAIT /B "C:\Program Files\KiCad\9.0\bin\kicad-cli.exe" "jobset" "run" "--file" "../jlcpcb.kicad_jobset" "%%x.kicad_pro"
