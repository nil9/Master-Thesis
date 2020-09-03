

@echo off
2>&1 >output.txt (
for /r "E:\Master-Thesis\TA2RDF2\TA2RDF\input" %%g in (*.xml) do (

rem echo %%g
 msxsl.exe "%%g" convertTransaction2RDFTriples.xsl -o "%%~dpng.nt"
 del "%%~dpng.xml"
)
)