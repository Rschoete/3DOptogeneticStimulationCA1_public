REM dummy batch file to initialize multiSim.py and writeout inputfilename to collection file
REM used in parallelMultiSim.py
python runSimwInput.py %1
echo %1>> %2