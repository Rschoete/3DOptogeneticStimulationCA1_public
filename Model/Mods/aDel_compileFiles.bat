echo Script to delete file .c
del ".\*.c" /s /f /q
echo Delete files .o
del ".\*.o" /s /f /q
echo Delete files .tmp
del ".\*.tmp" /s /f /q
echo Delete files .dll
del ".\*.dll" /s /f /q
echo Done!
PAUSE