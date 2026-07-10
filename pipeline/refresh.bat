@echo off
REM Weekly auto-refresh launcher for the scheduled task. Pulls latest nflverse
REM data, retrains the MTNN + rookie model, redeploys to Vercel.
cd /d C:\Users\jcdav\vector-gridiron
python pipeline\refresh.py
