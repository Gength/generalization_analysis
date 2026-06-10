#!/bin/bash
# Compile M4Runner.java (Anti-Alignment CLI wrapper)
CP="src/AntiAlignments/dist/AntiAlignments-20260609.jar"
for j in src/AntiAlignments/ivy/*.jar; do CP="$CP:$j"; done
CP="$CP:src/AutomataConformance/out/production/AutomataConformance"
for j in src/prom_workspace_link/dist/*.jar; do CP="$CP:$j"; done
for j in src/prom_workspace_link/lib/*.jar; do CP="$CP:$j"; done
CP="$CP:src/prom_workspace_link/packages/logfiltering-6.13.2/lib/fake-context-1.0.20180719.jar"
CP="$CP:/usr/lib/jvm/java-8-openjdk-amd64/lib/sa-jdi.jar"
mkdir -p build
javac -cp "$CP" -d build benchmark/bridges/M4Runner.java
echo "EXIT=$?"
