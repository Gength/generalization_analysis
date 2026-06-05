import os
import gc
import subprocess
import pandas as pd
import pm4py

def create_temp_xes(csv_path, temp_xes_path, case_col, act_col, time_col):
    """
    Reads the CSV, converts it to an Event Log using zero-copy strategies, 
    and writes it to a temporary uncompressed .xes file.
    
    By encapsulating this in a function, we guarantee that all heavy local variables 
    (DataFrames, EventLogs) will be destroyed when the function goes out of scope.
    """
    print(f"🚀 [1/4] Loading CSV into memory: {csv_path}")
    df = pd.read_csv(
        csv_path,
        sep=";",
        encoding="ISO-8859-1",
        dtype=str
    )
    
    print("⏳ [2/4] Parsing timestamps in-place...")
    df[time_col] = pd.to_datetime(df[time_col], format='mixed')
    
    print("🔄 [3/4] Renaming columns (Zero-Copy Strategy)...")
    df.rename(columns={
        case_col: 'case:concept:name',
        act_col: 'concept:name',
        time_col: 'time:timestamp'
    }, inplace=True)
    
    print("🏗️ [4/4] Building pm4py Event Log and exporting to temporary XES...")
    log = pm4py.convert_to_event_log(df)
    
    # write the log to a temporary .xes file (uncompressed)
    pm4py.write_xes(log, temp_xes_path)
    print("✅ Temporary .xes file generated successfully.")
    
    # Function ends here. 'df' and 'log' will go out of scope and be automatically marked for destruction.

if __name__ == "__main__":
    # ================= Configuration Area =================
    CSV_FILE = "./BPI-Challenge_2016/BPI2016_Clicks_NOT_Logged_In.csv"
    
    # temporary uncompressed XES file path (must be on the same drive for gzip to work efficiently)
    TEMP_XES_FILE = "./BPI-Challenge_2016/BPI2016_Clicks_NOT_Logged_In.xes"
    # generated .xes.gz file will be placed in the same directory with this name (gzip will overwrite if it already exists)
    FINAL_GZ_FILE = "./BPI-Challenge_2016/BPI2016_Clicks_NOT_Logged_In.xes.gz"
    
    CASE_ID_COLUMN = "SessionID"      
    ACTIVITY_COLUMN = "PAGE_NAME"    
    TIMESTAMP_COLUMN = "TIMESTAMP"  
    # ======================================================

    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: CSV file '{CSV_FILE}' not found.")
        exit(1)

    try:
        # Step 1: Execute the memory-heavy conversion inside an isolated function scope.
        create_temp_xes(
            csv_path=CSV_FILE,
            temp_xes_path=TEMP_XES_FILE,
            case_col=CASE_ID_COLUMN,
            act_col=ACTIVITY_COLUMN,
            time_col=TIMESTAMP_COLUMN
        )
        
        # Step 2: Absolute Memory Purge.
        # The function has returned, destroying its local scope. 
        # We now run the garbage collector in the main thread to thoroughly sweep RAM.
        print("🧹 [Memory Cleared] Forcing deep garbage collection in main thread...")
        gc.collect()
        
        # Step 3: Offload to System Gzip.
        # Now Python is using almost 0 MB of RAM, so the system has plenty of resources.
        print(f"🗜️ [System] Invoking Linux gzip for maximum performance...")
        subprocess.run(["gzip", "-f", TEMP_XES_FILE], check=True)
        print(f"✨ Success! Final compressed file is ready: {FINAL_GZ_FILE}")
        
    except FileNotFoundError:
        print("❌ Error: 'gzip' command not found. Please ensure it is installed in your WSL environment.")
    except subprocess.CalledProcessError as e:
        print(f"❌ System compression failed: {e}")
    except Exception as e:
        print(f"❌ Execution failed: {e}")