import subprocess
import platform

import paramiko
import re
import os
import threading
import time

import socket
import json
import time
import numpy as np

import pickle
import matplotlib.pyplot as plt

from wheelbot_dataset.recording.csvplot import plot_csv_preview
from wheelbot_dataset.recording.utils import next_log_number, continue_skip_abort
import uuid as uuid_module

class VideoRecorder:
    def __init__(self, camera_device="/dev/video0", output_path="log/video.mp4"):
        self.os_name = platform.system().lower()  # "darwin" (macOS) or "linux"
        self.output_path = output_path
        self.process = None
        self.running = False

        if self.os_name == "darwin":
            # Attempt to find the specific Intel RealSense RGB index
            detected_device = self._find_macos_realsense_rgb()
            if detected_device:
                self.camera_device = detected_device
            else:
                # Fallback to the original logic if detection fails
                self.camera_device = "0"
            print(f"[*] macOS detected. Selected camera device index: {self.camera_device}")
        else:
            self.camera_device = camera_device

    def _find_macos_realsense_rgb(self):
        """
        Parses ffmpeg output to find the index of the RealSense RGB camera.
        """
        try:
            # ffmpeg outputs device list to stderr
            cmd = ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
            result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, timeout=5)
            
            # Use regex to find the line containing 'RealSense' and 'RGB'
            # Looking for pattern: [x] ... RealSense ... RGB
            lines = result.stderr.splitlines()
            for line in lines:
                if "RealSense" in line and "RGB" in line:
                    match = re.search(r"\[(\d+)\]", line)
                    if match:
                        idx = match.group(1)
                        print(f"[+] Found RealSense RGB at index: {idx}")
                        return idx
            
            print("[!] Could not find RealSense RGB in device list.")
            return None
            
        except Exception as e:
            print(f"[!] Error during camera discovery: {e}")
            return None
        
    def start(self):
        """Start video recording in the background using ffmpeg."""
        if self.running:
            print("[!] Video recording is already running.")
            return
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        if self.os_name == "darwin":  # macOS AVFoundation
            # Most RealSense RGB cams require explicit pixel format
            cmd = [
                "ffmpeg",
                "-f",
                "avfoundation",
                "-framerate",
                "30",
                "-pixel_format",
                "uyvy422",
                "-video_size",
                "640x480",
                "-i",
                self.camera_device,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-y",
                self.output_path,
            ]
        else:
            # Start ffmpeg process to record video
            cmd = [
                "ffmpeg",
                "-f", "v4l2",
                "-fflags", "+genpts",
                "-framerate", "30",
                "-video_size", "640x480",
                "-i", self.camera_device,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-y",  # Overwrite output file if it exists
                "-reset_timestamps", "1",
                self.output_path
            ]
        
        try:
            print(f"[*] Starting video recording: {self.camera_device} -> {self.output_path}")
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.running = True
            print("[+] Video recording started.")
        except Exception as e:
            print(f"[!] Failed to start video recording: {e}")
            self.running = False
    
    def stop(self):
        """Stop video recording and save the file."""
        if not self.running or self.process is None:
            print("[!] Video recording is not running.")
            return
        
        print("[*] Stopping video recording...")
        
        try:
            # Send SIGINT to ffmpeg for graceful shutdown
            self.process.terminate()
            self.process.wait(timeout=10)
            print(f"[+] Video saved to {self.output_path}")
        except subprocess.TimeoutExpired:
            print("[!] Video recording did not stop gracefully, forcing kill...")
            self.process.kill()
            self.process.wait()
        except Exception as e:
            print(f"[!] Error stopping video recording: {e}")
        finally:
            self.running = False
            self.process = None

def run_experiment(velocity, roll, pitch, dt, host="192.168.10.102", locallogfile = "log/tmp.csv", video_device=None, yaw_delta=None):
    """
    Run an experiment on the wheelbot.
    
    Args:
        velocity: List of velocity setpoints
        roll: List of roll setpoints (in radians)
        pitch: List of pitch setpoints (in radians)
        dt: Time step between setpoints
        host: IP address of the wheelbot
        locallogfile: Path to save the local log file
        video_device: Video device for recording (optional)
        yaw_delta: List of yaw delta setpoints in degrees (optional, for yaw experiments).
                   If provided, the experiment will use AMPC controller instead of LQR.
    """
    is_yaw_experiment = yaw_delta is not None
    
    remotelogfile = "/tmp/tmp.csv"    
    controller = RemoteProgramController(
        host=host,
        remotelogfile=remotelogfile,
        locallogfile=locallogfile
    )
    if video_device is not None:
        video_recorder = VideoRecorder(camera_device=video_device, output_path=locallogfile.replace('.csv', '.mp4'))
    controller.start()
    if video_device is not None:
        video_recorder.start()
    time.sleep(3)
    
    data = {
    "yaw_delta": 0.0,
    "drive_delta": 0.0,
    "transition": None
    }

    def send_data(sock, data):
        json_data = json.dumps(data) + "\n"
        # print(f"Sending string: {json_data}")
        sock.sendall(json_data.encode('utf-8'))

    def receive_data(sock):
        try:
            received_data = sock.recv(1024).decode('utf-8')
            if received_data.strip():
                # print(f"Received string: {received_data.strip()}")
                odometry = json.loads(received_data)
                # print(f"Odometry data: {odometry}")
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
        except socket.error as e:
            print(f"Socket error: {e}")


    json_data = json.dumps(data) + "\n"
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, 8888))
    
    data["transition"] = "transition_to_standup"
    send_data(sock, data)
    receive_data(sock)
    time.sleep(4)
    
    experiment_failed = False
    try:
        # For yaw experiments, transition to AMPC controller first
        if is_yaw_experiment:
            print("Yaw experiment: waiting before AMPC transition...")
            time.sleep(3)
            print("Transitioning to AMPC controller...")
            data["transition"] = "transition_to_ampc"
            send_data(sock, data)
            receive_data(sock)
            time.sleep(3)
            data["transition"] = None
        
        print("Starting random signals")
        data["transition"] = None
        
        if is_yaw_experiment:
            # Yaw experiment: send yaw_delta along with velocity, roll, pitch
            for v, r, p, yd in zip(velocity, roll, pitch, yaw_delta):
                data["velocity"] = v
                data["roll"] = r
                data["pitch"] = p
                data["yaw_delta"] = float(yd)
                send_data(sock, data)
                receive_data(sock)
                time.sleep(dt)
        else:
            # Regular experiment: only send velocity, roll, pitch
            for v, r, p in zip(velocity, roll, pitch):
                data["velocity"] = v
                data["roll"] = r
                data["pitch"] = p
                send_data(sock, data)
                receive_data(sock)
                time.sleep(dt)
        
        print("Done sending random signals")
        data["velocity"] = 0
        data["roll"] = 0
        data["pitch"] = 0
        data["yaw_delta"] = 0.0
        send_data(sock, data)
        receive_data(sock)
        
        # For yaw experiments, transition back to LQR before laydown
        if is_yaw_experiment:
            print("Transitioning back to LQR controller...")
            data["transition"] = "transition_to_lqr"
            send_data(sock, data)
            receive_data(sock)
            time.sleep(3)
        
        time.sleep(5)
        
        data["transition"] = "transition_to_neg_roll_laydown"
        send_data(sock, data)
        receive_data(sock)
        time.sleep(2)
    except Exception as e:
        print(e)
        experiment_failed = True

    local_path = controller.stop()
    if video_device is not None:
        video_recorder.stop()

    print("Detected log path:", controller.log_path)
    output = controller.get_output()
    # print("Program output:")
    # print(output)
    print()
    print()
    print(f"Local log file: {local_path}")
    return experiment_failed

class RemoteProgramController:
    def __init__(self, host, remotelogfile, locallogfile):
        self.host = host
        self.user = "root"
        self.port = 2233
        self.command = f"cd /wheelbot-lib/build && ./Main -c /wheelbot-lib/config/default.json -o {remotelogfile}"
        
        self.ssh_client = None # Renamed to avoid confusion with self.ssh in _connect
        self.channel = None # Channel for the main program execution
        self.stdout_thread = None
        self.running = False
        
        self.remote_log_path = remotelogfile
        self.local_log_path = locallogfile
        self.output_buffer = []
        self.log_path = None # To store the path of the retrieved log

    # -------------------------
    # Internal helpers
    # -------------------------
    def _connect(self):
        """Create SSH client and connect using system keys for authentication."""
        if self.ssh_client is None or not self.ssh_client.get_transport() or not self.ssh_client.get_transport().is_active():
            print(f"[*] Connecting to {self.host}:{self.port} as {self.user}...")
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                self.ssh_client.connect(self.host, username=self.user, port=self.port)
                print("[+] SSH connection established.")
            except paramiko.AuthenticationException:
                raise Exception("SSH authentication failed. Check your keys or credentials.")
            except paramiko.SSHException as e:
                raise Exception(f"Could not establish SSH connection: {e}")
            except Exception as e:
                raise Exception(f"An unexpected error occurred during SSH connection: {e}")
        else:
            print("[*] SSH client already connected.")


    def _read_stdout(self):
        """Reads stdout from the running command asynchronously."""
        # Check if the channel is still open before trying to read
        while self.running and self.channel and self.channel.active:
            # Check for data readiness and exit status more frequently
            if self.channel.recv_ready():
                chunk = self.channel.recv(4096).decode("utf-8", errors="replace")
                self.output_buffer.append(chunk)
                # print(chunk, end="")  # optional: live printing of remote output
            elif self.channel.exit_status_ready():
                print(f"[*] Remote command finished with exit status: {self.channel.exit_status}")
                self.running = False # Mark as not running if the command finishes
                break # Exit the loop if the command has finished
            time.sleep(0.05) # Reduced sleep for faster response

        if self.running: # If loop exited but self.running is still True, something went wrong
             print("[!] stdout reading stopped unexpectedly.")
        self.running = False # Ensure running is false after loop exits


    # -------------------------
    # Public API
    # -------------------------
    def start(self):
        """Starts the remote process via SSH and begins capturing stdout."""
        if self.running:
            print("[!] Program is already running.")
            return

        self._connect() # Ensure connection is active

        # Open a new channel for the main command
        self.transport = self.ssh_client.get_transport()
        self.channel = self.transport.open_session()
        self.channel.set_combine_stderr(True) # Combine stderr with stdout for easier capture

        # Start the program *without blocking*
        print(f"[*] Executing remote command: {self.command}")
        self.channel.exec_command(self.command)
        self.running = True
        
        # Start thread to read stdout
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stdout_thread.start()

        print("[+] Program started on remote machine.")

    def stop(self):
        """Stops the remote process and retrieves the .log file if detected."""
        if not self.running and (self.channel is None or not self.channel.active):
            print("[!] Program is not running or channel is already closed.")
            # Even if not running, we might still want to try to retrieve logs if a connection exists
            if self.ssh_client and self.ssh_client.get_transport() and self.ssh_client.get_transport().is_active():
                self._retrieve_log_file()
            self._close_ssh_connections()
            return

        print("[*] Stopping remote program...")

        # If the main channel is still active, try to kill the process via that channel first
        # This is often unreliable for background processes that detach.
        # A more robust approach is to open a NEW channel for the pkill command.
        if self.ssh_client and self.ssh_client.get_transport() and self.ssh_client.get_transport().is_active():
            try:
                kill_channel = self.ssh_client.get_transport().open_session()
                kill_command = "pkill -f '{}'".format(self.command.split(' ')[-1]) # Be careful with pkill arguments!
                                                                                   # Using just the executable name is safer.
                print(f"[*] Sending kill command: {kill_command}")
                kill_channel.exec_command(kill_command)
                kill_channel.close()
            except paramiko.SSHException as e:
                print(f"[!] Warning: Could not open new channel to send kill command: {e}")
            except Exception as e:
                print(f"[!] Warning: An error occurred while trying to send kill command: {e}")
        else:
            print("[!] Warning: SSH client is not active, cannot send kill command.")


        # Mark as not running
        self.running = False

        # Wait for stdout thread to finish
        if self.stdout_thread and self.stdout_thread.is_alive():
            print("[*] Waiting for stdout thread to finish...")
            self.stdout_thread.join(timeout=5) # Increased timeout
            if self.stdout_thread.is_alive():
                print("[!] Warning: stdout thread did not terminate within timeout.")


        # Close the main program channel if it's still open
        if self.channel and self.channel.active:
            self.channel.close()

        self._retrieve_log_file()
        self._close_ssh_connections()

        print("[+] Remote program stopped.")
        return

    def _retrieve_log_file(self):
        """Helper to retrieve the log file."""
        if not (self.ssh_client and self.ssh_client.get_transport() and self.ssh_client.get_transport().is_active()):
            print("[!] SSH connection not active, cannot retrieve log file.")
            self.log_path = None
            return

        print(f"[+] Retrieving log file. Remote path: {self.remote_log_path}, local path: {self.local_log_path}")
        sftp = None
        try:
            sftp = self.ssh_client.open_sftp()
            local_dir = os.path.dirname(self.local_log_path)
            if local_dir and not os.path.exists(local_dir):
                os.makedirs(local_dir, exist_ok=True)

            sftp.stat(self.remote_log_path)  # ensure remote file exists
            sftp.get(self.remote_log_path, self.local_log_path)
            print(f"[+] Log file copied to {self.local_log_path}")
            self.log_path = self.local_log_path
        except FileNotFoundError:
            print(f"[!] Remote log file not found: {self.remote_log_path}")
            self.log_path = None
        except Exception as e:
            print(f"[!] Failed to retrieve log file: {e}")
            self.log_path = None
        finally:
            if sftp:
                sftp.close()

    def _close_ssh_connections(self):
        """Helper to close all SSH related connections."""
        if self.ssh_client:
            print("[*] Closing SSH client.")
            self.ssh_client.close()
            self.ssh_client = None


    def get_output(self):
        """Returns all collected stdout as a single string and saves it to a local .log file."""
        output = "".join(self.output_buffer)
        
        # Save output to log file (replace .csv with .log for the output stream)
        log_output_path = self.local_log_path.replace('.csv', '.log')
        log_dir = os.path.dirname(log_output_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        try:
            with open(log_output_path, 'w') as f:
                f.write(output)
            print(f"[+] Local program output saved to {log_output_path}")
        except Exception as e:
            print(f"[!] Failed to save program output to file: {e}")
        
        return output
    

def plot_and_run_sequence(velocity, roll, pitch, time, dt, wheelbot_name, surface, video_device, path="excitation_sequences/test", yaw_delta=None):
    """
    Plot setpoints and run an experiment.
    
    Args:
        velocity: Velocity setpoints array
        roll: Roll setpoints array (in degrees, will be converted to radians)
        pitch: Pitch setpoints array (in degrees, will be converted to radians)
        time: Time array
        dt: Time step
        wheelbot_name: Name of the wheelbot to use
        surface: Surface type for metadata
        video_device: Video device for recording (optional)
        path: Path prefix for saving files
        yaw_delta: Yaw delta setpoints array in degrees (optional, for yaw experiments)
    """
    is_yaw_experiment = yaw_delta is not None
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Plot ------------------------------------------------------------------
    if is_yaw_experiment:
        plt.figure(figsize=(14, 12))
        n_subplots = 4
    else:
        plt.figure(figsize=(14, 9))
        n_subplots = 3

    plt.subplot(n_subplots, 1, 1)
    plt.plot(time, velocity)
    plt.title("Velocity Setpoint (m/s)")
    plt.ylabel("Velocity [m/s]")
    plt.grid(True)

    plt.subplot(n_subplots, 1, 2)
    plt.plot(time, roll)
    plt.title("Roll Setpoint (deg)")
    plt.ylabel("Roll [deg]")
    plt.grid(True)

    plt.subplot(n_subplots, 1, 3)
    plt.plot(time, pitch)
    plt.title("Pitch Setpoint (deg)")
    plt.ylabel("Pitch [deg]")
    if not is_yaw_experiment:
        plt.xlabel("Time [s]")
    plt.grid(True)
    
    if is_yaw_experiment:
        plt.subplot(n_subplots, 1, 4)
        plt.plot(time, yaw_delta)
        plt.title("Yaw Delta Setpoint (deg)")
        plt.ylabel("Yaw Delta [deg]")
        plt.xlabel("Time [s]")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{path}.setpoints.pdf')
    plt.show()

    roll = np.deg2rad(roll).tolist()
    pitch = np.deg2rad(pitch).tolist()
    
    # Save setpoints to pickle file
    data = {
        'velocity': velocity,
        'roll':  roll,
        'pitch': pitch,
        'time': time,
        'dt': dt
    }
    if is_yaw_experiment:
        yaw_delta = np.deg2rad(yaw_delta).tolist()
        data['yaw_delta'] = yaw_delta.tolist() if hasattr(yaw_delta, 'tolist') else list(yaw_delta)

    # Find next available number in excitation_sequences folder
    filename = f'{path}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        
    print(f"Setpoints saved to {filename}")
    
    # Map wheelbot names to IP addresses
    wheelbot_hosts = {
        "wheelbot-beta-1": "192.168.10.101",
        "wheelbot-beta-2": "192.168.10.102",
        "wheelbot-beta-3": "192.168.10.103",
        "wheelbot-beta-4": "192.168.10.104",
        "wheelbot-beta-5": "192.168.10.105",
    }
    
    host = wheelbot_hosts.get(wheelbot_name)  # Default to beta-2
    locallogfile = f"{path}.csv"
    experiment_failed = run_experiment(velocity, roll, pitch, dt, host, locallogfile=locallogfile, video_device=video_device, yaw_delta=yaw_delta)
    # Analyze the fetched remote stdout log (if present) before asking the user
    log_output_path = locallogfile.replace('.csv', '.log')
    try:
        if os.path.exists(log_output_path):
            with open(log_output_path, 'r', errors='replace') as f:
                log_text = f.read()

            # Look for Vicon-related messages
            # Count exact unique lines that contain ViconLogger and our target messages
            matched_lines = [line for line in log_text.splitlines() if 'ViconLogger' in line and ('GetFrame failed' in line or 'Data occluded for wheelbot' in line)]
            from collections import Counter
            line_counts = Counter(matched_lines)

            if line_counts:
                print('\n[!] Detected Vicon-related log lines in program output:')
                for line, cnt in line_counts.items():
                    print(f'  - Occurred {cnt}x: {line}')
            else:
                print('\n[+] No Vicon "GetFrame failed" or "Data occluded for wheelbot" messages found in log output.')

            # Also count occurrences of the generic phrases (in case formatting differs)
            getframe_count = len(re.findall(r'GetFrame failed\.', log_text))
            occluded_count = len(re.findall(r'Data occluded for wheelbot', log_text))
            if getframe_count or occluded_count:
                print(f"\nSummary counts: GetFrame failed.: {getframe_count}, Data occluded for wheelbot: {occluded_count}")

            # Check for the 'New connection established' line specifically from the ViconLogger
            # Require the [ViconLogger] tag in the same line and do not require the IP address
            new_conn_match = re.search(r"\[ViconLogger\].*New connection established", log_text)
            if new_conn_match:
                print(f"[+] Vicon connection messages found: {new_conn_match.group(0)}")
            else:
                print('\n\n*** BIG ERROR: VICON WAS NOT ACTIVE DURING THE EXPERIMENT (no "New connection established" line from [ViconLogger] found) ***\n')
        else:
            print(f"[!] Log output file not found: {log_output_path}")
    except Exception as e:
        print(f"[!] Error while analyzing log file: {e}")

    if not experiment_failed:
        experiment_success_user = input("Was the experiment successful? (yes/no, default is yes): ") or "yes"
        success = experiment_success_user.lower() == "yes"
    else:
        success = False
    
    surface_actual = input(f"What surface was this on? default is {surface}") or surface
    surface_actual = surface_actual.lower()
    
    meta_data = {
        'experiment_status': 'success' if success else 'failed',
        'wheelbot': wheelbot_name,
        'surface': surface_actual,
        'uuid': str(uuid_module.uuid4())
    }
    
    meta_filename = f'{path}.meta'
    with open(meta_filename, 'w') as f:
        json.dump(meta_data, f, indent=2)
        
    plot_csv_preview(locallogfile)


def plot_and_run_with_repeat(*args, **kwargs):
    while True:
        plot_and_run_sequence(*args, **kwargs)
        repeat = input(
            "Repeat this experiment with same parameters? (yes/y to repeat, any other key to continue): "
        ).strip().lower()
        if repeat not in ('yes', 'y'):
            break


if __name__=="__main__":
    # controller = RemoteProgramController(
    #     host="192.168.10.102",
    #     remotelogfile = "/tmp/tmp.csv",
    #     locallogfile = "log/tmp.csv"
    # )

    # controller.start()

    # time.sleep(3)

    # controller.stop()

    # print("Detected log path:", controller.log_path)
    # print("Program output:")
    # print(controller.get_output())
    
    
    rec = VideoRecorder("1", "test/video.mp4")
    rec.start()
    time.sleep(5)
    rec.stop()
    
    # run_experiment([0.1], [0.1], [0.1], 1, host="192.168.10.102", video_device="/dev/video4")
