import numpy as np
import pickle
import os

from wheelbot_dataset import (
    Dataset,
    to_prediction_dataset,
)
            
def export_dynamics_model_learning_dataset():
    # Load datasets
    ds_train = Dataset("../../data")
    ds_test = Dataset("../../data_test")

    # Implement some acausal filter
    import scipy.signal as signal
    lowpass_cutoff_hz=20
    def my_lowpass(df):
        b, a = signal.butter(4, 2*(lowpass_cutoff_hz)/1000, btype="low", analog=False)
        df2 = df.copy()
        for col in df.columns:
            if col in ["/tau_DR_command/drive_wheel", "/tau_DR_command/reaction_wheel"]:
                continue
            df2[col] = signal.filtfilt(b, a, df[col])
        return df2

    # Define the same filters to all experiments in a group or dataset
    dt = 0.01 # 0.006 #0.01 # 0.006
    cut_and_filter_fn = lambda exp: (
        exp
        .cut_by_condition(
            start_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() > 0,
            end_condition=lambda df: df['/tau_DR_command/reaction_wheel'].abs() == 0,
        )
        # .apply_filter(my_lowpass)
        .resample(dt=dt)
        .cut_time(start=2.0, end=2.0)
    )
       
    # Convert and export states, actions, nextstates for neural network training
    fields_states = [
        "/q_yrp/yaw","/q_yrp/roll","/q_yrp/pitch",
        "/dq_yrp/yaw_vel","/dq_yrp/roll_vel","/dq_yrp/pitch_vel",
        "/q_DR/drive_wheel","/q_DR/reaction_wheel",
        "/dq_DR/drive_wheel","/dq_DR/reaction_wheel"
    ]
    fields_actions = [
        "/tau_DR_command/drive_wheel","/tau_DR_command/reaction_wheel",
    ]
        
    print("Processing 1-step datasets...")
    train_states_1, train_actions_1, train_nextstates_1, _ = to_prediction_dataset(
        ds_train.map(lambda exp: exp.filter_by_metadata(experiment_status="success")).map(cut_and_filter_fn),
        fields_states=fields_states,
        fields_actions=fields_actions,
        N_future=1,
        skip_N=1
    )
    test_states_1, test_actions_1, test_nextstates_1, _ = to_prediction_dataset(
        ds_test.map(lambda exp: exp.filter_by_metadata(experiment_status="success")).map(cut_and_filter_fn),
        fields_states=fields_states,
        fields_actions=fields_actions,
        N_future=1,
        skip_N=1
    )

    with open("dataset/dataset_1_step.pkl", "wb") as f:
        pickle.dump({
            "train": {
                "states": train_states_1,
                "actions": train_actions_1,
                "nextstates": train_nextstates_1,
            },
            "test": {
                "states": test_states_1,
                "actions": test_actions_1,
                "nextstates": test_nextstates_1,
            },
            "states_labels": fields_states,
            "actions_labels": fields_actions,
            "dt": dt
        }, f)

    print("Processing 100-step datasets...")
    train_states, train_actions, train_nextstates, _ = to_prediction_dataset(
        ds_train.map(lambda exp: exp.filter_by_metadata(experiment_status="success")).map(cut_and_filter_fn),
        fields_states=fields_states,
        fields_actions=fields_actions,
        N_future=100, #int(1/dt),
        skip_N=100 #int(1/dt)
    )
    
    print("Processing testing data...")
    test_states, test_actions, test_nextstates, _ = to_prediction_dataset(
        ds_test.map(lambda exp: exp.filter_by_metadata(experiment_status="success")).map(cut_and_filter_fn),
        fields_states=fields_states,
        fields_actions=fields_actions,
        N_future=100, #int(1/dt),
        skip_N=100 #int(1/dt)
    )

    print(f"Train States shape: {train_states.shape}")
    print(f"Test States shape: {test_states.shape}")
    
    os.makedirs("dataset", exist_ok=True)
    with open("dataset/dataset_100_step.pkl", "wb") as f:
        pickle.dump({
            "train": {
                "states": train_states,
                "actions": train_actions,
                "nextstates": train_nextstates,
            },
            "test": {
                "states": test_states,
                "actions": test_actions,
                "nextstates": test_nextstates,
            },
            "states_labels": fields_states,
            "actions_labels": fields_actions,
            "dt": dt
        }, f)
        

if __name__=="__main__":
    export_dynamics_model_learning_dataset()