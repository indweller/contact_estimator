import rosbag
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from common import *

JOINTS = ['L_hip', 'L_hip2', 'L_thigh', 'L_calf', 'L_toe',
            'R_hip', 'R_hip2', 'R_thigh', 'R_calf', 'R_toe']
IMU = ['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
        'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z']

def plot_joint_states(joint_states, name=None):
    fig, ax = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(2):
        for j in range(5):
            ax[i, j].plot(joint_states[:, i * 5 + j])
            ax[i, j].set_title(JOINTS[i * 5 + j])
    plt.tight_layout()
    plt.savefig(f'joint_states_{name}.png')

def plot_imu(imu_data, name=None):
    plt.ion()
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    for i in range(2):
        for j in range(3):
            ax[i, j].plot(imu_data[:, i * 3 + j])
            ax[i, j].set_title(IMU[i * 3 + j])
    plt.tight_layout()
    plt.savefig(f'imu_{name}.png')

def plot_contacts_hist(contact_data, name=None):
    hist_data = {'1 1': 0, '1 0': 0, '0 1': 0, '0 0': 0}
    for i in range(contact_data.shape[0]):
        hist_data[f'{int(contact_data[i, 0])} {int(contact_data[i, 1])}'] += 1
    print(hist_data)
    fig, ax = plt.subplots()
    ax.bar(hist_data.keys(), hist_data.values())
    plt.savefig(f'contacts_hist_{name}.png')

def read_joint_states(msg):
    positions = np.array([pos for pos in msg.position])
    velocities = np.array([vel for vel in msg.velocity])
    joint_states = np.concatenate((positions, velocities))
    return joint_states

def read_imu(msg):
    imu_data = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                         msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
    return imu_data

def get_csv_data(csv_file=None, parser=None):
    data = pd.read_csv(csv_file, header=None, sep='\t')
    data = data.to_numpy()
    print(f"csv_file: {data.shape}")
    return data

def get_rosbag_data(bag_file=None, topic_name=None, parser=None):
    messages = []

    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            message_data = parser(msg)
            messages.append(message_data)

    messages = np.array(messages)
    print(f"topic: {topic_name}, messages: {messages.shape}")
    return messages

def upsample_array(arr, factor):
    arr = np.repeat(arr, factor, axis=0)
    return arr

def preprocess_data(config=None, instance=None):
    state_data = []
    TOPIC_PARSER = {
        '/hector_gazebo_drift/joint_states': read_joint_states,
        '/vectornav/IMU': read_imu
    }
    for topic in instance["topics"]:
        state = get_rosbag_data(bag_file=config["package_path"] + 'data/' + instance["bag_file"], 
                                        topic_name=topic,
                                        parser=TOPIC_PARSER[topic],
        )
        state_data.append(state)
    state_data = np.concatenate(state_data, axis=1)

    contact_data = get_csv_data(csv_file=config["package_path"] + 'data/' + instance["csv_file"], 
                                parser=None
    )
    
    state_data = state_data[instance["start_time"]:instance["end_time"], :]
    contact_data = contact_data[instance["start_time"]:instance["end_time"], :]

    min_len = min(state_data.shape[0], contact_data.shape[0])
    contact_data = contact_data[:min_len, :]

    print(f"state_data: {state_data.shape}, contact_data: {contact_data.shape}\n")

    return state_data, contact_data

def run(config):
    assert config["package_path"] is not None, "Please provide a package path"
    state_data = []
    contact_data = []
    for instance in config['dataset']:
        states, contacts = preprocess_data(config, instance)
        state_data.append(states)
        contact_data.append(contacts)
    state_data = np.concatenate(state_data)
    contact_data = np.concatenate(contact_data)

    print(f"Final joint_state_data: {state_data.shape}, contact_data: {contact_data.shape}")
    os.makedirs(config["package_path"] + 'data/processed/', exist_ok=True)
    np.save(config["package_path"] + f'data/processed/state_data_{config["data_version"]}.npy', state_data)
    np.save(config["package_path"] + f'data/processed/contact_data_{config["data_version"]}.npy', contact_data)

if __name__ == '__main__':
    config = get_config('data')
    run(config)