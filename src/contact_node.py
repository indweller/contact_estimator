import rospy
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Imu
import numpy as np
import torch
import os
import sys
package_path = os.path.dirname(os.path.abspath(__file__)) + '/'
package_path = '/'.join(package_path.split('/')[:-2]) + '/'
sys.path.append(package_path + 'src')
from contact_estimator.msg import ContactArray, Contact
from models import MLP

class ContactNode:
    def __init__(self):
        self.joint_state = None
        self.imu_data = None
        self.contact_state = ContactArray()
        self.contact_state.contacts = [Contact(0, 1), Contact(1, 1)]
        self.model = torch.load(os.path.join(package_path, 'logs', 'latest', 'model.pth'))
        self.model.to('cpu')
        self.joint_state_sub = rospy.Subscriber('/hector_gazebo_drift/joint_states', JointState, self.joint_state_cb)
        self.contact_pub = rospy.Publisher('/learning/contact_state', ContactArray, queue_size=1)
        self.rate = rospy.Rate(1000)
        self.prediction_contact = {0: np.array([0, 0]), 1: np.array([0, 1]), 2: np.array([1, 0]), 3: np.array([1, 1])}

    def joint_state_cb(self, msg):
        velocities = np.array([vel for vel in msg.velocity])
        efforts = np.array([eff for eff in msg.effort])
        self.joint_state = np.concatenate((velocities, efforts))

    def imu_cb(self, msg):
        self.imu_data = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                         msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

    def get_contact_state(self):
        if self.joint_state is not None:
            state = self.joint_state
            state = torch.tensor(state, dtype=torch.float32)
            output = self.model(state)
            prediction = output.argmax()
            contact_state = self.prediction_contact[prediction.item()]
            self.contact_state = ContactArray()
            self.contact_state.contacts = [Contact(0, contact_state[0]), Contact(1, contact_state[1])]

    def run(self):
        while not rospy.is_shutdown():
            self.get_contact_state()
            self.contact_pub.publish(self.contact_state) 
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('contact_node', anonymous=True)
    contact_node = ContactNode()
    contact_node.run()
