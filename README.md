Place the bag files and the csv files in `data` folder. Specify the files, start and end times in `config/data.yaml`. The bag file should contain the following topics:
- /hector_gazebo_drift/joint_states
- /vectornav/IMU

The `config/train.yaml` file contains the NN training parameters and the `config/test.yaml` file contains the testing parameters. 

Running `python main.py` will create processed numpy files from the bag files and csv files, train the NN and test the NN. The model is saved in the `logs` folder.

After building the package, run `rosrun contact_estimator contact_node.py` to run the node. The node will publish the contact state of the robot in the topic `/contact_state`. It uses the latest model saved in the `logs` folder.